import logging
import sys
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from sklearn.utils import compute_class_weight
from torch import optim, nn, Tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.datasets_and_organizers.fast_tensor_data_loader import FastTensorDataLoader
from src.drawer import Drawer
from src.losses_aggregator import LossesAggregator
from src.model_trainer import ModelTrainer, Params
from src.technical_learning_parameters import TechnicalLearningParameters
from src.models.dudi_basic.simple_classifier import SimpleClassifier
from src.models.dudi_basic.simple_vae_net import SimpleVaeNet
from src.os_utilities import create_dir_if_not_exists
from src.random_manager import RandomManager
from src.training_batch_loss import TrainingBatchLoss
from src.training_summary import TrainingSummary


@dataclass
class MnistModelParameters(TechnicalLearningParameters):
    lr: float
    batch_size: int
    latent_dim: int
    encode_dims: List[int]
    decode_dims: List[int]
    classifier_inner_dims: List[int]
    loss_coefs: Dict[str, float]
    scheduler_factor: float
    scheduler_patience: int
    scheduler_cooldown: int


class MnistModelTrainer(ModelTrainer[MnistModelParameters]):

    def __init__(self, logger: Logger, params: MnistModelParameters):
        super().__init__(logger, params)
        self.organized_data_path = './organized_data/mnist'
        create_dir_if_not_exists(self.params.working_directory)
        self.drawer = Drawer(logger, self.params.working_directory)

    def on_epoch_started(self, i_epoch):
        self.logger.info(f'Epoch {i_epoch} has started.')

    def on_epoch_finished(self, i_epoch):
        self.losses_aggregator.end_epoch()
        epoch_loss = self.losses_aggregator.last_epoch_loss
        self.scheduler.step(epoch_loss.total_loss)
        vae_only_loss = epoch_loss.loss_name_to_value['kld_loss'] + epoch_loss.loss_name_to_value['reconstruction_loss']
        self.logger.info(f'Losses {epoch_loss.loss_name_to_value} ; VAE = {vae_only_loss} ; Total = {epoch_loss.total_loss}')

    def on_training_started(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset_train = datasets.MNIST(self.organized_data_path, train=True, download=True,
                                  transform=transform)
        dataset_test = datasets.MNIST(self.organized_data_path, train=False,
                                  transform=transform)

        # all_training_data = torch.flatten(self.dataset_train.train_data, start_dim=-2).to(self.params.device)
        # all_training_labels = self.dataset_train.train_labels.to(self.params.device)


        all_data = [x for x in DataLoader(self.dataset_train, batch_size=320, shuffle=True)]
        all_training_data = []
        all_training_labels = []
        for batch_images, batch_labels in all_data:
            all_training_data.append(torch.flatten(batch_images, start_dim=-2).squeeze())
            all_training_labels.append(batch_labels)

        self.all_training_data = torch.vstack(all_training_data).to(self.params.device)
        self.all_training_labels = torch.hstack(all_training_labels).to(self.params.device)

        self.train_loader = FastTensorDataLoader({
            'samples': self.all_training_data,
            'labels': self.all_training_labels
        },
        batch_size=self.params.batch_size,
        shuffle=True)

        # train_loader = DataLoader(self.dataset_train, batch_size=300)

        self.vae = SimpleVaeNet(
            input_dim=28*28,
            encode_dims=self.params.encode_dims,
            latent_dim=self.params.latent_dim,
            decode_dims=self.params.decode_dims,
            decoder_skip_connection=False,
            layer_type=nn.Linear
        ).to(self.params.device)
        self.vae.init_weights()

        class_weights = compute_class_weight('balanced', classes=np.unique(self.all_training_labels.cpu().numpy()), y=self.all_training_labels.cpu().numpy()).astype(np.float32)

        self.classifier = SimpleClassifier(self.params.latent_dim, self.params.classifier_inner_dims, len(class_weights), class_weights, nn.Linear).to(self.params.device)

        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.params.lr)
        self.losses_aggregator = LossesAggregator()

        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.params.scheduler_factor,
                                                        patience=self.params.scheduler_patience,
                                                        cooldown=self.params.scheduler_cooldown, verbose=True)

        self.visualise_images(self.all_training_data[0:100].clamp(0, 1), 10, 10, 'originals')


    @torch.no_grad()
    def on_training_finished(self) -> TrainingSummary:
        z_t, _, _ = self.vae(self.all_training_data)
        z = z_t.cpu()
        self.drawer.plot_2d_scatter(
            z,
            self.all_training_labels.cpu().numpy(),
            title='latent space 2d',
            file_name='plot'
        )

        images = self.all_training_data[0:100]
        images_z_t, _, _ = self.vae(images)
        recon_images = self.vae.decode(images_z_t).cpu()
        self.visualise_images(recon_images.clamp(0, 1), 10, 10, 'recons')

        n = 15
        x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        sampled_images = []
        for i_x in range(n):
            for i_y in range(n):
                sampled_images.append([x[i_x][i_y], y[i_x][i_y]])
                # print(f'{x[i_x][i_y]}, {y[i_x][i_y]}')
        sampled_images = np.array(sampled_images, dtype=np.float32)
        recon_sampled_images = self.vae.decode(torch.tensor(sampled_images, device=self.params.device)).cpu()
        self.visualise_images(recon_sampled_images.clamp(0, 1), n, n, 'sampled_recons')

    def get_data_loader(self):
        return self.train_loader

    @torch.no_grad()
    def visualise_images(self, images_t: Tensor, rows: int, cols: int, title: str):
        images = torch.reshape(images_t.unsqueeze(1), (images_t.shape[0], 1, 28, 28)).cpu()
        images = self.to_img(images)
        # np_imagegrid = torchvision.utils.make_grid(images[1:50], rows, cols).numpy()
        # plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        self.show_image(torchvision.utils.make_grid(images, rows, cols))
        # plt.show()
        plt.savefig(Path(self.params.working_directory, title))
        plt.close()

    def show_image(self, img):
        img = self.to_img(img)
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    @staticmethod
    def to_img(x: Tensor):
        return x.clamp(0, 1)

    def perform_batch(self, i_epoch, i_batch, batch):
        self.vae.train()
        self.optimizer.zero_grad()

        batch_images = batch['samples']
        batch_labels = batch['labels']

        z_t, mu_t, log_var_t = self.vae(batch_images)
        classifier_predicted_labels = self.classifier(z_t)

        x_tag = self.vae.decode(z_t)

        kld_loss_t = self.vae.kld_loss(mu_t, log_var_t)
        reconstruction_loss_t = self.vae.reconstruction_loss(x_tag, batch_images)

        classifier_loss_t = self.classifier.loss_fn(classifier_predicted_labels, batch_labels)

        loss_name_to_value_t={
            'kld_loss': kld_loss_t,
            'reconstruction_loss': reconstruction_loss_t,
            'classifier_loss': classifier_loss_t
        }

        batch_losses = TrainingBatchLoss(loss_name_to_value_t, self.params.loss_coefs)

        batch_losses.total_loss_t.backward()
        self.optimizer.step()

        self.losses_aggregator.add_batch_loss(batch_losses)


if __name__ == '__main__':
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    params = MnistModelParameters(
        device=torch.device('cuda'),
        n_epochs=100,
        random_manager=RandomManager(True, 0, logger),
        working_directory='./output/mnist_test/exp_9',
        lr=0.001,
        batch_size=300,
        latent_dim=2,
        encode_dims=[512],
        decode_dims=[512],
        classifier_inner_dims=[],
        loss_coefs={
            'kld_loss': 10.0,
            'reconstruction_loss': 1.0,
            'classifier_loss': 150.0
        },
        scheduler_factor=0.3,
        scheduler_patience=150,
        scheduler_cooldown=100
    )

    # for i, classifier_loss in enumerate([1.0, 5.0, 10.0]):
    #     params.working_directory = f'./output/mnist_test/9_{i}_1000'
    #     params.loss_coefs['classifier_loss'] = classifier_loss
    trainer = MnistModelTrainer(logger, params)
    x = trainer.train_model()


    # trainer = MnistModelTrainer(logger, params)
    # x = trainer.train_model()
