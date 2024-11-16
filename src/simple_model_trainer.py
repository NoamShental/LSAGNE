from __future__ import annotations

import dataclasses
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from numpy.typing import NDArray
from prefect.engine import signals
from torch import optim, Tensor
from torch.optim import lr_scheduler

from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_data_augmentation_v1.augment_cmap import AugmentorChooser
from src.cmap_dataset_splitter import SplittedCmapDataset
from src.cmap_evaluation_data import CmapEvaluationData, SplittedCmapEvaluationData
from src.dataframe_utils import df_to_str
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.drawer import Drawer
from src.logger_annotations import MICHAEL_NOT_INTERESTED_ANNOTATION, MICHAEL_ONLY_ANNOTATION
from src.losses_aggregator import LossesAggregator
from src.model_trainer import ModelTrainer
from src.models.best_epoch_checkpoint import BestEpochCheckpoint
from src.models.dudi_basic.anchor_points import AnchorPoints
from src.models.dudi_basic.cloud_trimmer import trim_clouds
from src.models.dudi_basic.data_loaders.all_clouds_multi_device_fast_data_loader import \
    AllCloudsMultiDeviceFastDataLoader
from src.models.dudi_basic.data_loaders.full_multi_device_fast_data_loader import FullMultiDeviceFastDataLoader
from src.models.dudi_basic.data_loaders.triangle_multi_device_fast_data_loader import TriangleMultiDeviceFastDataLoader
from src.models.dudi_basic.dynamic_reference_points_selector import DynamicAnchorPointsSelector
from src.models.dudi_basic.embedded_anchors_and_vectors import EmbeddedAnchorsAndVectors
from src.models.dudi_basic.in_training_evaluators.cloud_diameter_evaluator_ import CloudsDistancesEvaluator
from src.models.dudi_basic.in_training_evaluators.cloud_radius_evaluator_ import evaluate_cloud_radii
from src.models.dudi_basic.lsagne_model import LsagneModel
from src.models.dudi_basic.model_learning_parameters import ModelLearningParameters
from src.models.dudi_basic.multi_device_data import MultiDeviceData
from src.models.dudi_basic.static_reference_points_selector import StaticAnchorPointsSelector
from src.models.mtadam import MTAdam
from src.models.svm_utils import perform_svm_accuracy_evaluation
from src.random_manager import RandomManager
from src.training_batch_loss import TrainingBatchLoss
from src.training_summary import TrainingSummary
from src.write_to_file_utils import write_str_to_file


class SimpleModelTrainer(ModelTrainer[ModelLearningParameters]):

    def __init__(self, logger, params: ModelLearningParameters):
        super().__init__(logger, params)
        self.current_reference_points = None

    def set_reference_points(self, reference_points: AnchorPoints[NDArray[float]]):
        self.logger.info("Setting new reference points.")
        self.last_reference_points = self.current_reference_points
        self.current_reference_points = reference_points
        self.multi_device_data = MultiDeviceData.create(
            cmap=self.datasets.training_only,
            anchor_points=reference_points,
            device=self.params.device
        )

        if self.params.data_loader_type == FullMultiDeviceFastDataLoader:
            raise NotImplementedError("Please check code!")
            self.train_samples_loader, self.multi_device_data = self.params.data_loader_type.from_multi_device_data(
                multi_device_data,
                self.params.batch_size,
                self.params.device)
        elif self.params.data_loader_type == TriangleMultiDeviceFastDataLoader:
            raise NotImplementedError("Please check code!")
            self.train_samples_loader, self.multi_device_data = self.params.data_loader_type.from_multi_device_data(
                on_cpu_multi_device_data=multi_device_data,
                device=self.params.device,
                # number_of_batches=int(len(self.train_cmap_dataset) / 330),
                number_of_batches=30,
                number_of_tissues_per_batch=3,
                number_of_samples_per_cloud=30,
                desired_number_of_triangles_per_batch=3,
                cmap=self.datasets.training_only)
        elif self.params.data_loader_type == AllCloudsMultiDeviceFastDataLoader:
            self.train_samples_loader = AllCloudsMultiDeviceFastDataLoader(
                multi_device_data=self.multi_device_data,
                number_of_batches=self.params.number_of_batches,
                number_of_samples_per_cloud=self.params.number_of_samples_per_cloud,
                cmap=self.datasets.training_only
            )
        else:
            raise AssertionError(f'No support for {self.params.data_loader_type.__name__}.')

    def _update_training_dataset(self, new_training_cmap_dataset: RawCmapDataset):
        self.logger.info("Updating the CMAP")
        self.logger.info(f"Old training CMAP len was {len(self.datasets.training_only):,} ; new len is {len(new_training_cmap_dataset):,}")
        self.datasets = dataclasses.replace(self.datasets, training_only=new_training_cmap_dataset)
        z_t = self.model.get_embedding(torch.tensor(new_training_cmap_dataset.data, device=self.params.device)).z_t
        self.set_reference_points(DynamicAnchorPointsSelector(self.datasets.training_only, self.params.enable_triangle, self.logger)
                                  .select_points(z_t))
        self.model.update_training_cmap(new_training_cmap_dataset)

    def _perform_clouds_trimming(
            self,
            training_only_cloud_ref_to_distances_from_radius_t: Dict[CmapCloudRef, Tensor],
            training_concealed_cloud_ref_to_distances_from_radius_t: Dict[CmapCloudRef, Tensor],
    ):
        self.logger.info('Performing trimming of CMAP...')
        request_trimming_cloud_ref_to_keep_ratio: Dict[CmapCloudRef, float] = {}
        for cloud_ref in self.datasets.training_only.unique_cloud_refs:
            if cloud_ref.perturbation.is_dmso_6h or cloud_ref.perturbation.is_dmso_24h:
                ratio_to_keep = self.params.trim_untreated_clouds_and_time_24h_ratio_to_keep
            else:
                ratio_to_keep = self.params.trim_treated_clouds_ratio_to_keep
            request_trimming_cloud_ref_to_keep_ratio[cloud_ref] = ratio_to_keep
        trimmed_training_only_dataset = trim_clouds(
            self.datasets.training_only,
            training_only_cloud_ref_to_distances_from_radius_t,
            request_trimming_cloud_ref_to_keep_ratio
        )
        self._update_training_dataset(trimmed_training_only_dataset)
        trimmed_training_concealed = trim_clouds(
            self.datasets.training_concealed,
            training_concealed_cloud_ref_to_distances_from_radius_t,
            request_trimming_cloud_ref_to_keep_ratio
        )
        self.datasets = dataclasses.replace(self.datasets, training_concealed=trimmed_training_concealed)

    def _print_and_save_clouds_radius_df(
            self,
            clouds_radius_df: pd.DataFrame,
            i_epoch: int,
            file_name_note: str = '',
            print_all_columns: bool = False,
    ):
        columns_to_print = None if print_all_columns else ['original_label', 'display_name', 'cloud_size', 'std', 'min',
                                                           'q_5', 'q_25', 'mean', 'median', 'q_75', 'q_95', 'max']
        df_str = df_to_str(clouds_radius_df, columns_to_print)
        folder_path = os.path.join(self.params.working_directory, 'cloud_radius')
        file_name = f"{i_epoch:05d}_{file_name_note}.txt"
        df_str = f"""
            \nCloud radii evaluation ({file_name_note}):
            \n{"=" * 100}
            \n{df_str}
        """
        self.logger.info(df_str, extra=MICHAEL_NOT_INTERESTED_ANNOTATION)
        write_str_to_file(folder_path, file_name, df_str)

    def _calculate_clouds_radii(
            self,
            cmap_evaluation_data: CmapEvaluationData,
            # i_epoch: int,
    ) -> Tuple[pd.DataFrame, Dict[CmapCloudRef, Tensor]]:
        cloud_ref_to_cloud_center_latent_t = self.multi_device_data.on_device_original_space_anchor_points.\
            cloud_ref_to_cloud_center.transform_tensor_with(lambda x: self.model.get_embedding(x).mu_t)

        clouds_radius_df, cloud_ref_to_distances_from_radius_t = evaluate_cloud_radii(
                cmap=cmap_evaluation_data.cmap,
                z_t=cmap_evaluation_data.z_t,
                cloud_ref_to_cloud_center_t=cloud_ref_to_cloud_center_latent_t,
        )
        return clouds_radius_df, cloud_ref_to_distances_from_radius_t

    def augment_batch(self, batch: MultiDeviceData) -> MultiDeviceData:
        cmap_augmentor, augmentation_rate = self.cmap_augmentor_chooser.choose_augmentor()
        if cmap_augmentor is None:
            return batch
        # mask only clouds which are needed to be augmented
        clouds_to_augment_mask = np.isin(batch.on_cpu.cloud_refs, cmap_augmentor.supported_cloud_refs)
        augmentation_mask = np.random.binomial(n=1, p=augmentation_rate, size=clouds_to_augment_mask.sum()).astype(bool)
        mask = np.full(len(batch), False)
        mask[clouds_to_augment_mask] = augmentation_mask
        raw_samples = np.copy(batch.on_cpu.raw_samples)
        raw_samples_t = torch.clone(batch.on_device.raw_samples)
        # start = time.time()
        cmap_augmentor.augment_samples_inplace(raw_samples, batch.on_cpu.cloud_refs, mask)
        raw_samples_t[mask] = torch.tensor(raw_samples[mask], device=batch.device)
        # raw_samples_t[current_cloud_augmented_samples_mask] = torch.tensor(augmented_samples, device=batch.device)
        # end = time.time()
        # print(f'! batch augmentation took {end - start} secs.')
        return batch.augment(raw_samples, raw_samples_t)

    def on_training_started(self):
        self.random_manager = RandomManager(self.params.use_seed, self.params.random_seed)
        self.drawer = Drawer(self.logger, self.params.working_directory)
        self.datasets: SplittedCmapDataset = SplittedCmapDataset.split_cmap_dataset(
            logger=self.logger,
            original_cmap_dataset=self.params.raw_cmap_dataset,
            left_out_cloud_ref=self.params.left_out_cloud,
            cross_validation_cloud_refs=self.params.cross_validation_clouds,
            cloud_ref_to_partial_training_size=self.params.partial_cloud_training
        )
        self.cmap_augmentor_chooser = AugmentorChooser(
            logger=self.logger,
            augmentation_params=self.params.augmentation_params,
            random_seed=self.random_manager.random_seed
        )

        self.set_reference_points(StaticAnchorPointsSelector(self.datasets.training_only, self.params.enable_triangle, self.logger).select_points())

        # Set up the network and training parameters
        self.model = LsagneModel(
            train_cmap_dataset=self.datasets.training_only,
            input_dim=self.params.input_dim,
            latent_dim=self.params.embedding_dim,
            encoder_layers=self.params.vae_encode_inner_dims,
            decoder_layers=self.params.vae_decode_inner_dims,
            clouds_classifier_inner_layers=self.params.clouds_classifier_inner_layers,
            tissues_classifier_inner_layers=self.params.tissues_classifier_inner_layers,
            logger=self.logger,
            device=self.params.device,
            contrastive_margin=self.params.contrastive_margin,
            enable_vae_skip_connection=self.params.enable_vae_skip_connection,
            enable_triangle=self.params.enable_triangle,
            default_linear_layer=self.params.default_linear_layer,
            add_class_weights_to_classifiers=self.params.add_class_weights_to_classifiers,
            treatment_and_drug_vectors_distance_loss_cdist_usage=self.params.treatment_and_drug_vectors_distance_loss_cdist_usage,
            different_directions_loss_power_factor=self.params.different_directions_loss_power_factor,
            use_untrained_clouds_predictions_in_training=self.params.use_untrained_clouds_predictions_in_training,
            untrained_cloud_refs=[self.params.left_out_cloud, *self.params.cross_validation_clouds],
            perturbations_equivalence_sets=self.params.perturbations_equivalence_sets
        )
        self.desired_max_cloud_radius = self.params.max_radius

        if self.params.use_mtadam:
            raise NotImplementedError()
            self.logger.info(f'mtadam vae_mse, vae_kld, classifier_loss, collinearity, distance_from_predicted, '
                             f'distance_from_cloud_center ==> {self.params.mtadam_loss_weights}')
            self.optimizer = MTAdam(model.parameters(), lr=self.params.lr, amsgrad=self.params.use_amsgrad)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.params.lr, amsgrad=self.params.use_amsgrad)
            # self.optimizer = optim.NAdam(self.model.parameters(), lr=self.params.lr)

        if self.params.use_scheduler:
            self.logger.info(
                f'Using ReduceLROnPlateau with factor={self.params.scheduler_factor}, patience={self.params.scheduler_patience}, '
                f'cooldown={self.params.scheduler_cooldown}')
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.params.scheduler_factor,
                                                            patience=self.params.scheduler_patience,
                                                            cooldown=self.params.scheduler_cooldown, verbose=True)
        else:
            self.scheduler = None

        self.losses_aggregator = LossesAggregator()

        self.lr_history = []

        self.best_model_checkpoint = BestEpochCheckpoint()

    def get_data_loader(self):
        return self.train_samples_loader

    def on_epoch_started(self, i_epoch):
        self.model.train()
        if i_epoch <= self.params.warmup_reference_points_duration:
            self.current_epoch_loss_coef = self.params.warmup_reference_points_loss_coef
        else:
            self.current_epoch_loss_coef = self.params.loss_coef

    def perform_batch(self, i_epoch, i_batch, batch: MultiDeviceData):
        self.model.train()
        self.optimizer.zero_grad()

        if self.params.augment_during_warmup or self.params.warmup_reference_points_duration < i_epoch:
            batch = self.augment_batch(batch)

        samples_embedding = self.model.get_embedding(batch.on_device.raw_samples)

        loss_name_to_value_t = \
            self.model.loss_fn(
                i_epoch=i_epoch,
                i_batch=i_batch,
                samples_embedding=samples_embedding,
                batch_multi_device=batch,
                loss_coef=self.current_epoch_loss_coef,
                desired_max_cloud_radius=self.desired_max_cloud_radius,
                is_warmup_batch=i_epoch <= self.params.warmup_reference_points_duration,
                treatment_vectors_magnitude_regulator_relu_coef=self.params.treatment_vectors_magnitude_regulator_relu_coef,
                drug_vectors_magnitude_regulator_relu_coef=self.params.drug_vectors_magnitude_regulator_relu_coef,
                perturbations_equivalence_sets_loss_coef=self.params.perturbations_equivalence_losses_coefs,
            )

        batch_losses = TrainingBatchLoss(loss_name_to_value_t, self.current_epoch_loss_coef)

        total_loss_t = batch_losses.total_loss_t

        if total_loss_t.item() == np.nan:
            self.logger.fatal('!' * 100)
            raise signals.FAIL()

        if self.params.use_mtadam:
            raise NotImplementedError()
            # optimizer.step([vae_mse_loss, vae_kld_loss, classifier_loss, collinearity_loss,
            #                 distance_from_predicted_loss, distance_from_cloud_center_loss],
            #                mtadam_loss_weights, None)
        else:
            total_loss_t.backward()
            self.optimizer.step()

        self.losses_aggregator.add_batch_loss(batch_losses)

    def _calculate_epoch_evaluation_data(self) -> SplittedCmapEvaluationData:
        return SplittedCmapEvaluationData.create_instance(
            training_only_cmap=self.datasets.training_only,
            training_concealed_cmap=self.datasets.training_concealed,
            model=self.model
        )

    def _perform_radii_evaluation_after_trimming(
            self,
            epoch_evaluation_data: SplittedCmapEvaluationData,
            pre_trimming_training_only_clouds_radius_df: pd.DataFrame,
            pre_trimming_training_concealed_clouds_radius_df: pd.DataFrame,
            i_epoch: int
    ):
        training_only_clouds_radius_df, _ = self._calculate_clouds_radii(epoch_evaluation_data.training_only)
        training_concealed_clouds_radius_df, _ = self._calculate_clouds_radii(epoch_evaluation_data.training_concealed)
        original_label_column_name = 'original_label'
        pre_trimming_training_only_clouds_radius_df.sort_values(by=original_label_column_name, inplace=True)
        pre_trimming_training_concealed_clouds_radius_df.sort_values(by=original_label_column_name, inplace=True)
        training_only_clouds_radius_df.sort_values(by=original_label_column_name, inplace=True)
        training_concealed_clouds_radius_df.sort_values(by=original_label_column_name, inplace=True)
        for before_df, after_df in [
            (pre_trimming_training_only_clouds_radius_df, training_only_clouds_radius_df),
            (pre_trimming_training_concealed_clouds_radius_df, training_concealed_clouds_radius_df),
        ]:
            for column_name in ['cloud_size', 'std', 'min', 'q_5', 'q_25', 'mean', 'median', 'q_75', 'q_95', 'max']:
                after_df[f'{column_name}_diff'] = after_df[column_name] - before_df[column_name]
        self._print_and_save_clouds_radius_df(training_only_clouds_radius_df, i_epoch, "after_trimming_training_only",
                                              True)
        self._print_and_save_clouds_radius_df(training_concealed_clouds_radius_df, i_epoch,
                                              "after_trimming_training_concealed", True)

    def _perform_classifier_accuracy_evaluation(
            self,
            epoch_cmap_evaluation_data: CmapEvaluationData,
            cmap: RawCmapDataset,
            tag: str
    ) -> str:
        # We convert all to labels in order to use the quick Numpy.unique method
        y_encoded_labels = self.model.clouds_classifier(epoch_cmap_evaluation_data.z_t).argmax(-1).cpu().numpy()
        cloud_refs_encoded_labels = self.model.clouds_classifier.convert_class_to_encoded_label(cmap.cloud_refs)
        data_size = len(epoch_cmap_evaluation_data)
        correctly_predicted = np.sum(cloud_refs_encoded_labels == y_encoded_labels) * 100.0 / data_size
        # this is a very fast way to transform every value of an array using a dictionary.
        u, inv = np.unique(y_encoded_labels, return_inverse=True)
        y_tissues = np.array([
            self.model.clouds_classifier._class_encoder.encoded_label_to_class[encoded_label].tissue
            for encoded_label in u]
        )[inv].reshape(y_encoded_labels.shape)
        correctly_predicted_tissues = np.sum(y_tissues == cmap.tissues) * 100.0 / data_size
        return f' ; {tag} ACC - CLS: {round(correctly_predicted, 2)}; {tag} CLS-T: {round(correctly_predicted_tissues, 2)}'

    @torch.no_grad()
    def on_epoch_finished(self, i_epoch):
        self.model.eval()

        epoch_evaluation_data = self._calculate_epoch_evaluation_data()

        if i_epoch >= self.params.warmup_reference_points_duration:
            is_cloud_radii_evaluation_epoch = i_epoch % 10 == 0
            is_trimming_epoch = i_epoch in self.params.clouds_trimming_epochs
            if is_cloud_radii_evaluation_epoch or is_trimming_epoch:
                training_only_clouds_radius_df, training_only_cloud_ref_to_distances_from_radius_t = \
                    self._calculate_clouds_radii(epoch_evaluation_data.training_only)
                training_concealed_clouds_radius_df, training_concealed_cloud_ref_to_distances_from_radius_t = \
                    self._calculate_clouds_radii(epoch_evaluation_data.training_concealed)
                if is_cloud_radii_evaluation_epoch:
                    self._print_and_save_clouds_radius_df(training_only_clouds_radius_df, i_epoch, 'training_only')
                    self._print_and_save_clouds_radius_df(training_concealed_clouds_radius_df, i_epoch, 'training_concealed')
                if is_trimming_epoch:
                    # TODO add L2 distance between cloud centers
                    self._perform_clouds_trimming(
                        training_only_cloud_ref_to_distances_from_radius_t=training_only_cloud_ref_to_distances_from_radius_t,
                        training_concealed_cloud_ref_to_distances_from_radius_t=training_concealed_cloud_ref_to_distances_from_radius_t
                    )
                    epoch_evaluation_data = self._calculate_epoch_evaluation_data()
                    self._perform_radii_evaluation_after_trimming(
                        epoch_evaluation_data=epoch_evaluation_data,
                        pre_trimming_training_only_clouds_radius_df=training_only_clouds_radius_df,
                        pre_trimming_training_concealed_clouds_radius_df=training_concealed_clouds_radius_df,
                        i_epoch=i_epoch
                    )

        self.losses_aggregator.end_epoch()

        epoch_loss = self.losses_aggregator.last_epoch_loss

        # if i_epoch % 10 == 0:
        #     y = self.model.clouds_classifier(z_t).argmax(-1)
        #     correctly_predicted = int((encoded_numeric_labels_t == y).sum().cpu())
        #     accc = correctly_predicted / len(z_t)
        #
        #     self.logger.info(f"Epoch {i_epoch} - MSE: {round(epoch_loss.loss_name_to_value['vae_mse'],2)}," +
        #                      f" KLD {round(epoch_loss.loss_name_to_value['vae_kld'],2)}, " +
        #                      f" DISTCNTR: {round(epoch_loss.loss_name_to_value['distance_from_cloud_center'],2)}" +
        #                      f" CLS: {round(epoch_loss.loss_name_to_value['clouds_classifier'],2)}" +
        #                      f" TCLS: {round(epoch_loss.loss_name_to_value['tissues_classifier'],2)}" +
        #                      f" COLIN: {round(epoch_loss.loss_name_to_value['treatment_vectors_collinearity'],2)}" +
        #                      f" ACC_C: {round(accc,2)}")
        # self.logger.info(f"Epoch metrics: VAE loss -> {epoch_loss.loss_name_to_value['vae_mse'] + epoch_loss.loss_name_to_value['vae_kld']}")

        self.lr_history.append(self.optimizer.param_groups[0]['lr'])

        found_better_model, old_total_loss = self.best_model_checkpoint.add_checkpoint(epoch_loss.total_loss,
                                                                                       i_epoch,
                                                                                       self.model,
                                                                                       self.optimizer)
        if found_better_model:
           self.logger.info(f'Found better model: total loss {round(old_total_loss,2):,} -> {round(epoch_loss.total_loss,2):,}', extra=MICHAEL_NOT_INTERESTED_ANNOTATION)

        if self.scheduler and i_epoch > self.params.warmup_reference_points_duration:
            self.scheduler.step(epoch_loss.total_loss)

        if i_epoch == self.params.warmup_reference_points_duration:
            self.best_model_checkpoint.reset_total_loss()

        perform_distance_evaluation = i_epoch % self.params.distances_evaluation_interval == 0
        perform_reference_points_reselection = i_epoch in self.params.reselect_reference_points_epochs or \
                                               i_epoch == self.params.warmup_reference_points_duration or \
                                               i_epoch in self.params.clouds_trimming_epochs
        perform_classifier_accuracy_evaluation = i_epoch % self.params.evaluate_classifier_accuracy_interval == 0 

        tstr = ""
        if perform_distance_evaluation or perform_reference_points_reselection or perform_classifier_accuracy_evaluation:
            if perform_distance_evaluation:
                CloudsDistancesEvaluator(self.logger, self.params.working_directory, 'training_only').evaluate(
                    self.datasets.training_only,
                    epoch_evaluation_data.training_only.z_t,
                    i_epoch
                )
                if len(self.datasets.training_concealed) > 0:
                    CloudsDistancesEvaluator(self.logger, self.params.working_directory, 'training_concealed').evaluate(
                        self.datasets.training_concealed,
                        epoch_evaluation_data.training_concealed.z_t,
                        i_epoch
                    )

            if perform_classifier_accuracy_evaluation:
                self.logger.info(f'Epoch {i_epoch}', extra=MICHAEL_ONLY_ANNOTATION)
                tstr += f'Losses - KLD: {round(epoch_loss.loss_name_to_value["vae_kld"],2):,}; MSE: {round(epoch_loss.loss_name_to_value["vae_mse"],2):,}; CLS: {round(epoch_loss.loss_name_to_value["clouds_classifier"],2):,}; CLS-T: {round(epoch_loss.loss_name_to_value["tissues_classifier"],2):,};'
                tstr += f' COLIN-T-C: {round(epoch_loss.loss_name_to_value["treatment_vectors_collinearity_using_batch_control"],2):,}; COLIN-T-T: {round(epoch_loss.loss_name_to_value["treatment_vectors_collinearity_using_batch_treated"],2):,}'
                tstr += f' ; COLIN-D-T: {round(epoch_loss.loss_name_to_value["drug_vectors_collinearity_using_batch_treated"],2):,}; COLIN-D-C: {round(epoch_loss.loss_name_to_value["drug_vectors_collinearity_using_batch_control"],2):,}; DISTCC_6: {round(epoch_loss.loss_name_to_value["distance_from_cloud_center_6h"],2):,}; DISTCC_24: {round(epoch_loss.loss_name_to_value["distance_from_cloud_center_dmso_24h"],2):,}'
                tstr += f' ; MAX-R: {round(epoch_loss.loss_name_to_value["max_radius_limiter"], 2):,};'
                tstr += f' ; T - ref mag: {round(epoch_loss.loss_name_to_value["treatment_vectors_magnitude_regulator"], 2):,}; D - ref mag: {round(epoch_loss.loss_name_to_value["drug_vectors_magnitude_regulator"], 2):,};'
                tstr += self._perform_classifier_accuracy_evaluation(
                    epoch_cmap_evaluation_data=epoch_evaluation_data.training_only,
                    cmap=self.datasets.training_only,
                    tag='training_only'
                )
                if len(epoch_evaluation_data.training_concealed) > 0:
                    tstr += self._perform_classifier_accuracy_evaluation(
                        epoch_cmap_evaluation_data=epoch_evaluation_data.training_concealed,
                        cmap=self.datasets.training_concealed,
                        tag='training_concealed'
                    )

            if perform_reference_points_reselection:
                self.logger.info(f'Making reselection of reference points...')
                self.set_reference_points(DynamicAnchorPointsSelector(
                    self.datasets.training_only, self.params.enable_triangle, self.logger)
                                          .select_points(epoch_evaluation_data.training_only.z_t))
                # self.best_model_checkpoint.reset_total_loss()
                # self.scheduler._reset()

        if i_epoch % self.params.evaluate_svm_accuracy_interval == 0:
            epoch_evaluation_data = epoch_evaluation_data.add_cv_and_left_out(self.datasets.cross_validation, self.datasets.left_out, self.model)
            embedded_anchors_and_vectors = EmbeddedAnchorsAndVectors.create(
                original_space_anchor_points_lookup=self.multi_device_data.on_device_original_space_anchor_points,
                embedder=self.model
            )
            all_svm_results = perform_svm_accuracy_evaluation(
                splitted_evaluation_data=epoch_evaluation_data,
                embedded_anchors_and_vectors=embedded_anchors_and_vectors,
                predicted_cloud_max_size=self.params.predicted_cloud_max_size,
                random_seed=self.params.random_manager.random_seed,
                perturbations_equivalence_sets=self.params.perturbations_equivalence_sets
            )
            tstr += f'; CLS-DIR (true): {round(all_svm_results.svm_1.summary["equivalence_sets_svm_acc"].mean() * 100, 2)}'
            tstr += f'; CLS-DIR (pred): {round(all_svm_results.svm_2.summary["equivalence_sets_svm_acc"].mean() * 100, 2)}'
            all_svm_results.log_and_save_to_dir(
                logger=self.logger,
                directory_path=os.path.join(self.params.working_directory, "svm_results"),
                i_epoch=i_epoch
            )
            if len(self.datasets.cross_validation) > 0:
                cv_acc = all_svm_results.svm_1.summary[all_svm_results.svm_1.summary['tag'].str.contains('CV')]['absolute_svm_acc'].to_list()[0]
                found_better_model, old_cv_acc = self.best_model_checkpoint.add_cv_checkpoint(
                    cv_acc,
                    i_epoch,
                    self.model,
                    self.optimizer)
                if found_better_model:
                    self.logger.info(
                        f'Found better model (CV-wise): CV_accuracy {round(old_cv_acc, 2):,} -> {round(cv_acc, 2):,}')

        if tstr:
            self.logger.info(tstr, extra=MICHAEL_ONLY_ANNOTATION)

    def on_training_finished(self) -> TrainingSummary:
        self.logger.info(f'Training has finished! Best epoch {self.best_model_checkpoint.epoch} with loss={self.best_model_checkpoint.total_loss:,} .')
        self.model.load_state_dict(self.best_model_checkpoint.model_state_dict)
        # self.model.load_state_dict(self.best_model_checkpoint.cv_model_state_dict) - probably should use this...

        training_summary = TrainingSummary(
            final_total_loss=self.best_model_checkpoint.total_loss,
            params=self.params,
            best_model_epoch=self.best_model_checkpoint.epoch,
            model_state_dict=self.best_model_checkpoint.model_state_dict,
            model=self.model,
            optimizer_state_dict=self.best_model_checkpoint.optimizer_state_dict,
            lr_history=self.lr_history,
            epochs_losses_history=self.losses_aggregator.all_epochs_history,
            left_out_cloud=self.params.left_out_cloud,
            anchor_points=self.current_reference_points,
            cmap_datasets=self.datasets,
            n_epochs=self.params.n_epochs
        )

        losses_df = pd.DataFrame(self.losses_aggregator.all_epochs_history)
        losses_df['vae_loss'] = losses_df['vae_mse'] + losses_df['vae_kld']
        losses_df.to_csv(os.path.join(self.params.working_directory, 'losses_df.csv'))

        batches_df = pd.DataFrame(self.losses_aggregator.all_batches_history)
        batches_df.to_csv(os.path.join(self.params.working_directory, 'batches_df.csv'))

        lr_df = pd.DataFrame({'lr': self.lr_history})
        lr_df.to_csv(os.path.join(self.params.working_directory, 'lr_df.csv'))

        self.drawer.plot_curves(self.losses_aggregator.all_epochs_history, self.params.n_epochs, 'losses.png')

        return training_summary
