from typing import Optional, Type

import torch
import torch.utils.data
from torch import nn, Tensor
from torch.nn import functional as F

# TODO support frozen epsilon, in order to get same encoding during post training evaluations
from src.module_init import xavier_uniform_init, ModuleGain


class SimpleVaeNet(nn.Module):
    def __init__(self, input_dim, encode_dims, latent_dim, decode_dims, decoder_skip_connection: bool, layer_type: Type[nn.Linear]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim  # output and input are of the same size
        self.encode_dims = encode_dims
        self.latent_dim = latent_dim
        self.decode_dims = decode_dims
        self.layer_type = layer_type
        self.decoder_skip_connection = decoder_skip_connection
        self.activation_function = nn.PReLU
        self.activation_initialization_gain = ModuleGain.PRELU

        self._0_tensor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.l1_regularization = nn.L1Loss(reduction='sum')

        "prepare encoder"
        if len(encode_dims) == 0 or encode_dims is None:
            self.encoder = None
            mu_log_var_input_dim = input_dim
        else:
            encoder = [layer_type(input_dim, encode_dims[0]), self.activation_function()]
            for i in range(1, len(encode_dims)):
                encoder.extend([layer_type(encode_dims[i - 1], encode_dims[i]), self.activation_function()])
            mu_log_var_input_dim = encode_dims[-1]
            self.encoder = nn.Sequential(*encoder)

        "get mu and sigma"
        mu_nn = [layer_type(mu_log_var_input_dim, latent_dim), self.activation_function()]
        self.mu_nn = nn.Sequential(*mu_nn)
        log_var_nn = [layer_type(mu_log_var_input_dim, latent_dim), self.activation_function()]
        self.log_var_nn = nn.Sequential(*log_var_nn)

        "prepare decoder"
        self.decoder_residual = None
        if len(decode_dims) == 0 or decode_dims is None:
            decoder = [self.activation_function(), layer_type(latent_dim, self.output_dim)]
            self.decoder_last_fc = nn.Identity()
        else:
            decoder = [layer_type(latent_dim, decode_dims[0]), self.activation_function()]
            for i in range(1, len(decode_dims)):
                decoder.extend([layer_type(decode_dims[i - 1], decode_dims[i]), self.activation_function()])
            # decoder.pop()
            self.decoder_last_fc = layer_type(decode_dims[-1], input_dim)
            if decoder_skip_connection:
                self.decoder_residual = layer_type(latent_dim, decode_dims[-1])
            self.decoder_last_non_linearity = self.activation_function()

        # self.decoder = nn.Sequential(*decoder)
        self.decoder = nn.ModuleList(decoder)
        # self.decoder = nn.Sequential(*decoder, self.activation_function())
        self.init_weights()

    def encode(self, x: Tensor):
        if self.encoder is None:
            encoded_x = x
        else:
            encoded_x = self.encoder(x)
        mu = self.mu_nn(encoded_x)
        log_var = self.log_var_nn(encoded_x)
        return mu, log_var

    def decode(self, z: Tensor):
        residual = self.decoder_residual(z) if self.decoder_residual else None
        x_tag = z
        for layer_idx, decoder_layer in enumerate(self.decoder):
            x_tag = decoder_layer(x_tag)
            if residual is not None and layer_idx > 0 and type(decoder_layer) is not self.activation_function and \
                    residual.shape[1] == x_tag.shape[1]:
                x_tag += residual
        return self.decoder_last_fc(x_tag)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # def sample(self, mu, log_var, labels, n_samples=10):
    #     """return either n_samples from each mu or simply returns the mu(for regular auto encoder).
    #         Also Duplicates the labels for classifiers,center search etc."""
    #
    #     batch_size = mu.size()[0]
    #     std = log_var.mul(0.5).exp_()
    #
    #     if n_samples:
    #         sample_labels = []
    #         q = torch.distributions.Normal(mu.tile((n_samples, 1)), std.tile((n_samples, 1)))
    #         sample_z = q.rsample()
    #         return sample_z, labels.tile(n_samples)
    #
    #     else:
    #         return mu, labels

    # def forward(self, x, labels, n_samples=10):
    #     mu, log_var = self.encode(x)
    #     z_sample, sampled_labels = self.sample(mu, log_var, labels, n_samples)
    #     return self.decode(z_sample), z_sample, mu, log_var, sampled_labels

    def forward(self, x_input_t: torch.Tensor):
        mu, log_var = self.encode(x_input_t)
        if not self.training:
            log_var.fill_(-torch.inf)
        z_sample = self.reparameterize(mu, log_var)
        return z_sample, mu, log_var

    def kld_loss(self, mu_t: torch.Tensor, log_var_t: torch.Tensor, class_weight_t:Optional[torch.Tensor] = None):
        kld = -0.5 * (1 + log_var_t - mu_t ** 2 - log_var_t.exp()).sum(dim=1)
        if class_weight_t is not None:
            kld *= class_weight_t
        return kld.mean()

    def l2_regularization_loss(self):
        raise RuntimeError()
        layers_to_regularize = []
        for layer in [*self.encoder, *self.decoder, self.decoder_residual, self.decoder_last_fc, self.mu_nn, self.log_var_nn]:
            if type(layer) is self.layer_type:
                layers_to_regularize.append(layer.weight)
        l2_square = [layer.square() for layer in layers_to_regularize]
        l2_sum = [squared_layer.sum() for squared_layer in l2_square]
        return torch.stack(l2_sum).sum()

    def l1_regularization_loss(self):
        layers_to_regularize = []
        inspected_layers = [*self.decoder, self.decoder_residual, self.decoder_last_fc]
        if type(self.encoder) is nn.Sequential:
            inspected_layers.extend([*self.encoder])
        for layer in inspected_layers:
            if type(layer) is self.layer_type:
                layers_to_regularize.append(layer)
        # l1_losses = [self.l1_regularization(layer.weight, self._0_tensor) for layer in layers_to_regularize]
        l1_losses = []
        for layer in layers_to_regularize:
            l1_losses.append(torch.abs(layer.weight).sum())
        return torch.stack(l1_losses).sum()

    def reconstruction_loss(self, x_recons_samples_t: torch.Tensor, x_t: torch.Tensor, class_weight_t: Optional[torch.Tensor] = None):
        # n_samples = x_recons_samples.size()[0] // x.size()[0]
        # x_tiled = x.tile((n_samples, 1))
        return F.mse_loss(x_t, x_recons_samples_t, reduction='none').sum(dim=1).mean()
        x_delta = x_t - x_recons_samples_t
        # "*" operator should be faster than using "pow(2)"
        mse_error = (x_delta ** 2).sum(dim=1).sqrt()
        if class_weight_t is not None:
            mse_error *= class_weight_t
        return mse_error.mean()

    def init_weights(self):
        if self.encoder:
            xavier_uniform_init(self.encoder, self.activation_initialization_gain)
        xavier_uniform_init(self.mu_nn)
        xavier_uniform_init(self.log_var_nn)
        xavier_uniform_init(self.decoder_last_fc)
        if self.decoder_residual:
            xavier_uniform_init(self.decoder_residual, self.activation_initialization_gain)
        xavier_uniform_init(self.decoder, self.activation_initialization_gain)

    # def __str__(self):
    #     return f'Simple VAE model, {super.__str__(self)}'
