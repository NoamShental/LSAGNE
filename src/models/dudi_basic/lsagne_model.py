from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import List, Dict, Tuple, Type, Optional, Collection, Set

import numpy as np
import torch
import torch.utils.data
from numpy.typing import NDArray
from torch import nn, Tensor
from torch.nn import functional as F
from torch.types import Device

from src.assertion_utils import assert_normalized, assert_promise_true, assert_same_len
from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_clouds_utils import find_center_of_cloud_t
from src.configuration import config
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.fast_tensor_builder import FastTensorBuilder
from src.models.dudi_basic.embedded_anchors_and_vectors import EmbeddedAnchorsAndVectors
from src.models.dudi_basic.lookup_tensor import LookupTensor
from src.models.dudi_basic.multi_device_data import MultiDeviceData, OnDeviceData, OnCpuData
from src.models.dudi_basic.simple_classifier import SimpleClassifier
from src.models.dudi_basic.simple_vae_net import SimpleVaeNet
from src.models.pair_triplet.pair.online_contrastive_loss import OnlineContrastiveLoss
from src.models.pair_triplet.pair_triplet_online_utils import HardNegativePairSelector
from src.models.predicted_clouds_calculator import predict_triangle_cloud
from src.multi_dim_arithmetics import calculate_nearest_points_on_2_lines_t
from src.perturbation import Perturbation
from src.samples_embedder import SamplesEmbedder
from src.samples_embedding import SamplesEmbedding
from src.tissue import Tissue


@dataclass(frozen=True)
class _BatchMasks:
    cpu: _CpuMasks
    device: _DeviceMasks
    batch_size: int

    def __post_init__(self):
        assert_promise_true(lambda: len(self.cpu.is_dmso_6h_idx) == len(self.device.is_dmso_6h_idx_t))
        assert_promise_true(lambda: len(self.cpu.not_dmso_6h_idx) == len(self.device.not_dmso_6h_idx_t))
        assert_promise_true(lambda: len(self.cpu.is_dmso_24h_idx) == len(self.device.is_dmso_24h_idx_t))
        assert_promise_true(lambda: len(self.cpu.not_dmso_6h_or_24h_idx) == len(self.device.not_dmso_6h_or_24h_idx_t))
        assert_promise_true(
            lambda: self.batch_size == len(self.cpu.is_dmso_6h_idx) + len(self.cpu.is_dmso_24h_idx) +
                    len(self.cpu.not_dmso_6h_or_24h_idx)
        )
        assert_promise_true(
            lambda: self.batch_size == len(self.device.is_dmso_6h_idx_t) + len(self.device.is_dmso_24h_idx_t) +
                    len(self.device.not_dmso_6h_or_24h_idx_t)
        )

    @classmethod
    def create(cls, batch_multi_device_data: MultiDeviceData) -> _BatchMasks:
        return cls(
            cpu=_CpuMasks.create(batch_multi_device_data.on_cpu),
            device=_DeviceMasks.create(batch_multi_device_data.on_device),
            batch_size=len(batch_multi_device_data)
        )


@dataclass(frozen=True)
class _DeviceMasks:
    is_dmso_6h_mask_t: Tensor
    is_dmso_6h_idx_t: Tensor
    not_dmso_6h_mask_t: Tensor
    not_dmso_6h_idx_t: Tensor
    is_dmso_24h_mask_t: Tensor
    is_dmso_24h_idx_t: Tensor
    not_dmso_6h_or_24h_mask_t: Tensor
    not_dmso_6h_or_24h_idx_t: Tensor

    def __post_init__(self):
        assert_promise_true(lambda: len(self.is_dmso_6h_idx_t) == int((sum(self.is_dmso_6h_mask_t)).cpu()))
        assert_promise_true(lambda: len(self.not_dmso_6h_idx_t) == int((sum(self.not_dmso_6h_mask_t)).cpu()))
        assert_promise_true(lambda: len(self.is_dmso_24h_idx_t) == int((sum(self.is_dmso_24h_mask_t)).cpu()))
        assert_promise_true(lambda: len(self.not_dmso_6h_or_24h_idx_t) == int((sum(self.not_dmso_6h_or_24h_mask_t)).cpu()))
        assert_promise_true(lambda: len(self.not_dmso_6h_or_24h_idx_t) == len(self.not_dmso_6h_idx_t) - len(self.is_dmso_24h_idx_t))

    @classmethod
    def create(cls, batch_on_device: OnDeviceData) -> _DeviceMasks:
        is_dmso_6h_mask_t: Tensor = batch_on_device.is_dmso_6h_mask
        is_dmso_6h_idx_t: Tensor = is_dmso_6h_mask_t.nonzero().squeeze(dim=1)
        not_dmso_6h_mask_t: Tensor = batch_on_device.not_dmso_6h_mask
        not_dmso_6h_idx_t: Tensor = not_dmso_6h_mask_t.nonzero().squeeze(dim=1)
        is_dmso_24h_mask_t: Tensor = batch_on_device.is_dmso_24h_mask
        is_dmso_24h_idx_t: Tensor = is_dmso_24h_mask_t.nonzero().squeeze(dim=1)
        not_dmso_6h_or_24h_mask_t: Tensor = batch_on_device.not_dmso_6h_or_24h_mask
        not_dmso_6h_or_24h_idx_t: Tensor = not_dmso_6h_or_24h_mask_t.nonzero().squeeze(dim=1)
        return cls(
            is_dmso_6h_mask_t=is_dmso_6h_mask_t,
            is_dmso_6h_idx_t=is_dmso_6h_idx_t,
            not_dmso_6h_mask_t=not_dmso_6h_mask_t,
            not_dmso_6h_idx_t=not_dmso_6h_idx_t,
            is_dmso_24h_mask_t=is_dmso_24h_mask_t,
            is_dmso_24h_idx_t=is_dmso_24h_idx_t,
            not_dmso_6h_or_24h_mask_t=not_dmso_6h_or_24h_mask_t,
            not_dmso_6h_or_24h_idx_t=not_dmso_6h_or_24h_idx_t
        )


@dataclass(frozen=True)
class _CpuMasks:
    is_dmso_6h_mask: NDArray[bool]
    is_dmso_6h_idx: NDArray[int]
    not_dmso_6h_mask: NDArray[bool]
    not_dmso_6h_idx: NDArray[int]
    is_dmso_24h_mask: NDArray[bool]
    is_dmso_24h_idx: NDArray[int]
    not_dmso_6h_or_24h_mask: NDArray[bool]
    not_dmso_6h_or_24h_idx: NDArray[int]

    def __post_init__(self):
        assert_promise_true(lambda: len(self.is_dmso_6h_idx) == self.is_dmso_6h_mask.sum())
        assert_promise_true(lambda: len(self.not_dmso_6h_idx) == self.not_dmso_6h_mask.sum())
        assert_promise_true(lambda: len(self.is_dmso_24h_idx) == self.is_dmso_24h_mask.sum())
        assert_promise_true(lambda: len(self.not_dmso_6h_or_24h_idx) == self.not_dmso_6h_or_24h_mask.sum())
        assert_promise_true(lambda: len(self.not_dmso_6h_or_24h_idx) == len(self.not_dmso_6h_idx) - len(self.is_dmso_24h_idx))
    @classmethod
    def create(cls, batch_on_cpu: OnCpuData) -> _CpuMasks:
        is_dmso_6h_mask: NDArray[bool] = batch_on_cpu.is_dmso_6h_mask
        is_dmso_6h_idx: NDArray[int] = is_dmso_6h_mask.nonzero()[0]
        not_dmso_6h_mask: NDArray[bool] = batch_on_cpu.not_dmso_6h_mask
        not_dmso_6h_idx: NDArray[int] = not_dmso_6h_mask.nonzero()[0]
        is_dmso_24h_mask: NDArray[bool] = batch_on_cpu.is_dmso_24h_mask
        is_dmso_24h_idx: NDArray[int] = is_dmso_24h_mask.nonzero()[0]
        not_dmso_6h_or_24h_mask: NDArray[bool] = batch_on_cpu.not_dmso_6h_or_24h_mask
        not_dmso_6h_or_24h_idx: NDArray[int] = not_dmso_6h_or_24h_mask.nonzero()[0]
        return cls(
            is_dmso_6h_mask=is_dmso_6h_mask,
            is_dmso_6h_idx=is_dmso_6h_idx,
            not_dmso_6h_mask=not_dmso_6h_mask,
            not_dmso_6h_idx=not_dmso_6h_idx,
            is_dmso_24h_mask=is_dmso_24h_mask,
            is_dmso_24h_idx=is_dmso_24h_idx,
            not_dmso_6h_or_24h_mask=not_dmso_6h_or_24h_mask,
            not_dmso_6h_or_24h_idx=not_dmso_6h_or_24h_idx
        )


class LsagneModel(nn.Module, SamplesEmbedder):
    def __init__(
            self,
            train_cmap_dataset: RawCmapDataset,
            latent_dim: int,
            input_dim: int,
            encoder_layers: List[int],
            decoder_layers: List[int],
            clouds_classifier_inner_layers: List[int],
            tissues_classifier_inner_layers: List[int],
            logger: logging.Logger,
            device: Device,
            contrastive_margin: float,
            enable_vae_skip_connection: bool,
            enable_triangle: bool,
            default_linear_layer: Type[nn.Linear],
            add_class_weights_to_classifiers: bool,
            treatment_and_drug_vectors_distance_loss_cdist_usage: bool,
            different_directions_loss_power_factor: float,
            use_untrained_clouds_predictions_in_training: bool,
            untrained_cloud_refs: Collection[CmapCloudRef],
            perturbations_equivalence_sets: Collection[Set[Perturbation]]
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.train_cmap_dataset = train_cmap_dataset
        self.logger = logger
        self.treatment_and_drug_vectors_distance_loss_cdist_usage = treatment_and_drug_vectors_distance_loss_cdist_usage
        logger.info(f'The model is using {SimpleVaeNet.__name__} inner layers: encoder={encoder_layers}, decoder={decoder_layers}')
        self.vae = SimpleVaeNet(input_dim, encoder_layers, latent_dim, decoder_layers, enable_vae_skip_connection, default_linear_layer)
        # self.vae = SparseVaeNet(latent_dim, input_dim, sparse=True, noise_fixed=True)
        logger.info(f'Vae net: {self.vae}')
        cloud_classifier_weights = train_cmap_dataset.cloud_ref_to_balanced_class_weights if add_class_weights_to_classifiers else None
        self.clouds_classifier = SimpleClassifier(
            input_dim=latent_dim,
            inner_dims=clouds_classifier_inner_layers,
            all_classes=train_cmap_dataset.unique_cloud_refs +
                        list(untrained_cloud_refs if use_untrained_clouds_predictions_in_training else []),
            class_to_class_weights=cloud_classifier_weights,
            layer_type=default_linear_layer,
            reduction='none'
        )
        logger.info(f'clouds classifier net: {self.clouds_classifier}')
        tissues_classifier_weights = train_cmap_dataset.tissue_to_balanced_class_weights if add_class_weights_to_classifiers else None
        self.tissues_classifier = SimpleClassifier(
            input_dim=latent_dim,
            inner_dims=tissues_classifier_inner_layers,
            all_classes=train_cmap_dataset.tissues_unique,
            class_to_class_weights=tissues_classifier_weights,
            layer_type=default_linear_layer)
        logger.info(f'tissues classifier net: {self.tissues_classifier}')
        print(f'margin is {contrastive_margin}.')
        self.online_contrastive = OnlineContrastiveLoss(contrastive_margin, HardNegativePairSelector(device))
        # self.online_contrastive = OnlineContrastiveLoss(margin, HardNegativePairSelector())
        # self.online_contrastive = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
        # self.online_contrastive = OnlineContrastiveLoss(margin, AllPositivePairSelector())
        # self.online_contrastive = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin))
        # self.class_weights = multi_device_data.on_device['class_weights']
        # https://discuss.pytorch.org/t/is-the-cuda-operation-performed-in-place/84961
        self.device = device
        self.to(device)
        if self.device.type == 'cuda':
            assert next(self.parameters()).is_cuda, 'Model must be on cuda device.'
        self.enable_triangle = enable_triangle
        self.add_class_weights_to_classifiers = add_class_weights_to_classifiers
        self.different_directions_loss_power_factor = different_directions_loss_power_factor
        self.use_untrained_clouds_predictions_in_training = use_untrained_clouds_predictions_in_training
        self.untrained_cloud_refs = untrained_cloud_refs
        self.perturbations_equivalence_sets = perturbations_equivalence_sets

    @cached_property
    def _0_loss(self):
        return torch.tensor(0.0, device=self.device)

    # TODO do I need this?
    def train(self, mode: bool = True):
        self.vae.train(mode)

    def eval(self):
        self.vae.eval()

    def update_training_cmap(self, train_cmap_dataset: RawCmapDataset):
        self.logger.info("Model is updating CMAP...")
        self.train_cmap_dataset = train_cmap_dataset
        if self.add_class_weights_to_classifiers:
            self.clouds_classifier.update_class_weights_array(train_cmap_dataset.cloud_ref_to_balanced_class_weights)
            self.tissues_classifier.update_class_weights_array(train_cmap_dataset.tissue_to_balanced_class_weights)

    @staticmethod
    def normalize_tensor(x: Tensor, dim: int = 1) -> Tensor:
        return F.normalize(x, 2, dim=dim)

    def forward(self, x):
        return self.get_embedding(x)

    def loss_fn(
            self,
            i_epoch: int,
            i_batch: int,
            samples_embedding: SamplesEmbedding,
            batch_multi_device: MultiDeviceData,
            loss_coef: Dict[str, float],
            desired_max_cloud_radius: float,
            is_warmup_batch: bool,
            treatment_vectors_magnitude_regulator_relu_coef: float,
            drug_vectors_magnitude_regulator_relu_coef: float,
            perturbations_equivalence_sets_loss_coef: Dict[str, float]
            ) -> Dict[str, torch.Tensor]:
        z_t = samples_embedding.z_t
        mu_t = samples_embedding.mu_t
        log_var_t = samples_embedding.log_var_t
        z_std_t = self.safe_std(z_t)
        if i_epoch * i_batch % 50 == 0:
            # self.logger.info(f'std == {z_std_t}', extra=MICHAEL_NOT_INTERESTED_ANNOTATION)
            pass
        embedded_anchors_and_vectors = EmbeddedAnchorsAndVectors.create(
            original_space_anchor_points_lookup=batch_multi_device.on_device_original_space_anchor_points,
            embedder=self
        )

        masks = _BatchMasks.create(batch_multi_device)

        batch_cloud_refs: NDArray[CmapCloudRef] = batch_multi_device.on_cpu.cloud_refs
        unique_batch_cloud_refs: Set[CmapCloudRef] = set(batch_cloud_refs)
        unique_batch_dmso_6h_cloud_refs: Set[CmapCloudRef] = set(batch_cloud_refs[masks.cpu.is_dmso_6h_idx])
        unique_batch_not_dmso_6h_cloud_refs: Set[CmapCloudRef] = set(batch_cloud_refs[masks.cpu.not_dmso_6h_idx])
        unique_batch_dmso_24h_cloud_refs: Set[CmapCloudRef] = set(batch_cloud_refs[masks.cpu.is_dmso_24h_idx])
        unique_batch_not_dmso_6h_or_24h_cloud_refs: Set[CmapCloudRef] = set(batch_cloud_refs[masks.cpu.not_dmso_6h_or_24h_idx])


        # cloud_ref_to_mask: Dict[CmapCloudRef, NDArray[bool]] = {
        #     cloud_ref: (batch_cloud_refs == cloud_ref) for cloud_ref in unique_batch_cloud_refs
        # }
        cloud_ref_to_mask_t: Dict[CmapCloudRef, Tensor] = {
            cloud_ref: batch_multi_device.on_device.encoded_labels == self.train_cmap_dataset.cloud_ref_to_encoded_label[cloud_ref]
            for cloud_ref in unique_batch_cloud_refs
        }
        cloud_ref_to_batch_samples_t = {
            cloud_ref: z_t[mask_t] for cloud_ref, mask_t in cloud_ref_to_mask_t.items()
        }

        # Calculate losses
        vae_mse_loss = self._0_loss if loss_coef['vae_mse'] == 0 else self.vae_mse_loss(z_t, batch_multi_device.on_device.raw_samples)
        vae_kld_loss = self._0_loss if loss_coef['vae_kld'] == 0 else self.vae_kld_loss(mu_t, log_var_t)
        vae_l1_regularization_loss = self._0_loss if loss_coef['vae_l1_regularization'] == 0 else self.vae.l1_regularization_loss()
        online_contrastive_loss = self._0_loss if loss_coef['online_contrastive'] == 0 else self.online_contrastive(z_t, batch_multi_device.on_device.encoded_labels)

        distance_from_cloud_center_dmso_6h_loss, l2_distance_from_cloud_center_dmso_6h_t = (self._0_loss, None) \
            if loss_coef['distance_from_cloud_center_6h'] == 0 \
               and loss_coef['max_radius_limiter'] == 0 \
               and loss_coef['treatment_vectors_magnitude_regulator'] == 0 \
               and loss_coef['drug_vectors_magnitude_regulator'] \
            else self.distance_from_cloud_center_loss(
                z_t=z_t[masks.device.is_dmso_6h_idx_t],
                cloud_refs=batch_cloud_refs[masks.cpu.is_dmso_6h_idx],
                cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center,
                z_std_t=z_std_t
            )

        distance_from_cloud_center_dmso_24h_loss, l2_distance_from_cloud_center_dmso_24h_t = (self._0_loss, None) \
            if loss_coef['distance_from_cloud_center_dmso_24h'] == 0 and loss_coef['max_radius_limiter'] == 0 \
            else self.distance_from_cloud_center_loss(
                z_t=z_t[masks.device.is_dmso_24h_idx_t],
                cloud_refs=batch_cloud_refs[masks.cpu.is_dmso_24h_idx],
                cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center,
                z_std_t=z_std_t
            )

        distance_from_cloud_center_treated_without_dmso_24h_loss, l2_distance_from_cloud_center_treated_without_dmso_24h_t = (self._0_loss, None) \
            if loss_coef['distance_from_cloud_center_24h_without_dmso_24h'] == 0 and loss_coef[
            'max_radius_limiter'] == 0 \
            else self.distance_from_cloud_center_loss(
                z_t=z_t[masks.device.not_dmso_6h_or_24h_mask_t],
                cloud_refs=batch_cloud_refs[masks.cpu.not_dmso_6h_or_24h_mask],
                cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center,
                z_std_t=z_std_t
            )

        max_radius_limiter_loss_dmso_6h = self._0_loss \
            if loss_coef['max_radius_limiter'] == 0 \
            else self.max_radius_limiter_loss(
                desired_max_cloud_radius=desired_max_cloud_radius,
                l2_distance_from_center_t=l2_distance_from_cloud_center_dmso_6h_t
            )

        max_radius_limiter_loss_dmso_24h = self._0_loss \
            if loss_coef['max_radius_limiter'] == 0 \
            else self.max_radius_limiter_loss(
                desired_max_cloud_radius=desired_max_cloud_radius,
                l2_distance_from_center_t=l2_distance_from_cloud_center_dmso_24h_t
            )

        max_radius_limiter_loss_treated_no_dmso_24h = self._0_loss \
            if loss_coef['max_radius_limiter'] == 0 \
            else self.max_radius_limiter_loss(
                desired_max_cloud_radius=desired_max_cloud_radius,
                l2_distance_from_center_t=l2_distance_from_cloud_center_treated_without_dmso_24h_t
            )

        max_radius_limiter_loss = max_radius_limiter_loss_dmso_6h + max_radius_limiter_loss_treated_no_dmso_24h + max_radius_limiter_loss_dmso_24h

        no_treated_samples_in_this_batch: bool = len(masks.cpu.not_dmso_6h_idx) == 0
        if no_treated_samples_in_this_batch:
            self.logger.warning(f'No treated samples in this batch #{i_batch}.')

        if no_treated_samples_in_this_batch or(
                loss_coef['treatment_vectors_collinearity_using_batch_treated'] == 0 and
                loss_coef['treatment_vectors_collinearity_using_batch_control'] == 0 and
                loss_coef['drug_vectors_collinearity_using_batch_treated'] == 0 and
                loss_coef['drug_vectors_collinearity_using_batch_control'] == 0 and
                loss_coef['treatment_vectors_different_directions_using_anchors'] == 0 and
                loss_coef['drug_and_treatment_vectors_different_directions_using_anchors'] == 0 and
                loss_coef['distance_from_treatment_vector_predicted'] == 0 and
                loss_coef['distance_from_drug_vector_predicted'] == 0 and
                loss_coef['treatment_and_drug_vectors_distance_p1_p2_loss'] == 0 and
                loss_coef['treatment_and_drug_vectors_distance_p1_p2_to_treated_loss'] == 0 and
                loss_coef['treatment_vectors_magnitude_regulator'] == 0 and
                loss_coef['drug_vectors_magnitude_regulator'] == 0):
                treatment_vectors_collinearity_using_batch_treated_loss = \
                treatment_vectors_collinearity_using_batch_control_loss = \
                drug_vectors_collinearity_using_batch_treated_loss = \
                drug_vectors_collinearity_using_batch_control_loss = \
                treatment_vectors_different_directions_using_anchors_loss = \
                drug_and_treatment_vectors_different_directions_using_anchors_loss = \
                treatment_vectors_different_directions_using_batch_loss = \
                drug_and_treatment_vectors_different_directions_using_batch_loss = \
                distance_from_treatment_vector_predicted_loss = \
                distance_from_drug_vector_predicted_loss = \
                treatment_and_drug_vectors_distance_p1_p2_loss = \
                treatment_and_drug_vectors_distance_p1_p2_to_treated_loss = \
                treatment_vectors_magnitude_regulator_loss = \
                drug_vectors_magnitude_regulator_loss = self._0_loss
        else:
            treated_max_radius_t = 3 * self.get_max_radius(
                cloud_ref_to_batch_samples_t=cloud_ref_to_batch_samples_t,
                relevant_cloud_refs=unique_batch_not_dmso_6h_or_24h_cloud_refs,
                cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center
            )

            dmso_6h_samples_cloud_ref_to_within_radius_samples_t = \
                self.calculate_relevant_cloud_ref_to_within_radius_samples_t(
                    cloud_ref_to_batch_samples_t=cloud_ref_to_batch_samples_t,
                    relevant_cloud_refs=unique_batch_dmso_6h_cloud_refs,
                    treated_max_radius_t=treated_max_radius_t,
                    cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center
                )

            batch_anchor_treatment_vectors_t, \
            batch_dmso_6h_anchor_to_real_treated_vectors_t, \
            batch_normalized_anchor_treatment_vectors_t, \
            batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t, \
            batch_treatment_non_zero_mask_using_real_treated_vectors_t = \
                self.calculate_batch_vectors_t_generic_using_treated_samples(
                    z_t=z_t,
                    relevant_treated_idx=masks.cpu.not_dmso_6h_idx,
                    relevant_treated_idx_t=masks.device.not_dmso_6h_idx_t,
                    perturbations=batch_multi_device.on_cpu.perturbations,
                    tissues=batch_multi_device.on_cpu.tissues,
                    tissue_to_control_sample=embedded_anchors_and_vectors.tissue_to_anchor_dmso_6h,
                    perturbation_to_anchor_vector=embedded_anchors_and_vectors.perturbation_to_anchor_treatment_vector
                )
            assert_normalized(batch_normalized_anchor_treatment_vectors_t)
            assert_normalized(batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t)

            cloud_ref__to__dmso_6h_anchor_to_real_treated_nonzero_vectors_t = \
                self.calculate_cloud_ref_to_vectors_t(
                    vectors=batch_dmso_6h_anchor_to_real_treated_vectors_t[batch_treatment_non_zero_mask_using_real_treated_vectors_t],
                    cloud_refs=batch_cloud_refs[masks.device.not_dmso_6h_idx_t[batch_treatment_non_zero_mask_using_real_treated_vectors_t].cpu().numpy()]
                )

            treatment_vectors_collinearity_using_batch_treated_loss = self._0_loss \
                if loss_coef['treatment_vectors_collinearity_using_batch_treated'] == 0 \
                else self.collinearity_loss(
                    non_zero_normalized_anchor_vectors_t=batch_normalized_anchor_treatment_vectors_t[batch_treatment_non_zero_mask_using_real_treated_vectors_t],
                    non_zero_normalized_predicted_control_anchor_to_z_treatment_vectors_t=batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t[batch_treatment_non_zero_mask_using_real_treated_vectors_t]
                )

            _, _, \
            normalized_anchor_treatment_vectors_using_batch_control_t, \
            normalized_batch_real_control_to_anchor_treated_t, \
            treatment_non_zero_mask_using_batch_control_t = \
                self.calculate_batch_vectors_t_generic_using_control_samples(
                    relevant_treated_idx=masks.cpu.not_dmso_6h_mask,
                    cloud_refs=batch_cloud_refs,
                    cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center,
                    perturbation_to_anchor_vector=embedded_anchors_and_vectors.perturbation_to_anchor_treatment_vector,
                    batch_control_cloud_ref_to_samples_t=dmso_6h_samples_cloud_ref_to_within_radius_samples_t
                )
            assert_normalized(normalized_anchor_treatment_vectors_using_batch_control_t)
            assert_normalized(normalized_batch_real_control_to_anchor_treated_t)

            treatment_vectors_collinearity_using_batch_control_loss = self._0_loss \
                if loss_coef['treatment_vectors_collinearity_using_batch_control'] == 0 \
                else self.collinearity_loss(
                    non_zero_normalized_anchor_vectors_t=normalized_anchor_treatment_vectors_using_batch_control_t[treatment_non_zero_mask_using_batch_control_t],
                    non_zero_normalized_predicted_control_anchor_to_z_treatment_vectors_t=normalized_batch_real_control_to_anchor_treated_t[treatment_non_zero_mask_using_batch_control_t]
                )

            distance_from_treatment_vector_predicted_loss = self._0_loss

            no_treated_without_dmso_24h_samples_in_this_batch: bool = len(masks.cpu.not_dmso_6h_or_24h_idx) == 0
            if no_treated_without_dmso_24h_samples_in_this_batch:
                self.logger.warning('No treated samples without time-24h in this batch.')

            if (loss_coef['drug_vectors_collinearity_using_batch_treated'] == 0 and
                loss_coef['drug_vectors_collinearity_using_batch_control'] == 0 and
                loss_coef['treatment_and_drug_vectors_distance_p1_p2_loss'] == 0 and
                loss_coef['treatment_and_drug_vectors_distance_p1_p2_to_treated_loss'] == 0 and
                loss_coef['distance_from_drug_vector_predicted'] == 0 and
                loss_coef['treatment_vectors_different_directions_using_anchors'] == 0 and
                loss_coef['drug_and_treatment_vectors_different_directions_using_anchors'] == 0 and
                loss_coef['treatment_vectors_different_directions_using_batch'] == 0 and
                loss_coef['drug_and_treatment_vectors_different_directions_using_batch'] == 0 and
                loss_coef['treatment_vectors_magnitude_regulator'] == 0 and
                loss_coef['drug_vectors_magnitude_regulator'] == 0) or \
                    no_treated_without_dmso_24h_samples_in_this_batch:
                    drug_vectors_collinearity_using_batch_treated_loss = \
                    drug_vectors_collinearity_using_batch_control_loss = \
                    treatment_and_drug_vectors_distance_p1_p2_loss = \
                    treatment_and_drug_vectors_distance_p1_p2_to_treated_loss = \
                    treatment_vectors_different_directions_using_anchors_loss = \
                    drug_and_treatment_vectors_different_directions_using_anchors_loss = \
                    treatment_vectors_different_directions_using_batch_loss = \
                    drug_and_treatment_vectors_different_directions_using_batch_loss = \
                    distance_from_drug_vector_predicted_loss = \
                    treatment_vectors_magnitude_regulator_loss = \
                    drug_vectors_magnitude_regulator_loss = self._0_loss
            else:
                _, \
                batch_anchor_dmso_24h_to_real_treated_t, \
                batch_normalized_anchor_drug_vectors_t, \
                batch_normalized_anchor_dmso_24h_to_real_treated_t, \
                batch_drug_non_zero_mask_using_real_treated_t = \
                    self.calculate_batch_vectors_t_generic_using_treated_samples(
                        z_t=z_t,
                        relevant_treated_idx=masks.cpu.not_dmso_6h_or_24h_idx,
                        relevant_treated_idx_t=masks.device.not_dmso_6h_or_24h_idx_t,
                        perturbations=batch_multi_device.on_cpu.perturbations,
                        tissues=batch_multi_device.on_cpu.tissues,
                        tissue_to_control_sample=embedded_anchors_and_vectors.tissue_to_anchor_dmso_24h,
                        perturbation_to_anchor_vector=embedded_anchors_and_vectors.perturbation_to_anchor_drug_vector
                    )

                cloud_ref__to__dmso_24h_anchor_to_real_treated_nonzero_vectors_t = \
                    self.calculate_cloud_ref_to_vectors_t(
                        vectors=batch_anchor_dmso_24h_to_real_treated_t[batch_drug_non_zero_mask_using_real_treated_t],
                        cloud_refs=batch_cloud_refs[masks.cpu.not_dmso_6h_or_24h_idx][batch_drug_non_zero_mask_using_real_treated_t.cpu().numpy()]
                    )

                drug_vectors_collinearity_using_batch_treated_loss = self._0_loss \
                    if loss_coef['drug_vectors_collinearity_using_batch_treated'] == 0 \
                    else self.collinearity_loss(
                        non_zero_normalized_anchor_vectors_t=batch_normalized_anchor_drug_vectors_t[batch_drug_non_zero_mask_using_real_treated_t],
                        non_zero_normalized_predicted_control_anchor_to_z_treatment_vectors_t=batch_normalized_anchor_dmso_24h_to_real_treated_t[batch_drug_non_zero_mask_using_real_treated_t]
                    )

                dmso_24h_samples_cloud_ref_to_within_radius_samples_t = \
                    self.calculate_relevant_cloud_ref_to_within_radius_samples_t(
                        cloud_ref_to_batch_samples_t=cloud_ref_to_batch_samples_t,
                        relevant_cloud_refs=unique_batch_dmso_24h_cloud_refs,
                        treated_max_radius_t=treated_max_radius_t,
                        cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center
                    )

                _, _, \
                normalized_dmso_24h_drug_vectors_using_batch_control_t, \
                normalized_predicted_dmso_24h_anchor_to_z_drug_vectors_using_batch_control_t, \
                drug_non_zero_mask_using_batch_control_t = \
                    self.calculate_batch_vectors_t_generic_using_control_samples(
                        relevant_treated_idx=masks.cpu.not_dmso_6h_or_24h_idx,
                        cloud_refs=batch_cloud_refs,
                        cloud_ref_to_cloud_center=embedded_anchors_and_vectors.cloud_ref_to_cloud_center,
                        perturbation_to_anchor_vector=embedded_anchors_and_vectors.perturbation_to_anchor_drug_vector,
                        batch_control_cloud_ref_to_samples_t=dmso_24h_samples_cloud_ref_to_within_radius_samples_t
                    )

                drug_vectors_collinearity_using_batch_control_loss = self._0_loss \
                    if loss_coef['drug_vectors_collinearity_using_batch_control'] == 0 \
                    else self.collinearity_loss(
                        non_zero_normalized_anchor_vectors_t=normalized_dmso_24h_drug_vectors_using_batch_control_t[drug_non_zero_mask_using_batch_control_t],
                        non_zero_normalized_predicted_control_anchor_to_z_treatment_vectors_t=normalized_predicted_dmso_24h_anchor_to_z_drug_vectors_using_batch_control_t[drug_non_zero_mask_using_batch_control_t]
                    )

                treatment_vectors_different_directions_using_anchors_loss, drug_and_treatment_vectors_different_directions_using_anchors_loss = (self._0_loss, self._0_loss) \
                    if loss_coef['treatment_vectors_different_directions_using_anchors'] == 0 and loss_coef['drug_and_treatment_vectors_different_directions_using_anchors'] == 0 \
                    else self.different_directions_using_anchors_loss(
                        perturbation_to_anchor_drug_vector=embedded_anchors_and_vectors.perturbation_to_anchor_drug_vector,
                        perturbation_to_anchor_treatment_vector=embedded_anchors_and_vectors.perturbation_to_anchor_treatment_vector,
                        perturbations_equivalence_sets=self.perturbations_equivalence_sets,
                        treatment_vectors_perturbations_equivalence_set_loss_coef=perturbations_equivalence_sets_loss_coef['treatment_vectors_different_directions_using_anchors'],
                    )

                treatment_vectors_different_directions_using_batch_loss, \
                drug_and_treatment_vectors_different_directions_using_batch_loss = (self._0_loss, self._0_loss) \
                    if loss_coef['treatment_vectors_different_directions_using_batch'] == 0 and \
                       loss_coef['drug_and_treatment_vectors_different_directions_using_batch'] == 0 \
                    else self.different_directions_using_batch_loss(
                        treatment_non_zero_mask_using_batch_treated_t=batch_treatment_non_zero_mask_using_real_treated_vectors_t,
                        drug_non_zero_mask_using_batch_treated_t=batch_drug_non_zero_mask_using_real_treated_t,
                        not_dmso_6h_mask=masks.cpu.not_dmso_6h_mask,
                        not_dmso_6h_mask_t=masks.device.not_dmso_6h_mask_t,
                        not_dmso_6h_or_24h_mask_t=masks.device.not_dmso_6h_or_24h_mask_t,
                        batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t=batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t,
                        batch_normalized_anchor_dmso_24h_to_real_treated_t=batch_normalized_anchor_dmso_24h_to_real_treated_t,
                        batch_perturbations=batch_multi_device.on_cpu.perturbations,
                        perturbation_to_anchor_treatment_vector=embedded_anchors_and_vectors.perturbation_to_anchor_treatment_vector,
                        perturbations_equivalence_sets=self.perturbations_equivalence_sets,
                        treatment_vectors_perturbations_equivalence_set_loss_coef=perturbations_equivalence_sets_loss_coef['treatment_vectors_different_directions_using_batch']
                    )

                (treatment_and_drug_vectors_distance_p1_p2_loss, treatment_and_drug_vectors_distance_p1_p2_to_treated_loss) = (self._0_loss, self._0_loss) \
                    if loss_coef['treatment_and_drug_vectors_distance_p1_p2_loss'] == 0 and \
                       loss_coef['treatment_and_drug_vectors_distance_p1_p2_to_treated_loss'] == 0 \
                    else self.treatment_and_drug_vectors_distance_loss(
                        cloud_ref_to_reference_treatment_vector=embedded_anchors_and_vectors.cloud_ref_to_reference_treatment_vector,
                        cloud_ref_to_reference_drug_vector=embedded_anchors_and_vectors.cloud_ref_to_reference_drug_vector,
                        dmso_6h_and_24h_cloud_ref_to_within_radius_samples_t={
                            **dmso_6h_samples_cloud_ref_to_within_radius_samples_t,
                            **dmso_24h_samples_cloud_ref_to_within_radius_samples_t
                        },
                        cloud_ref_to_samples_t=cloud_ref_to_batch_samples_t,
                        z_std_t=z_std_t
                    )

                distance_from_drug_vector_predicted_loss = self._0_loss

                tissue_to_mean_dmso_6h_radius_t = self.calculate_tissue_to_mean_cloud_radius_t(
                    tissues=batch_multi_device.on_cpu.tissues[masks.cpu.is_dmso_6h_idx],
                    l2_distance_from_center_t=l2_distance_from_cloud_center_dmso_6h_t
                )

                treatment_vectors_magnitude_regulator_loss = self._0_loss \
                    if loss_coef['treatment_vectors_magnitude_regulator'] == 0 \
                    else self.vectors_magnitude_regulator_loss(
                        cloud_ref_to_vectors_t=cloud_ref__to__dmso_6h_anchor_to_real_treated_nonzero_vectors_t,
                        tissue_to_dmso_6h_mean_cloud_radius_t=tissue_to_mean_dmso_6h_radius_t,
                        distance_coef=treatment_vectors_magnitude_regulator_relu_coef
                    )

                drug_vectors_magnitude_regulator_loss = self._0_loss \
                    if loss_coef['drug_vectors_magnitude_regulator'] == 0 \
                    else self.vectors_magnitude_regulator_loss(
                        cloud_ref_to_vectors_t=cloud_ref__to__dmso_24h_anchor_to_real_treated_nonzero_vectors_t,
                        tissue_to_dmso_6h_mean_cloud_radius_t=tissue_to_mean_dmso_6h_radius_t,
                        distance_coef=drug_vectors_magnitude_regulator_relu_coef
                    )

        std_1_loss = self._0_loss \
                    if loss_coef['std_1'] == 0 \
                    else torch.pow(torch.abs(1 - z_std_t) + 1, 10)

        classifier_z_t, \
        classifier_cloud_refs, \
        classifier_perturbations, \
        classifier_tissues = self.predict_clouds_for_classifier(
            z_t=z_t,
            cloud_refs=batch_multi_device.on_cpu.cloud_refs,
            tissues=batch_multi_device.on_cpu.tissues,
            perturbations=batch_multi_device.on_cpu.perturbations,
            cloud_ref_to_batch_samples_t=cloud_ref_to_batch_samples_t,
            perturbation_to_anchor_treatment_vector=embedded_anchors_and_vectors.perturbation_to_anchor_treatment_vector,
            perturbation_to_anchor_drug_vector=embedded_anchors_and_vectors.perturbation_to_anchor_drug_vector,
        ) if self.use_untrained_clouds_predictions_in_training else (
            z_t,
            batch_multi_device.on_device.encoded_labels,
            batch_multi_device.on_device.encoded_tissues,
            batch_multi_device.on_cpu.perturbations,
            batch_multi_device.on_cpu.tissues
        )

        clouds_classifier_loss = self._0_loss \
            if loss_coef['clouds_classifier'] == 0 \
            else self.clouds_classifier_loss(
                classifier=self.clouds_classifier,
                z_t=classifier_z_t,
                cloud_refs=classifier_cloud_refs,
                perturbations_equivalence_sets=self.perturbations_equivalence_sets,
                perturbations=classifier_perturbations,
                tissues=classifier_tissues,
                equivalence_sets_loss_coef=perturbations_equivalence_sets_loss_coef['clouds_classifier']
            )

        tissues_classifier_loss = self._0_loss \
            if loss_coef['tissues_classifier'] == 0 \
            else self.tissues_classifier_loss(
                classifier=self.tissues_classifier,
                z_t=classifier_z_t,
                tissues=classifier_tissues
            )

        return {
            'vae_mse': vae_mse_loss,
            'vae_kld': vae_kld_loss,
            'vae_l1_regularization': vae_l1_regularization_loss,
            'online_contrastive': online_contrastive_loss,
            'clouds_classifier': clouds_classifier_loss,
            'tissues_classifier': tissues_classifier_loss,
            'treatment_vectors_collinearity_using_batch_treated': treatment_vectors_collinearity_using_batch_treated_loss,
            'treatment_vectors_collinearity_using_batch_control': treatment_vectors_collinearity_using_batch_control_loss,
            'drug_vectors_collinearity_using_batch_treated': drug_vectors_collinearity_using_batch_treated_loss,
            'drug_vectors_collinearity_using_batch_control': drug_vectors_collinearity_using_batch_control_loss,
            'treatment_vectors_different_directions_using_anchors': treatment_vectors_different_directions_using_anchors_loss,
            'drug_and_treatment_vectors_different_directions_using_anchors': drug_and_treatment_vectors_different_directions_using_anchors_loss,
            'treatment_vectors_different_directions_using_batch': treatment_vectors_different_directions_using_batch_loss,
            'drug_and_treatment_vectors_different_directions_using_batch': drug_and_treatment_vectors_different_directions_using_batch_loss,
            'distance_from_treatment_vector_predicted': distance_from_treatment_vector_predicted_loss,
            'distance_from_drug_vector_predicted': distance_from_drug_vector_predicted_loss,
            'distance_from_cloud_center_6h': distance_from_cloud_center_dmso_6h_loss,
            'distance_from_cloud_center_dmso_24h': distance_from_cloud_center_dmso_24h_loss,
            'distance_from_cloud_center_24h_without_dmso_24h': distance_from_cloud_center_treated_without_dmso_24h_loss,
            'max_radius_limiter': max_radius_limiter_loss,
            'treatment_and_drug_vectors_distance_p1_p2_loss': treatment_and_drug_vectors_distance_p1_p2_loss,
            'treatment_and_drug_vectors_distance_p1_p2_to_treated_loss': treatment_and_drug_vectors_distance_p1_p2_to_treated_loss,
            'treatment_vectors_magnitude_regulator': treatment_vectors_magnitude_regulator_loss,
            'drug_vectors_magnitude_regulator': drug_vectors_magnitude_regulator_loss,
            'std_1': std_1_loss
        }

    def calculate_cloud_sample_radii_norm_from_ref(
            self,
            cloud_samples_t: Tensor,
            cloud_center_ref_t: Tensor
    ) -> Tensor:
        treated_radii_t = cloud_samples_t - cloud_center_ref_t
        return torch.norm(treated_radii_t, p=2, dim=1)

    @torch.no_grad()
    def get_max_radius(
            self,
            cloud_ref_to_batch_samples_t: Dict[CmapCloudRef, Tensor],
            relevant_cloud_refs: Collection[CmapCloudRef],
            cloud_ref_to_cloud_center: LookupTensor
    ) -> Tensor:
        """
        :return: max radius of all samples
        """
        treated_radii = []
        for cloud_ref, batch_samples_z_t in cloud_ref_to_batch_samples_t.items():
            if cloud_ref in relevant_cloud_refs:
                treated_radii.append(batch_samples_z_t - cloud_ref_to_cloud_center[cloud_ref])

        treated_radii_t = torch.vstack(treated_radii)
        treated_radii_norm_t = torch.norm(treated_radii_t, p=2, dim=1)
        # todo maybe use quantile 0.95?
        return treated_radii_norm_t.max()

    def vae_mse_loss(self, z_t: torch.Tensor, x_input_t: torch.Tensor) -> Tensor:
        x_decoded_t = self.vae.decode(z_t)
        vae_mse_loss = self.vae.reconstruction_loss(x_decoded_t, x_input_t)
        return vae_mse_loss

    def vae_kld_loss(self, mu, log_var) -> Tensor:
        return self.vae.kld_loss(mu, log_var)

    def safe_std(self, x: Tensor):
        # if the batch is not balanced, we need to consider it here.
        x = x.detach()
        return torch.linalg.norm(x.std(dim=0))

    def normalize_and_check_zero_for_predicted_and_anchor_vectors_t(
            self,
            batch_anchor_vectors_t: Tensor,
            batch_predicted_vectors_t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        epsilon = 1e-5
        with torch.no_grad():
            non_zero_anchor_vectors_mask_t: Tensor = batch_anchor_vectors_t.norm(p=2, dim=1) > epsilon
            if torch.any(~non_zero_anchor_vectors_mask_t):
                self.logger.warning('At least one anchor vector has zero norm!.')
            non_zero_predicted_vectors_mask_t: Tensor = batch_predicted_vectors_t.norm(p=2, dim=1) > epsilon
            non_zero_mask_t: Tensor = torch.logical_and(non_zero_anchor_vectors_mask_t, non_zero_predicted_vectors_mask_t)
            if torch.all(~non_zero_mask_t):
                raise Exception('Something is off!')
            if non_zero_mask_t.sum() < len(non_zero_mask_t):
                self.logger.warning('At least one vector is non zero.')
        normalized_anchor_vectors = self.normalize_tensor(batch_anchor_vectors_t)
        normalized_predicted_vectors = self.normalize_tensor(batch_predicted_vectors_t)
        return normalized_anchor_vectors, normalized_predicted_vectors, non_zero_mask_t

    def calculate_batch_vectors_t_generic_using_treated_samples(
            self,
            z_t: Tensor,
            relevant_treated_idx: NDArray[int],
            relevant_treated_idx_t: Tensor,
            perturbations: NDArray[Perturbation],
            tissues: NDArray[Tissue],
            tissue_to_control_sample: LookupTensor[Tissue],
            perturbation_to_anchor_vector: LookupTensor[Perturbation]
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        relevant_samples_count = len(relevant_treated_idx)
        batch_anchor_vectors_t_builder = FastTensorBuilder(relevant_samples_count)
        treated_perturbations = perturbations[relevant_treated_idx]
        for treated_perturbation in np.unique(treated_perturbations):
            mask = treated_perturbations == treated_perturbation
            batch_anchor_vectors_t_builder.add_slice(
                mask,
                perturbation_to_anchor_vector[treated_perturbation]
            )
        batch_anchor_vectors_t = batch_anchor_vectors_t_builder.create_tensor()

        batch_tissue_control_samples_t_builder = FastTensorBuilder(relevant_samples_count)
        treated_tissues: NDArray = tissues[relevant_treated_idx]
        for tissue in np.unique(treated_tissues):
            mask = treated_tissues == tissue
            batch_tissue_control_samples_t_builder.add_slice(mask, tissue_to_control_sample[tissue])
        batch_tissue_control_samples_t = batch_tissue_control_samples_t_builder.create_tensor()

        z_treated_t = z_t[relevant_treated_idx_t]
        batch_predicted_vectors_t = z_treated_t - batch_tissue_control_samples_t

        normalized_anchor_vectors, normalized_predicted_vectors, non_zero_mask_t = \
            self.normalize_and_check_zero_for_predicted_and_anchor_vectors_t(
                batch_anchor_vectors_t=batch_anchor_vectors_t,
                batch_predicted_vectors_t=batch_predicted_vectors_t
            )

        return batch_anchor_vectors_t, batch_predicted_vectors_t, normalized_anchor_vectors, normalized_predicted_vectors, non_zero_mask_t

    def calculate_batch_vectors_t_generic_using_control_samples(
            self,
            relevant_treated_idx: NDArray[int],
            cloud_refs: NDArray[CmapCloudRef],
            cloud_ref_to_cloud_center: LookupTensor,
            perturbation_to_anchor_vector: LookupTensor[Perturbation],
            batch_control_cloud_ref_to_samples_t: Dict[CmapCloudRef, Tensor],
            ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        batch_control_tissue_to_samples_t: Dict[Tissue, Tensor] = {
            cloud_ref.tissue: samples_t for cloud_ref, samples_t in batch_control_cloud_ref_to_samples_t.items()
        }

        unstacked_batch_anchor_vectors_t = []
        unstacked_batch_predicted_vectors_t = []
        for treated_cloud_ref in set(cloud_refs[relevant_treated_idx]):
            treated_anchor_t = cloud_ref_to_cloud_center[treated_cloud_ref]
            untreated_samples_t = batch_control_tissue_to_samples_t[treated_cloud_ref.tissue]
            predicted_vectors_t = treated_anchor_t - untreated_samples_t
            unstacked_batch_predicted_vectors_t.append(predicted_vectors_t)
            anchor_vector_t = perturbation_to_anchor_vector[treated_cloud_ref.perturbation]
            unstacked_batch_anchor_vectors_t.append(anchor_vector_t.repeat(len(predicted_vectors_t), 1))

        batch_anchor_vectors_t = torch.vstack(unstacked_batch_anchor_vectors_t)
        batch_predicted_vectors_t = torch.vstack(unstacked_batch_predicted_vectors_t)

        normalized_anchor_vectors, normalized_predicted_vectors, non_zero_mask_t = \
            self.normalize_and_check_zero_for_predicted_and_anchor_vectors_t(
                batch_anchor_vectors_t=batch_anchor_vectors_t,
                batch_predicted_vectors_t=batch_predicted_vectors_t
            )
        assert_normalized(normalized_anchor_vectors)
        assert_normalized(normalized_predicted_vectors)

        return batch_anchor_vectors_t, batch_predicted_vectors_t, normalized_anchor_vectors, normalized_predicted_vectors, non_zero_mask_t

    def predict_clouds_for_classifier(
            self,
            z_t: Tensor,
            cloud_refs: NDArray[CmapCloudRef],
            tissues: NDArray[Tissue],
            perturbations: NDArray[Perturbation],
            cloud_ref_to_batch_samples_t: Dict[CmapCloudRef, Tensor],
            perturbation_to_anchor_treatment_vector: LookupTensor,
            perturbation_to_anchor_drug_vector: LookupTensor,
    ) -> Tuple[Tensor, NDArray[CmapCloudRef], NDArray[Perturbation], NDArray[Tissue]]:
        untrained_predicted_perturbations: List[NDArray[Perturbation]] = []
        untrained_predicted_tissues: List[NDArray[Tissue]] = []
        untrained_predicted_clouds_t: List[Tensor] = []
        untrained_predicted_cloud_refs: List[NDArray[CmapCloudRef]] = []
        for i, untrained_cloud_ref in enumerate(self.untrained_cloud_refs):
            dmso_24h_samples_t = cloud_ref_to_batch_samples_t[untrained_cloud_ref.dmso_24h]
            dmso_6h_samples_t = cloud_ref_to_batch_samples_t[untrained_cloud_ref.dmso_6h]

            p1_t, p2_t = predict_triangle_cloud(
                batch_tissue_untreated_6h_samples=dmso_6h_samples_t,
                batch_tissue_untreated_24h_samples=dmso_24h_samples_t,
                treatment_vector=perturbation_to_anchor_treatment_vector[untrained_cloud_ref.perturbation],
                drug_vector=perturbation_to_anchor_drug_vector[untrained_cloud_ref.perturbation]
            )
            predicted_cloud_t = (p1_t + p2_t) / 2
            untrained_predicted_clouds_t.append(predicted_cloud_t)
            cloud_size = len(predicted_cloud_t)
            untrained_predicted_cloud_refs.append(np.full(cloud_size, untrained_cloud_ref))
            untrained_predicted_perturbations.append(np.full(cloud_size, untrained_cloud_ref.perturbation))
            untrained_predicted_tissues.append(np.full(cloud_size, untrained_cloud_ref.tissue))

        classifier_z_t = torch.vstack([z_t, *untrained_predicted_clouds_t])
        classifier_cloud_refs = np.hstack([cloud_refs, *untrained_predicted_cloud_refs])
        classifier_perturbations = np.hstack([perturbations, *untrained_predicted_perturbations])
        classifier_tissues = np.hstack([tissues, *untrained_predicted_tissues])

        assert_same_len(classifier_z_t, classifier_cloud_refs, classifier_perturbations, classifier_tissues)
        return classifier_z_t, classifier_cloud_refs, classifier_perturbations, classifier_tissues

    def clouds_classifier_loss(
            self,
            classifier: SimpleClassifier,
            z_t: Tensor,
            cloud_refs: NDArray[CmapCloudRef],
            perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]] = None,
            tissues: Optional[NDArray[Tissue]] = None,
            perturbations: Optional[NDArray[Perturbation]] = None,
            equivalence_sets_loss_coef: float = 1.0
    ):
        classifier_results_t = classifier(z_t)
        if not perturbations_equivalence_sets or equivalence_sets_loss_coef < config.epsilon:
            return classifier.loss_fn(classifier_results_t, cloud_refs).mean()

        assert tissues is not None and perturbations is not None
        all_equivalence_sets_encoded_perturbations = list(set().union(*perturbations_equivalence_sets))
        not_in_any_equivalence_set_mask = ~np.isin(perturbations, all_equivalence_sets_encoded_perturbations)
        not_in_any_equivalence_set_loss_t = classifier.loss_fn(
            classifier_results_t[not_in_any_equivalence_set_mask],
            cloud_refs[not_in_any_equivalence_set_mask]
        )

        in_some_equivalence_set_mask = ~not_in_any_equivalence_set_mask
        in_equivalence_set_classifier_results_t = classifier_results_t[in_some_equivalence_set_mask]
        in_equivalence_set_classifier_labels = cloud_refs[in_some_equivalence_set_mask]
        in_equivalence_set_tissues = tissues[in_some_equivalence_set_mask]
        in_equivalence_set_perturbations = perturbations[in_some_equivalence_set_mask]

        in_equivalence_set_loss_t_list = []

        for perturbations_equivalence_set in perturbations_equivalence_sets:
            current_equivalence_set_mask = np.isin(in_equivalence_set_perturbations, perturbations_equivalence_set)
            current_equivalence_set_classifier_results_t = in_equivalence_set_classifier_results_t[current_equivalence_set_mask]
            current_equivalence_set_true_classifier_labels = in_equivalence_set_classifier_labels[current_equivalence_set_mask]
            current_equivalence_set_tissues = in_equivalence_set_tissues[current_equivalence_set_mask]
            current_equivalence_set_perturbations = in_equivalence_set_perturbations[current_equivalence_set_mask]
            for perturbation in perturbations_equivalence_set:
                classifier_labels_list = []
                non_existing_cloud_ref_mask_list = []
                perturbation_mask = current_equivalence_set_perturbations == perturbation
                current_perturbation_classifier_results_t = current_equivalence_set_classifier_results_t[perturbation_mask]
                current_perturbation_classifier_labels = current_equivalence_set_true_classifier_labels[perturbation_mask]
                classifier_labels_list.append(current_perturbation_classifier_labels)
                current_perturbation_tissues = current_equivalence_set_tissues[perturbation_mask]
                # The result has chucks as the number of different perturbations in the equivalence set.
                # The first chuck is the "real" classifications, and the other chucks are for the classification
                # of other perturbations in the equivalence set.
                results_t = torch.tile(current_perturbation_classifier_results_t, (len(perturbations_equivalence_set), 1))
                for other_perturbation in set(perturbations_equivalence_set).difference({perturbation}):
                    # None == CMAP does not contain this combination
                    classifier_labels = np.vectorize(
                        lambda tissue: CmapCloudRef(tissue, other_perturbation)
                        if CmapCloudRef(tissue, other_perturbation) in self.train_cmap_dataset.unique_cloud_refs
                        else None)(current_perturbation_tissues)
                    non_existent_clouds_mask = classifier_labels == None
                    non_existing_cloud_ref_mask_list.append(non_existent_clouds_mask)
                    # Put some random valid value, after classifier loss calculation,
                    # later these values will be swapped with torch.inf value
                    classifier_labels[non_existent_clouds_mask] = cloud_refs[0]
                    classifier_labels_list.append(classifier_labels)
                current_classifier_labels = np.hstack(classifier_labels_list)
                non_existing_cloud_ref_mask_list = np.hstack(non_existing_cloud_ref_mask_list)
                current_label_classifier_loss_t = classifier.loss_fn(results_t, current_classifier_labels)
                current_label_size = len(current_perturbation_classifier_labels)
                current_label_classifier_loss_t[current_label_size:][non_existing_cloud_ref_mask_list] = torch.inf
                current_label_classifier_loss_t = current_label_classifier_loss_t.unflatten(0, (len(perturbations_equivalence_set), current_label_size))
                true_t = current_label_classifier_loss_t[0]
                all_equivalence_set_clouds_t = current_label_classifier_loss_t[0:]
                all_equivalence_set_clouds_min_t = all_equivalence_set_clouds_t.min(dim=0)[0]
                loss_t = (1 - equivalence_sets_loss_coef) * true_t + equivalence_sets_loss_coef * all_equivalence_set_clouds_min_t
                in_equivalence_set_loss_t_list.append(loss_t)

        return torch.hstack([not_in_any_equivalence_set_loss_t, *in_equivalence_set_loss_t_list]).mean()

    def tissues_classifier_loss(
            self,
            classifier: SimpleClassifier,
            z_t: Tensor,
            tissues: NDArray[Tissue],
    ):
        classifier_results_t = classifier(z_t)
        return classifier.loss_fn(classifier_results_t, tissues).mean()

    def collinearity_loss(self,
                          non_zero_normalized_anchor_vectors_t: Tensor,
                          non_zero_normalized_predicted_control_anchor_to_z_treatment_vectors_t: Tensor
                          ) -> Tensor:
        assert_normalized(non_zero_normalized_anchor_vectors_t)
        assert_normalized(non_zero_normalized_predicted_control_anchor_to_z_treatment_vectors_t)
        dot_prod_t = torch.sum(non_zero_normalized_anchor_vectors_t * non_zero_normalized_predicted_control_anchor_to_z_treatment_vectors_t, dim=1)
        # if vectors are collinear the dot_prod should be 1 = cos(0)
        return (1.0 - dot_prod_t).mean()

    def _different_directions_loss(
            self,
            normalized_vectors_1: Tensor,
            normalized_vectors_2: Tensor,
            loss_pow_factor: float,
            perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]] = None,
            perturbations_equivalence_set_loss_coef: Optional[float] = None,
            vectors_1_perturbations: Optional[NDArray[Perturbation]] = None,
            vectors_2_perturbations: Optional[NDArray[Perturbation]] = None
    ):
        assert_normalized(normalized_vectors_1)
        assert_normalized(normalized_vectors_2)
        assert_same_len(normalized_vectors_1, normalized_vectors_2)
        if perturbations_equivalence_sets:
            assert_same_len(normalized_vectors_1, vectors_1_perturbations)
            assert_same_len(normalized_vectors_2, vectors_2_perturbations)
            assert_promise_true(lambda: (vectors_1_perturbations == vectors_2_perturbations).sum() == 0)
        # the dot_prod is the cosine of the angle between the vectors
        dot_prod_t = torch.sum(normalized_vectors_1 * normalized_vectors_2, dim=1)
        # scaled_between_0_and_1 = (dot_prod_t + 1) / 2
        # loss = torch.pow(scaled_between_0_and_1, loss_pow_factor)
        loss = torch.pow(dot_prod_t, loss_pow_factor)

        if perturbations_equivalence_sets:
            is_in_any_equivalence_set = np.full((len(normalized_vectors_1),), False)
            for equivalence_set in perturbations_equivalence_sets:
                in_equivalence_set = np.logical_and(
                    np.isin(vectors_1_perturbations, equivalence_set),
                    np.isin(vectors_2_perturbations, equivalence_set)
                )
                is_in_any_equivalence_set = np.logical_or(is_in_any_equivalence_set, in_equivalence_set)
            loss[is_in_any_equivalence_set] *= 1 - perturbations_equivalence_set_loss_coef

        return loss.mean()

    def different_directions_using_anchors_loss(
            self,
            perturbation_to_anchor_drug_vector: LookupTensor,
            perturbation_to_anchor_treatment_vector: LookupTensor,
            perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]],
            treatment_vectors_perturbations_equivalence_set_loss_coef: float,
    ) -> Tuple[Tensor, Tensor]:
        normalized_anchor_drug_vectors_t = self.normalize_tensor(perturbation_to_anchor_drug_vector.stacked_tensor)
        normalized_anchor_treatment_vectors_t = self.normalize_tensor(perturbation_to_anchor_treatment_vector.stacked_tensor)

        treatment_vectors_perturbations = np.array(list(perturbation_to_anchor_treatment_vector.keys()))
        with torch.no_grad():
            all_treatment_pairs_t = torch.combinations(torch.arange(len(normalized_anchor_treatment_vectors_t), device=self.device, dtype=torch.long))
        all_treatment_pairs = all_treatment_pairs_t.cpu().numpy()
        treatment_vectors_different_directions_loss = self._different_directions_loss(
            normalized_vectors_1=normalized_anchor_treatment_vectors_t[all_treatment_pairs_t[:, 0]],
            normalized_vectors_2=normalized_anchor_treatment_vectors_t[all_treatment_pairs_t[:, 1]],
            loss_pow_factor=self.different_directions_loss_power_factor,
            perturbations_equivalence_sets=perturbations_equivalence_sets,
            perturbations_equivalence_set_loss_coef=treatment_vectors_perturbations_equivalence_set_loss_coef,
            vectors_1_perturbations=treatment_vectors_perturbations[all_treatment_pairs[:, 0]],
            vectors_2_perturbations=treatment_vectors_perturbations[all_treatment_pairs[:, 1]]
        )

        # We want all the treatment and drug vectors to be different - there is no use for equivalence sets here
        all_treatment_and_drug_pairs_t = torch.cartesian_prod(
            torch.arange(len(normalized_anchor_drug_vectors_t), device=self.device, dtype=torch.long),
            torch.arange(len(normalized_anchor_treatment_vectors_t), device=self.device, dtype=torch.long)
        )
        drug_and_treatment_vectors_different_directions_loss = self._different_directions_loss(
            normalized_vectors_1=normalized_anchor_drug_vectors_t[all_treatment_and_drug_pairs_t[:, 0]],
            normalized_vectors_2=normalized_anchor_treatment_vectors_t[all_treatment_and_drug_pairs_t[:, 1]],
            loss_pow_factor=self.different_directions_loss_power_factor
        )

        return treatment_vectors_different_directions_loss, drug_and_treatment_vectors_different_directions_loss

    def different_directions_using_batch_loss(
            self,
            treatment_non_zero_mask_using_batch_treated_t: Tensor,
            drug_non_zero_mask_using_batch_treated_t: Tensor,
            not_dmso_6h_mask: NDArray[bool],
            not_dmso_6h_mask_t: Tensor,
            not_dmso_6h_or_24h_mask_t: Tensor,
            batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t: Tensor,
            batch_normalized_anchor_dmso_24h_to_real_treated_t: Tensor,
            batch_perturbations: NDArray[Perturbation],
            perturbation_to_anchor_treatment_vector: LookupTensor[Perturbation],
            perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]],
            treatment_vectors_perturbations_equivalence_set_loss_coef: float,
    ) -> Tuple[Tensor, Tensor]:
        treated_excluding_dmso_24h_mask_t = not_dmso_6h_or_24h_mask_t[not_dmso_6h_mask_t]
        drug_and_treatment_non_zero_mask_using_batch_treated_t = treatment_non_zero_mask_using_batch_treated_t[treated_excluding_dmso_24h_mask_t] & drug_non_zero_mask_using_batch_treated_t
        drug_and_treatment_vectors_different_directions_loss = self._different_directions_loss(
            normalized_vectors_1=batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t[treated_excluding_dmso_24h_mask_t][drug_and_treatment_non_zero_mask_using_batch_treated_t],
            normalized_vectors_2=batch_normalized_anchor_dmso_24h_to_real_treated_t[drug_and_treatment_non_zero_mask_using_batch_treated_t],
            loss_pow_factor=self.different_directions_loss_power_factor
        )

        treatment_non_zero_mask_using_batch_treated = treatment_non_zero_mask_using_batch_treated_t.cpu().numpy()
        treated_non_zero_batch_perturbations = batch_perturbations[not_dmso_6h_mask][treatment_non_zero_mask_using_batch_treated]
        unique_treated_perturbations = np.unique(treated_non_zero_batch_perturbations)

        perturbation_to_normalized_anchor_treatment_vector = perturbation_to_anchor_treatment_vector.transform_tensor_with(lambda x: self.normalize_tensor(x))

        perturbation_to_all_other_normalized_treatment_anchors_t = {}
        perturbation_to_all_other_normalized_treatment_anchors_perturbations = {}
        for perturbation in unique_treated_perturbations:
            other_perturbations = [pert for pert in unique_treated_perturbations if perturbation != pert]
            perturbation_to_all_other_normalized_treatment_anchors_t[perturbation] = torch.stack([perturbation_to_normalized_anchor_treatment_vector[pert] for pert in other_perturbations])
            perturbation_to_all_other_normalized_treatment_anchors_perturbations[perturbation] = other_perturbations

        batch_non_zero_normalized_treatment_vectors_t = batch_normalized_dmso_6h_anchor_to_real_treated_vectors_t[treatment_non_zero_mask_using_batch_treated_t]
        non_zero_anchor_normalized_treatment_vectors_to_stack = []
        non_zero_anchor_normalized_treatment_vectors_perturbations_to_stack = []

        for perturbation in treated_non_zero_batch_perturbations:
            non_zero_anchor_normalized_treatment_vectors_to_stack.append(perturbation_to_all_other_normalized_treatment_anchors_t[perturbation])
            non_zero_anchor_normalized_treatment_vectors_perturbations_to_stack.append(perturbation_to_all_other_normalized_treatment_anchors_perturbations[perturbation])

        non_zero_batch_normalized_treatment_vectors_t = torch.repeat_interleave(
            batch_non_zero_normalized_treatment_vectors_t,
            repeats=len(unique_treated_perturbations) - 1,
            dim=0
        )
        non_zero_batch_normalized_treatment_vectors_perturbations = np.repeat(
            treated_non_zero_batch_perturbations,
            repeats=len(unique_treated_perturbations) - 1,
            axis=0
        )
        non_zero_anchor_normalized_treatment_vectors_t = torch.vstack(non_zero_anchor_normalized_treatment_vectors_to_stack)
        non_zero_anchor_normalized_treatment_vectors_perturbations = np.hstack(non_zero_anchor_normalized_treatment_vectors_perturbations_to_stack)
        treatment_vectors_different_directions_loss = self._different_directions_loss(
            normalized_vectors_1=non_zero_batch_normalized_treatment_vectors_t,
            normalized_vectors_2=non_zero_anchor_normalized_treatment_vectors_t,
            loss_pow_factor=self.different_directions_loss_power_factor,
            perturbations_equivalence_sets=perturbations_equivalence_sets,
            perturbations_equivalence_set_loss_coef=treatment_vectors_perturbations_equivalence_set_loss_coef,
            vectors_1_perturbations=non_zero_batch_normalized_treatment_vectors_perturbations,
            vectors_2_perturbations=non_zero_anchor_normalized_treatment_vectors_perturbations
        )

        return treatment_vectors_different_directions_loss, drug_and_treatment_vectors_different_directions_loss

    def calculate_center_for_each_cloud_in_batch(
            self,
            encoded_label_to_batch_samples_t: Dict[int, Tensor]
    ) -> Dict[int, Tensor]:
        return {
            encoded_label: find_center_of_cloud_t(cloud_samples_t)[0]
            for encoded_label, cloud_samples_t in encoded_label_to_batch_samples_t.items()
        }

    def distance_from_cloud_center_loss(
            self,
            z_t: Tensor,
            cloud_refs: NDArray[CmapCloudRef],
            cloud_ref_to_cloud_center: LookupTensor,
            z_std_t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        cloud_centers_latent = FastTensorBuilder(len(z_t))
        for cloud_ref in set(cloud_refs):
            cloud_centers_latent.add_slice(cloud_refs == cloud_ref, cloud_ref_to_cloud_center[cloud_ref])

        cloud_centers_latent_t = cloud_centers_latent.create_tensor()
        l2_distance_from_center_t = torch.linalg.norm(z_t - cloud_centers_latent_t, 2, dim=1)

        if z_std_t is not None:
            l2_distance_from_center_t = l2_distance_from_center_t / z_std_t

        return l2_distance_from_center_t.mean(), l2_distance_from_center_t

    def max_radius_limiter_loss(
            self,
            desired_max_cloud_radius: float,
            l2_distance_from_center_t: Tensor
    ) -> Tensor:
        return F.relu(l2_distance_from_center_t - desired_max_cloud_radius).mean()

    def calculate_tissue_to_mean_cloud_radius_t(
            self,
            tissues: NDArray[Tissue],
            l2_distance_from_center_t: Tensor
    ) -> Dict[Tissue, Tensor]:
        cloud_encoded_tissue_to_mean_cloud_radius_t = {}
        for tissue in np.unique(tissues):
            cloud_mask = tissues == tissue
            cloud_encoded_tissue_to_mean_cloud_radius_t[tissue] = l2_distance_from_center_t[cloud_mask].mean()
        return cloud_encoded_tissue_to_mean_cloud_radius_t

    def vectors_magnitude_regulator_loss(
            self,
            cloud_ref_to_vectors_t: Dict[CmapCloudRef, Tensor],
            tissue_to_dmso_6h_mean_cloud_radius_t: Dict[Tissue, Tensor],
            distance_coef: float
    ) -> Tensor:
        distances_tensors_before_stack = []
        for cloud_ref, vectors_t in cloud_ref_to_vectors_t.items():
            dmso_6h_mean_cloud_radius_t = tissue_to_dmso_6h_mean_cloud_radius_t[cloud_ref.tissue]
            distances_tensors_before_stack.append(F.relu(distance_coef * dmso_6h_mean_cloud_radius_t - torch.norm(vectors_t, dim=1)))
        return torch.vstack(distances_tensors_before_stack).mean()

    def calculate_cloud_ref_to_vectors_t(
            self,
            vectors: Tensor,
            cloud_refs: NDArray[CmapCloudRef]
    ) -> Dict[CmapCloudRef, Tensor]:
        cloud_ref_to_vectors = {}
        for cloud_ref in set(cloud_refs):
            cloud_mask = cloud_refs == cloud_ref
            cloud_ref_to_vectors[cloud_ref] = vectors[cloud_mask]
        return cloud_ref_to_vectors

    def _l2_treatment_and_drug_vectors_distance_loss(
            self,
            batch_tissue_untreated_6h_samples_t: Tensor,
            batch_tissue_untreated_24h_samples_t: Tensor,
            batch_treated_samples_t: Tensor,
            ref_drug_vector: Tensor,
            ref_treatment_vector: Tensor,
            z_std_t: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        p1_t, p2_t = predict_triangle_cloud(
            batch_tissue_untreated_6h_samples=batch_tissue_untreated_6h_samples_t,
            batch_tissue_untreated_24h_samples=batch_tissue_untreated_24h_samples_t,
            treatment_vector=ref_treatment_vector,
            drug_vector=ref_drug_vector
        )
        predicted_cloud_t = (p1_t + p2_t) / 2
        distance_p1_and_p2_to_treated = torch.norm(predicted_cloud_t - batch_treated_samples_t, dim=1).mean()
        distance_p1_and_p2 = torch.norm(p1_t - p2_t, dim=1).mean()

        if z_std_t is not None:
            distance_p1_and_p2_to_treated = distance_p1_and_p2_to_treated / z_std_t
            distance_p1_and_p2 = distance_p1_and_p2 / z_std_t

        return distance_p1_and_p2, distance_p1_and_p2_to_treated

    def _cdist_treatment_and_drug_vectors_distance_loss(
            self,
            batch_tissue_untreated_6h_samples_t: Tensor,
            batch_tissue_untreated_24h_samples_t: Tensor,
            batch_treated_samples_t: Tensor,
            ref_drug_vector: Tensor,
            ref_treatment_vector: Tensor,
            z_std_t: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Please re-check this method before usage")
        control_24_prod_6h = torch.cartesian_prod(
            torch.arange(0, len(batch_tissue_untreated_24h_samples_t), device=self.device, dtype=torch.long),
            torch.arange(0, len(batch_tissue_untreated_6h_samples_t), device=self.device, dtype=torch.long)
        )

        drug_vectors_line_x0_t = batch_tissue_untreated_24h_samples_t[control_24_prod_6h[:, 0]]
        drug_vectors_line_x_t = ref_drug_vector

        treatment_vectors_line_x0_t = batch_tissue_untreated_6h_samples_t[control_24_prod_6h[:, 1]]
        treatment_vectors_line_x_t = ref_treatment_vector

        p1, p2 = calculate_nearest_points_on_2_lines_t(
            l1=(drug_vectors_line_x0_t, drug_vectors_line_x_t),
            l2=(treatment_vectors_line_x0_t, treatment_vectors_line_x_t)
        )

        distance_p1_and_p2_to_treated = torch.cdist(torch.stack([p1, p2]), batch_treated_samples_t).mean()
        distance_p1_and_p2 = torch.cdist(p1, p2).mean()

        if z_std_t is not None:
            distance_p1_and_p2_to_treated = distance_p1_and_p2_to_treated / z_std_t
            distance_p1_and_p2 = distance_p1_and_p2 / z_std_t

        return distance_p1_and_p2, distance_p1_and_p2_to_treated

    def calculate_relevant_cloud_ref_to_within_radius_samples_t(
            self,
            cloud_ref_to_batch_samples_t: Dict[CmapCloudRef, Tensor],
            relevant_cloud_refs: Collection[CmapCloudRef],
            treated_max_radius_t: Tensor,
            cloud_ref_to_cloud_center: LookupTensor,
    ) -> Dict[CmapCloudRef, Tensor]:
        relevant_cloud_ref_to_within_radius_samples_t: Dict[CmapCloudRef, Tensor] = {}
        for cloud_ref in relevant_cloud_refs:
            current_cloud_samples_t = cloud_ref_to_batch_samples_t[cloud_ref]
            samples_radii_norm_t = self.calculate_cloud_sample_radii_norm_from_ref(
                cloud_samples_t=current_cloud_samples_t,
                cloud_center_ref_t=cloud_ref_to_cloud_center[cloud_ref]
            )
            relevant_cloud_ref_to_within_radius_samples_t[cloud_ref] = \
                current_cloud_samples_t[samples_radii_norm_t <= treated_max_radius_t]
        return relevant_cloud_ref_to_within_radius_samples_t


    def treatment_and_drug_vectors_distance_loss(
            self,
            cloud_ref_to_reference_treatment_vector: LookupTensor,
            cloud_ref_to_reference_drug_vector: LookupTensor,
            cloud_ref_to_samples_t: Dict[CmapCloudRef, Tensor],
            z_std_t: Tensor,
            dmso_6h_and_24h_cloud_ref_to_within_radius_samples_t: Dict[CmapCloudRef, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        distance_p1_and_p2_to_treated_dist_list = []
        distance_p1_and_p2_dist_list = []

        # drug reference vectors cloud ref should contain all that mutual cloud refs.
        for cloud_ref in cloud_ref_to_reference_drug_vector.keys():
            if cloud_ref.dmso_24h not in dmso_6h_and_24h_cloud_ref_to_within_radius_samples_t or \
                    cloud_ref.dmso_6h not in dmso_6h_and_24h_cloud_ref_to_within_radius_samples_t:
                self.logger.warning(f'No samples for 6h/24h DMSO for cloud {cloud_ref}.')
                continue
            current_tissue_dmso_24h_samples_t = dmso_6h_and_24h_cloud_ref_to_within_radius_samples_t[cloud_ref.dmso_24h]
            current_tissue_dmso_6h_samples_t = dmso_6h_and_24h_cloud_ref_to_within_radius_samples_t[cloud_ref.dmso_6h]
            if len(current_tissue_dmso_6h_samples_t) == 0 or len(current_tissue_dmso_24h_samples_t) == 0:
                self.logger.warning(f'No samples for 6h/24h DMSO for cloud {cloud_ref}.')
                continue
            treated_samples_t = cloud_ref_to_samples_t[cloud_ref]

            loss_subroutine = self._cdist_treatment_and_drug_vectors_distance_loss \
                if self.treatment_and_drug_vectors_distance_loss_cdist_usage \
                else self._l2_treatment_and_drug_vectors_distance_loss

            distance_p1_and_p2, distance_p1_and_p2_to_treated = loss_subroutine(
                batch_tissue_untreated_6h_samples_t=current_tissue_dmso_6h_samples_t,
                batch_tissue_untreated_24h_samples_t=current_tissue_dmso_24h_samples_t,
                batch_treated_samples_t=treated_samples_t,
                ref_drug_vector=cloud_ref_to_reference_drug_vector[cloud_ref],
                ref_treatment_vector=cloud_ref_to_reference_treatment_vector[cloud_ref],
                z_std_t=z_std_t
            )

            distance_p1_and_p2_to_treated_dist_list.append(distance_p1_and_p2_to_treated)
            distance_p1_and_p2_dist_list.append(distance_p1_and_p2)

        assert len(distance_p1_and_p2_to_treated_dist_list) == len(distance_p1_and_p2_dist_list)

        if len(distance_p1_and_p2_to_treated_dist_list) == 0:
            return self._0_loss, self._0_loss

        return torch.stack(distance_p1_and_p2_dist_list).mean(), \
               torch.stack(distance_p1_and_p2_to_treated_dist_list).mean()

    def get_embedding(self, samples: Tensor) -> SamplesEmbedding:
        z, mu, log_var = self.vae(samples)
        return SamplesEmbedding(
            z_t=z,
            mu_t=mu,
            log_var_t=log_var
        )

