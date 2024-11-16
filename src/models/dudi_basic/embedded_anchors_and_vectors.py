from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Set

from torch import Tensor

from src.assertion_utils import assert_promise
from src.cmap_cloud_ref import CmapCloudRef
from src.models.dudi_basic.lookup_tensor import LookupTensor
from src.models.dudi_basic.multi_device_data import AnchorPointsLookup
from src.perturbation import Perturbation
from src.samples_embedder import SamplesEmbedder


@dataclass(frozen=True)
class EmbeddedAnchorsAndVectors(AnchorPointsLookup):
    perturbation_to_anchor_treatment_vector: LookupTensor[Perturbation]
    perturbation_to_anchor_drug_vector: LookupTensor[Perturbation]
    cloud_ref_to_reference_treatment_vector: LookupTensor[CmapCloudRef]
    cloud_ref_to_reference_drug_vector: LookupTensor[CmapCloudRef]

    def _make_assertions(self):
        assert set(self.perturbation_to_anchor_treatment_vector.keys()) == set([k.perturbation for k in self.cloud_ref_to_reference_treatment_vector.keys()])
        assert set(self.perturbation_to_anchor_drug_vector.keys()).difference({Perturbation.TIME_24H()}) == {k.perturbation for k in self.cloud_ref_to_reference_drug_vector.keys()}
        # drug vector should contain all the treated clouds except for <dmso_24, some tissue>
        assert set(self.cloud_ref_to_reference_drug_vector.keys()).issubset(self.cloud_ref_to_reference_treatment_vector.keys())

    def __post_init__(self):
        assert_promise(lambda: self._make_assertions())

    @staticmethod
    def _calculate_perturbation_to_anchor_vector(
            perturbation_to_embedded_treated_anchor: LookupTensor[Perturbation],
            perturbation_to_embedded_control_anchor: LookupTensor[Perturbation],
            ) -> LookupTensor[Perturbation]:
        # if it is equal, we can use simple operator between the stacked tensors
        assert perturbation_to_embedded_treated_anchor.lookup == perturbation_to_embedded_control_anchor.lookup
        return LookupTensor(
            lookup=perturbation_to_embedded_treated_anchor.lookup,
            stacked_tensor=perturbation_to_embedded_treated_anchor.stacked_tensor - perturbation_to_embedded_control_anchor.stacked_tensor
        )

    @staticmethod
    def _calculate_cloud_ref_to_reference_vector(
            cloud_ref_to_embedded_anchor: LookupTensor[CmapCloudRef],
            control_perturbation: Literal['dmso_6h', 'dmso_24h']
    ) -> LookupTensor[CmapCloudRef]:
        if control_perturbation == 'dmso_6h':
            is_control_dmso_6h = True
        elif control_perturbation == 'dmso_24h':
            is_control_dmso_6h = False
        else:
            raise AssertionError(f'control_perturbation cannot be "{control_perturbation}".')

        if is_control_dmso_6h:
            treated_cloud_refs: Set[CmapCloudRef] = {cloud_ref for cloud_ref in cloud_ref_to_embedded_anchor.keys() if not cloud_ref.is_dmso_6h}
        else:
            treated_cloud_refs: Set[CmapCloudRef] = {cloud_ref for cloud_ref in cloud_ref_to_embedded_anchor.keys() if cloud_ref.is_not_dmso_6h_or_24h}
        cloud_ref_to_reference_vector: Dict[CmapCloudRef, Tensor] = {}
        for cloud_ref in treated_cloud_refs:
            treated_anchor = cloud_ref_to_embedded_anchor[cloud_ref]
            if is_control_dmso_6h:
                control = cloud_ref_to_embedded_anchor[cloud_ref.dmso_6h]
            else:
                control = cloud_ref_to_embedded_anchor[cloud_ref.dmso_24h]
            cloud_ref_to_reference_vector[cloud_ref] = treated_anchor - control
        return LookupTensor.create_from_dict(cloud_ref_to_reference_vector)

    @classmethod
    def create(cls, original_space_anchor_points_lookup: AnchorPointsLookup, embedder: SamplesEmbedder) -> EmbeddedAnchorsAndVectors:
        mu_embedder = lambda x: embedder.get_embedding(x).mu_t
        latent_space_anchor_points_lookup = AnchorPointsLookup(
            perturbation_to_anchor_dmso_6h=original_space_anchor_points_lookup.perturbation_to_anchor_dmso_6h.transform_tensor_with(
                mu_embedder),
            perturbation_to_anchor_dmso_24h=original_space_anchor_points_lookup.perturbation_to_anchor_dmso_24h.transform_tensor_with(
                mu_embedder),
            perturbation_to_anchor_treated=original_space_anchor_points_lookup.perturbation_to_anchor_treated.transform_tensor_with(
                mu_embedder),
            tissue_to_anchor_dmso_6h=original_space_anchor_points_lookup.tissue_to_anchor_dmso_6h.transform_tensor_with(
                mu_embedder),
            tissue_to_anchor_dmso_24h=original_space_anchor_points_lookup.tissue_to_anchor_dmso_24h.transform_tensor_with(
                mu_embedder),
            cloud_ref_to_cloud_center=original_space_anchor_points_lookup.cloud_ref_to_cloud_center.transform_tensor_with(
                mu_embedder)
        )
        return EmbeddedAnchorsAndVectors(
            perturbation_to_anchor_dmso_6h=latent_space_anchor_points_lookup.perturbation_to_anchor_dmso_6h,
            perturbation_to_anchor_dmso_24h=latent_space_anchor_points_lookup.perturbation_to_anchor_dmso_24h,
            perturbation_to_anchor_treated=latent_space_anchor_points_lookup.perturbation_to_anchor_treated,
            tissue_to_anchor_dmso_6h=latent_space_anchor_points_lookup.tissue_to_anchor_dmso_6h,
            tissue_to_anchor_dmso_24h=latent_space_anchor_points_lookup.tissue_to_anchor_dmso_24h,
            cloud_ref_to_cloud_center=latent_space_anchor_points_lookup.cloud_ref_to_cloud_center,
            perturbation_to_anchor_treatment_vector=cls._calculate_perturbation_to_anchor_vector(
                perturbation_to_embedded_treated_anchor=latent_space_anchor_points_lookup.perturbation_to_anchor_treated,
                perturbation_to_embedded_control_anchor=latent_space_anchor_points_lookup.perturbation_to_anchor_dmso_6h
            ),
            perturbation_to_anchor_drug_vector=cls._calculate_perturbation_to_anchor_vector(
                perturbation_to_embedded_treated_anchor=latent_space_anchor_points_lookup.perturbation_to_anchor_treated,
                perturbation_to_embedded_control_anchor=latent_space_anchor_points_lookup.perturbation_to_anchor_dmso_24h
            ).remove_key(Perturbation.TIME_24H()),
            cloud_ref_to_reference_treatment_vector=cls._calculate_cloud_ref_to_reference_vector(
                cloud_ref_to_embedded_anchor=latent_space_anchor_points_lookup.cloud_ref_to_cloud_center,
                control_perturbation='dmso_6h'
            ),
            cloud_ref_to_reference_drug_vector=cls._calculate_cloud_ref_to_reference_vector(
                cloud_ref_to_embedded_anchor=latent_space_anchor_points_lookup.cloud_ref_to_cloud_center,
                control_perturbation='dmso_24h'
            )
        )
