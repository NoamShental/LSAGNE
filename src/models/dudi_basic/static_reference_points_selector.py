from typing import Dict

from numpy.typing import NDArray

from src.cmap_cloud_ref import CmapCloudRef
from src.models.dudi_basic.anchor_points import AnchorPoints
from src.models.dudi_basic.perturbations_anchors import PerturbationsAnchors
from src.models.dudi_basic.reference_points_selector import AnchorPointsSelector
from src.perturbation import Perturbation
from src.tissue import Tissue


class StaticAnchorPointsSelector(AnchorPointsSelector):

    def select_points_impl(self, *args, **kwargs) -> AnchorPoints[NDArray]:
        cloud_ref_to_anchor: Dict[CmapCloudRef, NDArray[float]] = {}
        tissue_to_dmso_6h_sample: Dict[Tissue, NDArray[float]] = {}
        perturbation_to_anchors: Dict[Perturbation, PerturbationsAnchors[NDArray[float]]] = {}
        perturbation_to_largest_treated_cloud: Dict[Perturbation, CmapCloudRef] = self.calculate_perturbation_to_largest_treated_cloud()

        self.logger.info(f"Selecting anchor samples using {self.__class__.__name__}...")

        # 1. Create a representative point for each cloud + control ref for each tissue.
        for cloud_ref, samples in self.cmap_dataset.cloud_ref_to_samples.items():
            # FIXME make it random
            sample = samples[0]
            cloud_ref_to_anchor[cloud_ref] = sample
            if cloud_ref.is_dmso_6h:
                tissue_to_dmso_6h_sample[cloud_ref.tissue] = sample

        # 2. Create reference points for each perturbation.
        for p in self.cmap_dataset.perturbations_unique:
            if p.is_dmso_6h:
                continue
            # choose by taking the largest cloud for this perturbation
            largest_cloud_ref = perturbation_to_largest_treated_cloud[p]
            perturbation_to_anchors[p] = PerturbationsAnchors(
                treated=cloud_ref_to_anchor[largest_cloud_ref],
                dmso_6=cloud_ref_to_anchor[largest_cloud_ref.dmso_6h],
                dmso_24=cloud_ref_to_anchor[largest_cloud_ref.dmso_24h]
            )

        return AnchorPoints(
            perturbation_to_anchors=perturbation_to_anchors,
            cloud_ref_to_cloud_center=cloud_ref_to_anchor
        )
