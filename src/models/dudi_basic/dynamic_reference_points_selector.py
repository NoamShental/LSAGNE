from dataclasses import dataclass
from typing import Dict

import torch
from numpy.typing import NDArray
from torch import Tensor

from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_clouds_utils import find_idx_for_center_of_cloud_t
from src.models.dudi_basic.anchor_points import AnchorPoints
from src.models.dudi_basic.perturbations_anchors import PerturbationsAnchors
from src.models.dudi_basic.reference_points_selector import AnchorPointsSelector
from src.perturbation import Perturbation
from src.tissue import Tissue


@dataclass
class DynamicAnchorPointsSelector(AnchorPointsSelector):

    @torch.no_grad()
    def select_points_impl(self, z_t: Tensor) -> AnchorPoints:
        self.logger.info('Performing dynamic reference point re-selection...')
        original_samples = self.cmap_dataset.data
        cloud_ref_to_anchor: Dict[CmapCloudRef, NDArray[float]] = {}
        perturbation_to_anchors: Dict[Perturbation, PerturbationsAnchors[NDArray[float]]] = {}
        perturbation_to_largest_treated_cloud: Dict[Perturbation, CmapCloudRef] = self.calculate_perturbation_to_largest_treated_cloud()

        # select the center point for each cloud
        for cloud_ref, idx in self.cmap_dataset.cloud_ref_to_idx.items():
            cloud_z_t = z_t[idx]
            cloud_center_idx = int(find_idx_for_center_of_cloud_t(cloud_z_t).cpu())
            cloud_ref_to_anchor[cloud_ref] = original_samples[idx][cloud_center_idx]

        for perturbation in self.cmap_dataset.perturbations_unique:
            if perturbation.is_dmso_6h:
                continue
            largest_treated_cloud_ref = perturbation_to_largest_treated_cloud[perturbation]
            perturbation_to_anchors[perturbation] = PerturbationsAnchors(
                treated=cloud_ref_to_anchor[largest_treated_cloud_ref],
                dmso_6=cloud_ref_to_anchor[largest_treated_cloud_ref.dmso_6h],
                dmso_24=cloud_ref_to_anchor[largest_treated_cloud_ref.dmso_24h]
            )

        return AnchorPoints(
            perturbation_to_anchors=perturbation_to_anchors,
            cloud_ref_to_cloud_center=cloud_ref_to_anchor
        )
