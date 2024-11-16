from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger
from operator import itemgetter
from typing import Dict

from numpy.typing import NDArray

from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.anchor_points import AnchorPoints
from src.perturbation import Perturbation


@dataclass
class AnchorPointsSelector(ABC):
    cmap_dataset: RawCmapDataset
    enable_drug_vector: bool
    logger: Logger

    def _log_selections(self, selected_anchors: AnchorPoints[NDArray]):
        self.logger.info("Anchors for cloud_ref --> cloud center")
        self.logger.info('=' * 50)
        for cloud_ref, sample in selected_anchors.cloud_ref_to_cloud_center.items():
            self.logger.info(f"{cloud_ref} --> {self.cmap_dataset.identify_sample_idx_and_id(sample)}")

        self.logger.info("Anchors for perturbation --> anchors")
        self.logger.info('=' * 50)
        for perturbation, anchors in selected_anchors.perturbation_to_anchors.items():
            self.logger.info(f"{perturbation} treated --> {self.cmap_dataset.identify_sample_idx_and_id(anchors.treated)}")
            self.logger.info(f"{perturbation} dmso_6 --> {self.cmap_dataset.identify_sample_idx_and_id(anchors.dmso_6)}")
            self.logger.info(f"{perturbation} dmso_24 --> {self.cmap_dataset.identify_sample_idx_and_id(anchors.dmso_24)}")

    def select_points(self, *args, **kwargs) -> AnchorPoints[NDArray]:
        self.logger.info(f"Selecting anchors using {self.__class__.__name__}")
        selected_anchors = self.select_points_impl(*args, **kwargs)
        self._log_selections(selected_anchors)
        return selected_anchors

    def calculate_perturbation_to_largest_treated_cloud(self) -> Dict[Perturbation, CmapCloudRef]:
        perturbation_to_largest_treated_cloud = {}
        for cloud_ref, samples in self.cmap_dataset.cloud_ref_to_samples.items():
            if cloud_ref.is_dmso_6h:
                continue
            if cloud_ref.perturbation not in perturbation_to_largest_treated_cloud:
                largest_cloud_ref = cloud_ref
            else:
                old_cloud_ref = perturbation_to_largest_treated_cloud[cloud_ref.perturbation]
                old_size = len(self.cmap_dataset.cloud_ref_to_samples[old_cloud_ref])
                new_size = len(self.cmap_dataset.cloud_ref_to_samples[cloud_ref])
                largest_cloud_ref = max((old_size, old_cloud_ref), (new_size, cloud_ref), key=itemgetter(0))[1]
            perturbation_to_largest_treated_cloud[cloud_ref.perturbation] = largest_cloud_ref
        return perturbation_to_largest_treated_cloud

    @abstractmethod
    def select_points_impl(self, *args, **kwargs) -> AnchorPoints[NDArray]:
        ...
