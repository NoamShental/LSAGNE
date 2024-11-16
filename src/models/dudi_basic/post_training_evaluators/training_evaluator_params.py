from dataclasses import dataclass
from typing import Any, Dict, Collection

from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.drawer import Drawer
from src.models.dudi_basic.lsagne_model import LsagneModel
from src.random_manager import RandomManager
from src.training_summary import TrainingSummary


@dataclass
class TrainingEvaluatorParams:
    working_directory: str
    left_out_cloud_ref: CmapCloudRef
    cross_validation_cloud_refs: Collection[CmapCloudRef]
    drawer: Drawer
    original_full_cmap_dataset: RawCmapDataset
    updated_full_cmap_dataset: RawCmapDataset
    original_training_cmap_dataset: RawCmapDataset
    updated_training_cmap_dataset: RawCmapDataset
    left_out_cmap_dataset: RawCmapDataset
    cross_validation_cloud_ref_to_cmap_dataset: Dict[CmapCloudRef, RawCmapDataset]
    training_summary: TrainingSummary
    model: LsagneModel
    device: Any
    random_manager: RandomManager
    embedding_dim: int
    enable_triangle: bool
    predicted_cloud_max_size: int
