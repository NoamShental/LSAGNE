from dataclasses import dataclass
from typing import Dict

from numpy.typing import NDArray

from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_dataset_splitter import SplittedCmapDataset
from src.models.dudi_basic.anchor_points import AnchorPoints
from src.models.dudi_basic.lsagne_model import LsagneModel
from src.models.dudi_basic.model_learning_parameters import ModelLearningParameters


@dataclass
class TrainingSummary:
    final_total_loss: float
    params: ModelLearningParameters
    # total_epochs: int
    best_model_epoch: int
    model_state_dict: Dict
    model: LsagneModel
    optimizer_state_dict: Dict
    lr_history: list
    epochs_losses_history: Dict
    left_out_cloud: CmapCloudRef
    anchor_points: AnchorPoints[NDArray[float]]
    cmap_datasets: SplittedCmapDataset
    n_epochs: int
