from dataclasses import dataclass
from typing import Optional

from numpy.typing import NDArray
from torch import Tensor

@dataclass
class TrainingTensors:
    training_original_space_t: Tensor
    training_z_t: Tensor
    training_mu_t: Optional[Tensor]
    training_log_var_t: Optional[Tensor]

    training_original_space: NDArray[float]
    training_z: NDArray[float]
    training_mu: Optional[NDArray[float]]
    training_log_var: Optional[NDArray[float]]


@dataclass
class LeftOutTensors:
    true_left_out_original_space_t: Tensor
    true_left_out_z_t: Tensor
    true_left_out_mu_t: Optional[Tensor]
    true_left_out_log_var_t: Optional[Tensor]

    true_left_out_original_space: NDArray[float]
    true_left_out_z: NDArray[float]
    true_left_out_mu: Optional[NDArray[float]]
    true_left_out_log_var: Optional[NDArray[float]]

@dataclass
class TrainingAndLeftOutTensors(TrainingTensors, LeftOutTensors):
    ...
