from __future__ import annotations
from dataclasses import dataclass

from numpy.typing import NDArray
from torch import Tensor

# TODO - delete file


@dataclass
class MultiDeviceVector:
    vector_z_t: Tensor
    vector_z: NDArray[float]

    @classmethod
    def create_using_torch(cls, tensor: Tensor) -> MultiDeviceVector:
        return MultiDeviceVector(
            vector_z_t=tensor,
            vector_z=tensor.cpu().numpy()
        )
