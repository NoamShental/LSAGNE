from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, BoolTensor, LongTensor
from torch.types import Device

TensorOrNumpyType = TypeVar('TensorOrNumpyType', Tensor, np.ndarray)


def convert_array_to_tensor(array: ArrayLike, device: Device) -> Tensor:
    # TODO add more types
    array: NDArray = np.array(array)
    if np.issubdtype(array.dtype, np.integer):
        tensor = LongTensor(array)
    elif np.issubdtype(array.dtype, np.bool8):
        tensor = BoolTensor(array)
    else:
        tensor = Tensor(array)
    if device:
        tensor = tensor.to(device)
    return tensor