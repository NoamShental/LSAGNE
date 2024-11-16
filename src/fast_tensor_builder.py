from typing import Union

import numpy as np
import torch
from numpy import ndarray
from numpy.typing import ArrayLike
from torch import Tensor


class FastTensorBuilder:
    def __init__(self, size: int):
        self._np_array: ndarray = np.empty(size, dtype=object)

    def add_slice(self, idx_or_mask: Union[ArrayLike, ndarray], tensor: Tensor):
        if isinstance(idx_or_mask, list):
            idx_or_mask = np.array(idx_or_mask)
        if np.issubdtype(idx_or_mask.dtype, np.integer):
            tensor_len = len(idx_or_mask)
        else:
            # must be boolean
            tensor_len = idx_or_mask.sum()
        if tensor_len == 0:
            return
        tt = np.empty(tensor_len, dtype=object)
        tt.fill(tensor)
        self._np_array[idx_or_mask] = tt

    def create_tensor(self) -> Tensor:
        return torch.stack(tuple(self._np_array))