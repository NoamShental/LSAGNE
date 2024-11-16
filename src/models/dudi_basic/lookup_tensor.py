from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, TypeVar, Generic, Union, Mapping, Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.types import Device

from src.assertion_utils import assert_same_len
from src.conversions_utils import convert_array_to_tensor

T = TypeVar('T')


@dataclass(frozen=True)
class LookupTensor(Mapping[T, Tensor], Generic[T]):
    lookup: Dict[T, int]
    stacked_tensor: Tensor

    def __post_init__(self):
        assert_same_len(self.lookup, self.stacked_tensor)

    def __iter__(self) -> Iterator[T]:
        yield from self.lookup.keys()

    def __getitem__(self, item: T) -> Tensor:
        return self.stacked_tensor[self.lookup[item]]

    def __len__(self):
        return len(self.stacked_tensor)

    def transform_keys_with(self, old_to_new_key: Dict[T, T]) -> LookupTensor:
        return LookupTensor({
            old_to_new_key[k]: v for k, v in self.lookup.items()
        }, self.stacked_tensor)

    def transform_tensor_with(self, tensor_transformer: Callable[[Tensor], Tensor]) -> LookupTensor:
        return LookupTensor(self.lookup, tensor_transformer(self.stacked_tensor))

    @classmethod
    def create_from_dict(cls, k_to_t: Dict[T, Union[Tensor, NDArray[float]]], device: Device = None) -> LookupTensor:
        if len(k_to_t) == 0:
            raise ValueError("No empty lookup is supported.")
        tensor_list = []
        lookup = {}
        for i, (k, t) in enumerate(k_to_t.items()):
            tensor_list.append(t)
            lookup[k] = i
        if isinstance(tensor_list[0], Tensor):
            stacked_tensor = torch.stack(tensor_list)
            if stacked_tensor.device != device:
                stacked_tensor = stacked_tensor.to(device)
        else:
            stacked_tensor = convert_array_to_tensor(np.stack(tensor_list), device)

        return cls(
            lookup=lookup,
            stacked_tensor=stacked_tensor
        )

    def remove_key(self, key: T) -> LookupTensor[T]:
        if key not in self.lookup:
            raise KeyError(f"Key {key} not found in lookup")
        key_i = self.lookup[key]
        lookup_without_key = {}
        for k, i in self.lookup.items():
            if k == key:
                continue
            if i > key_i:
                lookup_without_key[k] = i - 1
            else:
                lookup_without_key[k] = i
        mask_without_key = np.full(len(self.stacked_tensor), True)
        mask_without_key[key_i] = False
        return LookupTensor(
            lookup=lookup_without_key,
            stacked_tensor=self.stacked_tensor[mask_without_key]
        )
