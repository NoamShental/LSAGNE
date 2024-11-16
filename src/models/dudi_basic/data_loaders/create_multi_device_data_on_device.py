from typing import Dict, Any

import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor
from torch.types import Device

from src.models.dudi_basic.lookup_tensor import LookupTensor

import numpy as np

from src.models.dudi_basic.multi_device_data import MultiDeviceData, OnDeviceData


# TODO - this file is to be removed


def create_multi_device_data_on_device(on_cpu_multi_device_data: MultiDeviceData, device: Device):
    raise NotImplementedError("Not to be used!")
    key_to_lookup_tensor = {key: _convert_to_lookup_tensor(d, device) for key, d in on_cpu_multi_device_data.on_device_lookup.items()}
    key_to_tensor_dict_on_cpu = _convert_to_key_to_numpy_dict(on_cpu_multi_device_data.on_cpu)
    multi_device_data_t = MultiDeviceData[Tensor](
        on_device=OnDeviceData[Tensor](
                raw_samples=_convert_np_array_to_tensor(on_cpu_multi_device_data.on_device.raw_samples, device),
                encoded_numeric_labels=_convert_np_array_to_tensor(on_cpu_multi_device_data.on_device.encoded_labels, device),
                encoded_tissues=_convert_np_array_to_tensor(on_cpu_multi_device_data.on_device.encoded_tissues, device),
                encoded_perturbations=_convert_np_array_to_tensor(on_cpu_multi_device_data.on_device.encoded_perturbations, device),
                is_untreated=_convert_np_array_to_tensor(on_cpu_multi_device_data.on_device.is_dmso_6h_mask, device),
                class_weights=_convert_np_array_to_tensor(on_cpu_multi_device_data.on_device.class_weights, device)
            ),
        on_device_lookup=key_to_lookup_tensor,
        on_cpu=key_to_tensor_dict_on_cpu
    )
    return multi_device_data_t


# FIXME: move all this code into the Lookup tensor class
def _convert_to_lookup_tensor(key_to_arraylike_dict: Dict[Any, ArrayLike],
                              device: Device) -> LookupTensor:
    raise NotImplementedError("Not to be used!")
    array_t = [_convert_np_array_to_tensor(array, device) for array in key_to_arraylike_dict.values()]
    stacked_t = torch.stack(array_t)
    lookup = {key: stack_i for key, stack_i in zip(key_to_arraylike_dict.keys(), range(len(key_to_arraylike_dict)))}
    return LookupTensor(lookup, stacked_t)


def _convert_to_key_to_tensor_dict(key_to_arraylike_dict: Dict[Any, ArrayLike], device: Device) -> Dict[Any, torch.Tensor]:
    raise NotImplementedError("Not to be used!")
    return {key: _convert_np_array_to_tensor(array, device) for key, array in key_to_arraylike_dict.items()}


def _convert_to_key_to_numpy_dict(key_to_arraylike_dict: Dict[Any, ArrayLike]) -> Dict[Any, NDArray]:
    raise NotImplementedError("Not to be used!")
    return {key: np.array(array) for key, array in key_to_arraylike_dict.items()}


def _convert_np_array_to_tensor(array: ArrayLike, device: Device) -> Tensor:
    raise NotImplementedError("Not to be used!")
    # TODO add more types
    array: NDArray = np.array(array)
    if np.issubdtype(array.dtype, np.integer):
        tensor = torch.LongTensor(array)
    elif np.issubdtype(array.dtype, np.bool8):
        tensor = torch.BoolTensor(array)
    else:
        tensor = torch.Tensor(array)
    tensor = tensor.to(device)
    return tensor