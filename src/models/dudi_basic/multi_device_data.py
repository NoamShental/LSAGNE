from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Any, Union

import numpy as np
from numpy.typing import NDArray
from torch import Tensor, LongTensor, BoolTensor
from torch.types import Device

from src.assertion_utils import assert_same_len, assert_promise_true
from src.cmap_cloud_ref import CmapCloudRef
from src.conversions_utils import convert_array_to_tensor
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.anchor_points import AnchorPoints
from src.models.dudi_basic.lookup_tensor import LookupTensor
from src.perturbation import Perturbation
from src.tissue import Tissue


@dataclass(frozen=True)
class OnDeviceData:
    raw_samples: Tensor
    encoded_labels: Tensor
    encoded_tissues: Tensor
    encoded_perturbations: Tensor
    is_dmso_6h_mask: Tensor
    is_dmso_24h_mask: Tensor
    not_dmso_6h_or_24h_mask: Tensor
    not_dmso_6h_mask: Tensor
    class_weights: Tensor

    def create_batch(self, idx: NDArray[Union[int, bool]]) -> OnDeviceData:
        return OnDeviceData(
            raw_samples=self.raw_samples[idx],
            encoded_labels=self.encoded_labels[idx],
            encoded_tissues=self.encoded_tissues[idx],
            encoded_perturbations=self.encoded_perturbations[idx],
            is_dmso_6h_mask=self.is_dmso_6h_mask[idx],
            is_dmso_24h_mask=self.is_dmso_24h_mask[idx],
            not_dmso_6h_or_24h_mask=self.not_dmso_6h_or_24h_mask[idx],
            not_dmso_6h_mask=self.not_dmso_6h_mask[idx],
            class_weights=self.class_weights[idx]
        )


@dataclass(frozen=True)
class OnCpuData:
    raw_samples: NDArray[float]
    encoded_labels: NDArray[int]
    encoded_tissues: NDArray[int]
    encoded_perturbations: NDArray[int]
    is_dmso_6h_mask: NDArray[bool]
    is_dmso_24h_mask: NDArray[bool]
    not_dmso_6h_or_24h_mask: NDArray[bool]
    not_dmso_6h_mask: NDArray[bool]
    cloud_refs: NDArray[CmapCloudRef]
    perturbations: NDArray[Perturbation]
    tissues: NDArray[Tissue]

    def create_batch(self, idx: NDArray[Union[int, bool]]) -> OnCpuData:
        return OnCpuData(
            raw_samples=self.raw_samples[idx],
            encoded_labels=self.encoded_labels[idx],
            encoded_tissues=self.encoded_tissues[idx],
            encoded_perturbations=self.encoded_perturbations[idx],
            is_dmso_6h_mask=self.is_dmso_6h_mask[idx],
            is_dmso_24h_mask=self.is_dmso_24h_mask[idx],
            not_dmso_6h_or_24h_mask=self.not_dmso_6h_or_24h_mask[idx],
            not_dmso_6h_mask=self.not_dmso_6h_mask[idx],
            cloud_refs=self.cloud_refs[idx],
            perturbations=self.perturbations[idx],
            tissues=self.tissues[idx]
        )


@dataclass(frozen=True)
class AnchorPointsLookup:
    perturbation_to_anchor_dmso_6h: LookupTensor[Perturbation]
    perturbation_to_anchor_dmso_24h: LookupTensor[Perturbation]
    perturbation_to_anchor_treated: LookupTensor[Perturbation]
    tissue_to_anchor_dmso_6h: LookupTensor[Tissue]
    tissue_to_anchor_dmso_24h: LookupTensor[Tissue]
    cloud_ref_to_cloud_center: LookupTensor[CmapCloudRef]

    @classmethod
    def create_from_anchor_points(cls, anchor_points: AnchorPoints[NDArray[float]], device: Device) -> AnchorPointsLookup:
        tissue_to_anchor_dmso_6h = {
            cloud_ref.tissue: cloud_center
            for cloud_ref, cloud_center in anchor_points.cloud_ref_to_cloud_center.items()
            if cloud_ref.is_dmso_6h
        }
        tissue_to_anchor_dmso_24h = {
            cloud_ref.tissue: cloud_center
            for cloud_ref, cloud_center in anchor_points.cloud_ref_to_cloud_center.items()
            if cloud_ref.is_dmso_24h
        }
        return cls(
            perturbation_to_anchor_dmso_6h=LookupTensor.create_from_dict({p: anchors.dmso_6 for p, anchors in anchor_points.perturbation_to_anchors.items()}, device),
            perturbation_to_anchor_dmso_24h=LookupTensor.create_from_dict({p: anchors.dmso_24 for p, anchors in anchor_points.perturbation_to_anchors.items()}, device),
            perturbation_to_anchor_treated=LookupTensor.create_from_dict({p: anchors.treated for p, anchors in anchor_points.perturbation_to_anchors.items()}, device),
            tissue_to_anchor_dmso_6h=LookupTensor.create_from_dict(tissue_to_anchor_dmso_6h, device),
            tissue_to_anchor_dmso_24h=LookupTensor.create_from_dict(tissue_to_anchor_dmso_24h, device),
            cloud_ref_to_cloud_center=LookupTensor.create_from_dict(anchor_points.cloud_ref_to_cloud_center, device)
        )


@dataclass(frozen=True)
class MultiDeviceData:
    on_device: OnDeviceData
    on_device_original_space_anchor_points: AnchorPointsLookup
    on_cpu: OnCpuData

    @cached_property
    def device(self) -> Device:
        return self.on_device.raw_samples.device

    def __post_init__(self):
        assert_promise_true(lambda: self._all_dict_values_same_length(dataclasses.asdict(self.on_device)), "Not all iters in 'on_device' are of the same length")
        assert_promise_true(lambda: self._all_dict_values_same_length(dataclasses.asdict(self.on_cpu)), "Not all iters in 'on_cpu' are of the same length")
        assert_promise_true(lambda: np.all(self.on_cpu.raw_samples == self.on_device.raw_samples.cpu().numpy()))
        assert_promise_true(lambda: np.all(self.on_cpu.encoded_labels == self.on_device.encoded_labels.cpu().numpy()))
        assert_promise_true(lambda: np.all(self.on_cpu.encoded_tissues == self.on_device.encoded_tissues.cpu().numpy()))

    @staticmethod
    def _all_dict_values_same_length(d: Dict[Any, Any]):
        lengths = [len(val) for val in d.values()]
        return all(x == lengths[0] for x in lengths)

    def __len__(self):
        return len(self.on_cpu.raw_samples)

    def create_batch(self, idx: NDArray[Union[int, bool]]) -> MultiDeviceData:
        return MultiDeviceData(
            on_device_original_space_anchor_points=self.on_device_original_space_anchor_points,
            on_device=self.on_device.create_batch(idx),
            on_cpu=self.on_cpu.create_batch(idx)
        )

    @classmethod
    def create(cls, cmap: RawCmapDataset, anchor_points: AnchorPoints[NDArray[float]], device: Device):
        is_dmso_6h_t = convert_array_to_tensor(cmap.is_dmso_6h_mask, device)
        return cls(
            on_device=OnDeviceData(
                raw_samples=convert_array_to_tensor(cmap.data, device),
                encoded_labels=convert_array_to_tensor(cmap.encoded_labels, device),
                encoded_tissues=convert_array_to_tensor(cmap.encoded_tissues, device),
                encoded_perturbations=convert_array_to_tensor(cmap.encoded_perturbations, device),
                is_dmso_6h_mask=is_dmso_6h_t,
                is_dmso_24h_mask=convert_array_to_tensor(cmap.is_dmso_24h_mask, device),
                not_dmso_6h_or_24h_mask=convert_array_to_tensor(cmap.not_dmso_6h_or_24h_mask, device),
                not_dmso_6h_mask=~is_dmso_6h_t,
                class_weights=convert_array_to_tensor(
                    np.array([cmap.encoded_labels_to_balanced_class_weights[label]
                              for label in cmap.encoded_labels]),
                    device
                )
            ),
            on_device_original_space_anchor_points=AnchorPointsLookup.create_from_anchor_points(anchor_points, device),
            on_cpu=OnCpuData(
                raw_samples=cmap.data,
                encoded_labels=cmap.encoded_labels,
                encoded_tissues=cmap.encoded_tissues,
                encoded_perturbations=cmap.encoded_perturbations,
                is_dmso_6h_mask=cmap.is_dmso_6h_mask,
                is_dmso_24h_mask=cmap.is_dmso_24h_mask,
                not_dmso_6h_or_24h_mask=cmap.not_dmso_6h_or_24h_mask,
                not_dmso_6h_mask=~cmap.is_dmso_6h_mask,
                cloud_refs=cmap.cloud_refs,
                perturbations=cmap.perturbations,
                tissues=cmap.tissues
            )
        )

    def augment(self, augmented_raw_samples: NDArray[float], augmented_raw_samples_t: Tensor) -> MultiDeviceData:
        assert_same_len(self.on_cpu.raw_samples, augmented_raw_samples, augmented_raw_samples_t)
        return MultiDeviceData(
            on_device_original_space_anchor_points=self.on_device_original_space_anchor_points,
            on_device=dataclasses.replace(self.on_device, raw_samples=augmented_raw_samples_t),
            on_cpu=dataclasses.replace(self.on_cpu, raw_samples=augmented_raw_samples),
        )
