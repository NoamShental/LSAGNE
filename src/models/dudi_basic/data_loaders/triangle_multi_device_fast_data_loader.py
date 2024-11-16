from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

from numpy.typing import NDArray
from torch.types import Device

from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.data_loaders.create_multi_device_data_on_device import create_multi_device_data_on_device
from src.models.dudi_basic.multi_device_data import MultiDeviceData
from src.models.dudi_basic.data_loaders.multi_device_fast_data_loader import MultiDeviceFastDataLoader

import numpy as np

@dataclass
class TriangleMultiDeviceFastDataLoader(MultiDeviceFastDataLoader):
    multi_device_data: MultiDeviceData
    number_of_batches: int
    number_of_tissues_per_batch: int
    number_of_samples_per_cloud: int
    desired_number_of_triangles_per_batch: int
    cmap: RawCmapDataset
    encoded_tissues_balanced_probabilities: NDArray[float] = field(init=False)
    encoded_dmso_6h_perturbation: int = field(init=False)
    encoded_time_24h_perturbation: int = field(init=False)
    encoded_label_to_idx: Dict[int, NDArray[int]] = field(init=False)
    encoded_tissue_to_treated_encoded_labels: Dict[int, List[int]] = field(init=False, default_factory=lambda: defaultdict(list))
    encoded_tissue_to_treated_encoded_labels_balanced_probabilities: Dict[int, List[int]] = field(init=False, default_factory=dict)

    def __post_init__(self):
        if len(np.unique(self.multi_device_data.on_cpu['encoded_tissues'])) < self.number_of_tissues_per_batch:
            raise AssertionError('Number of tissues is less than the number desired in each batch')

        assert all(self.multi_device_data.on_cpu['encoded_numeric_labels'] == self.cmap.encoded_labels), \
            "multi_device_data & CMAP are uncorrelated!"

        self.encoded_tissues_balanced_probabilities = self.cmap.encoded_tissues_to_balanced_class_weights / \
                                                      self.cmap.encoded_tissues_to_balanced_class_weights.sum()
        self.encoded_dmso_6h_perturbation = self.cmap.control_encoded_perturbations[0]
        self.encoded_time_24h_perturbation = self.cmap.encoded_dmso_24h_perturbation
        self.encoded_label_to_idx = self.cmap.encoded_label_to_idx
        for encoded_tissue in self.cmap.encoded_tissues_unique:
            current_tissue_encoded_perturbations = self.cmap.encoded_tissue_to_encoded_perturbations[encoded_tissue].copy()
            for encoded_perturbation in current_tissue_encoded_perturbations:
                if encoded_perturbation in [self.encoded_dmso_6h_perturbation, self.encoded_time_24h_perturbation]:
                    continue
                encoded_label = self.cmap.encoded_perturbation_and_tissue_to_encoded_label[encoded_perturbation, encoded_tissue]
                self.encoded_tissue_to_treated_encoded_labels[encoded_tissue].append(encoded_label)
            self.encoded_tissue_to_treated_encoded_labels_balanced_probabilities[encoded_tissue] = \
                self.cmap.encoded_labels_to_balanced_class_weights[
                    self.encoded_tissue_to_treated_encoded_labels[encoded_tissue]]
            self.encoded_tissue_to_treated_encoded_labels_balanced_probabilities[encoded_tissue] /= \
                np.sum(self.encoded_tissue_to_treated_encoded_labels_balanced_probabilities[encoded_tissue])

    def __iter__(self):
        self.batch_i = 0
        return self

    def __next__(self):
        if self.batch_i >= self.number_of_batches:
            raise StopIteration
        batch_idx_list = []
        chosen_encoded_tissues = np.random.choice(self.cmap.encoded_tissues_unique,
                                                  self.number_of_tissues_per_batch,
                                                  replace=False,
                                                  p=self.encoded_tissues_balanced_probabilities)
        for encoded_tissue in chosen_encoded_tissues:
            current_tissue_dmso_6h_encoded_label = self.cmap.encoded_perturbation_and_tissue_to_encoded_label[
                (self.encoded_dmso_6h_perturbation, encoded_tissue)]
            current_tissue_time_24h_encoded_label = self.cmap.encoded_perturbation_and_tissue_to_encoded_label[
                (self.encoded_time_24h_perturbation, encoded_tissue)]
            current_tissue_dmso_6h_idx = self.encoded_label_to_idx[current_tissue_dmso_6h_encoded_label]
            current_tissue_time_24h_idx = self.encoded_label_to_idx[current_tissue_time_24h_encoded_label]
            batch_idx_list.append(np.random.choice(current_tissue_dmso_6h_idx, self.number_of_samples_per_cloud, replace=False))
            batch_idx_list.append(np.random.choice(current_tissue_time_24h_idx, self.number_of_samples_per_cloud, replace=False))

            number_of_triangles_to_choose = self.desired_number_of_triangles_per_batch
            if len(self.encoded_tissue_to_treated_encoded_labels[encoded_tissue]) < number_of_triangles_to_choose:
                number_of_triangles_to_choose = len(self.encoded_tissue_to_treated_encoded_labels[encoded_tissue])

            chosen_clouds_encoded_labels = np.random.choice(
                self.encoded_tissue_to_treated_encoded_labels[encoded_tissue],
                number_of_triangles_to_choose,
                replace=False,
                p=self.encoded_tissue_to_treated_encoded_labels_balanced_probabilities[encoded_tissue]
            )

            for treated_cloud_encoded_label in chosen_clouds_encoded_labels:
                batch_idx_list.append(np.random.choice(self.encoded_label_to_idx[treated_cloud_encoded_label],
                                              self.number_of_samples_per_cloud,
                                              replace=False))

        batch_idx = np.concatenate(batch_idx_list)
        batch = MultiDeviceData(
            on_device_lookup=self.multi_device_data.on_device_lookup,
            on_device={k: t[batch_idx] for k, t in self.multi_device_data.on_device.items()},
            on_cpu={k: t[batch_idx] for k, t in self.multi_device_data.on_cpu.items()}
        )

        self.batch_i += 1

        return batch

    def __len__(self):
        return self.number_of_batches

    @classmethod
    def from_multi_device_data(
            cls,
            on_cpu_multi_device_data: MultiDeviceData,
            device: Device,
            number_of_batches: int,
            number_of_tissues_per_batch: int,
            number_of_samples_per_cloud: int,
            desired_number_of_triangles_per_batch: int,
            cmap: RawCmapDataset
    ):
        multi_device_data_t = create_multi_device_data_on_device(on_cpu_multi_device_data, device)
        return cls(
            multi_device_data=multi_device_data_t,
            number_of_batches=number_of_batches,
            number_of_tissues_per_batch=number_of_tissues_per_batch,
            number_of_samples_per_cloud=number_of_samples_per_cloud,
            desired_number_of_triangles_per_batch=desired_number_of_triangles_per_batch,
            cmap=cmap
        ), multi_device_data_t
