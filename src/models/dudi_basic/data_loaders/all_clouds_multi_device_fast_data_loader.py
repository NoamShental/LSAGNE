from dataclasses import dataclass

import numpy as np

from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.data_loaders.multi_device_fast_data_loader import MultiDeviceFastDataLoader
from src.models.dudi_basic.multi_device_data import MultiDeviceData


@dataclass
class AllCloudsMultiDeviceFastDataLoader(MultiDeviceFastDataLoader):
    multi_device_data: MultiDeviceData
    number_of_batches: int
    number_of_samples_per_cloud: int
    cmap: RawCmapDataset

    def __post_init__(self):
        assert all(self.multi_device_data.on_cpu.encoded_labels == self.cmap.encoded_labels), \
            "multi_device_data & CMAP are uncorrelated!"

    def __iter__(self):
        self.batch_i = 0
        return self

    def __next__(self):
        if self.batch_i >= self.number_of_batches:
            raise StopIteration
        batch_idx_list = []
        for cloud_ref in self.cmap.unique_cloud_refs:
            batch_idx_list.append(np.random.choice(self.cmap.cloud_ref_to_idx[cloud_ref], self.number_of_samples_per_cloud, replace=False))
        batch_idx = np.concatenate(batch_idx_list)
        np.random.shuffle(batch_idx)
        batch = self.multi_device_data.create_batch(batch_idx)
        self.batch_i += 1
        return batch

    def __len__(self):
        return self.number_of_batches
