import torch
from torch.types import Device

from src.models.dudi_basic.data_loaders.create_multi_device_data_on_device import create_multi_device_data_on_device
from src.models.dudi_basic.multi_device_data import MultiDeviceData
from src.models.dudi_basic.data_loaders.multi_device_fast_data_loader import MultiDeviceFastDataLoader


class FullMultiDeviceFastDataLoader(MultiDeviceFastDataLoader):

    def __init__(self, multi_device_data: MultiDeviceData, batch_size: int, shuffle: bool):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        self.multi_device_data: MultiDeviceData = multi_device_data

        self.dataset_len = multi_device_data.data_count
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        # TODO use torch.split
        if self.shuffle:
            device = self.multi_device_data.device
            r_device = torch.randperm(self.dataset_len, device=device)
            r_cpu = r_device.cpu().numpy()
            self.multi_device_data = MultiDeviceData(
                on_device_lookup=self.multi_device_data.on_device_lookup,
                on_device={k: t[r_device] for k,t in self.multi_device_data.on_device.items()},
                on_cpu={k: t[r_cpu] for k, t in self.multi_device_data.on_cpu.items()}
            )
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = MultiDeviceData(
            on_device_lookup=self.multi_device_data.on_device_lookup,
            on_device={k: t[self.i:self.i + self.batch_size] for k, t in self.multi_device_data.on_device.items()},
            on_cpu={k: t[self.i:self.i + self.batch_size] for k, t in self.multi_device_data.on_cpu.items()}
        )
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

    @classmethod
    def from_multi_device_data(cls, on_cpu_multi_device_data: MultiDeviceData, batch_size: int, device: Device):
        multi_device_data_t = create_multi_device_data_on_device(on_cpu_multi_device_data, device)
        return cls(multi_device_data_t, batch_size=batch_size, shuffle=True), multi_device_data_t
