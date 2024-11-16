from typing import Dict

import numpy as np
import torch
from torch.types import Device


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size: int, shuffle: bool):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert len(tensors) > 0, 'at least one tensor or dict should be passed.'
        if isinstance(tensors[0], dict):
            self.dict = True
            self.names = list(tensors[0].keys())
            self.tensors = list(tensors[0].values())
        else:
            assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
            self.dict = False
            self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.dict:
            batch = {name: t[self.i:self.i + self.batch_size] for name, t in zip(self.names, self.tensors)}
        else:
            batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

    @classmethod
    def from_dict_dataset(cls, all_data_lists_dict: Dict, batch_size: int, device: Device):
        all_data_dict_t = dict()
        for k in all_data_lists_dict:
            all_data_dict_t[k] = np.array(all_data_lists_dict[k])
        for k in all_data_dict_t:
            if np.issubdtype(all_data_dict_t[k].dtype, np.integer):
                if all_data_dict_t[k].dtype == np.int32:
                    all_data_dict_t[k] = torch.IntTensor(all_data_dict_t[k])
                else:
                    all_data_dict_t[k] = torch.LongTensor(all_data_dict_t[k])
            else:
                all_data_dict_t[k] = torch.Tensor(all_data_dict_t[k])
            all_data_dict_t[k] = all_data_dict_t[k].to(device)
        return cls(all_data_dict_t, batch_size=batch_size, shuffle=True), all_data_dict_t

    # @classmethod
    # def from_dict_dataset(cls, dict_dataset: Dataset, batch_size: int, device: torch.device):
    #     all_data_lists_dict = defaultdict(list)
    #     for train_sample in dict_dataset:
    #         for k, v in train_sample.items():
    #             all_data_lists_dict[k].append(v)
    #     all_data_dict_t = dict()
    #     for k in all_data_lists_dict:
    #         all_data_dict_t[k] = np.array(all_data_lists_dict[k])
    #     for k in all_data_dict_t:
    #         if np.issubdtype(all_data_dict_t[k].dtype, np.integer):
    #             if all_data_dict_t[k].dtype == np.int32:
    #                 all_data_dict_t[k] = torch.IntTensor(all_data_dict_t[k])
    #             else:
    #                 all_data_dict_t[k] = torch.LongTensor(all_data_dict_t[k])
    #         else:
    #             all_data_dict_t[k] = torch.Tensor(all_data_dict_t[k])
    #         all_data_dict_t[k] = all_data_dict_t[k].to(device)
    #     return cls(all_data_dict_t, batch_size=batch_size, shuffle=True), all_data_dict_t
