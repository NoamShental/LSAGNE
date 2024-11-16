import numpy as np
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset

from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset


class SiameseCmap(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, raw_cmap_dataset: RawCmapDataset, transform=None):
        self.raw_cmap_dataset = raw_cmap_dataset
        self.transform = transform
        self.labels = self.raw_cmap_dataset.labels
        self.data = self.raw_cmap_dataset.raw_cmap_data
        self.unique_labels = list(set(self.labels))
        self.labels_set = set(self.labels)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.labels), y=self.labels)
        self.normalized_class_weights = self.class_weights / sum(self.class_weights)
        self.counter = 0

    def __getitem__(self, index):
        return self._regular_getitem(index)
        # self.counter = self.counter + 1
        # if self.counter > 0 * self.__len__():
        #     return self._probability_getitem(index)
        # else:
        #     return self._regular_getitem(index)


    def _probability_getitem(self, index):
        # target == 0 -> choose different labels
        # target == 1 -> choose same labels
        target = np.random.randint(0, 2)
        label1 = np.random.choice(self.unique_labels, p=self.normalized_class_weights)
        index1 = np.random.choice(self.label_to_indices[label1])
        sample1 = self.data[index1]
        if target == 1:
            siamese_index = index
            siamese_label = label1
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - {label1}))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        sample2 = self.data[siamese_index]

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return (sample1, sample2), target, max(self.class_weights[label1], self.class_weights[siamese_label])

    def _probability_getitem2(self, index):
        # target == 0 -> choose different labels
        # target == 1 -> choose same labels
        target = np.random.randint(0, 2)
        label1 = np.random.choice(self.unique_labels, p=self.normalized_class_weights)
        index1 = np.random.choice(self.label_to_indices[label1])
        sample1 = self.data[index1]
        if target == 1:
            siamese_index = index
            siamese_label = label1
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            labels_without_label1 = np.delete(self.unique_labels, label1)
            normalized_class_weights_without_label1 = np.delete(self.normalized_class_weights, label1)
            normalized_class_weights_without_label1 = normalized_class_weights_without_label1 / \
                                                      sum(normalized_class_weights_without_label1)
            siamese_label = np.random.choice(labels_without_label1, p=normalized_class_weights_without_label1 )
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        sample2 = self.data[siamese_index]

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        return (sample1, sample2), target, (self.class_weights[label1] + self.class_weights[siamese_label]) / 2



    def _regular_getitem(self, index):
        # target == 0 -> choose different labels
        # target == 1 -> choose same labels
        target = np.random.randint(0, 2)
        sample1, label1 = self.data[index], self.labels[index]
        if target == 1:
            siamese_index = index
            siamese_label = label1
            while siamese_index == index:
                siamese_index = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = np.random.choice(list(self.labels_set - {label1}))
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])
        sample2 = self.data[siamese_index]

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        # return (sample1, sample2), target, (self.class_weights[label1] + self.class_weights[siamese_label]) / 2
        return (sample1, sample2), target, max(self.class_weights[label1], self.class_weights[siamese_label])

    def __len__(self):
        return len(self.raw_cmap_dataset)