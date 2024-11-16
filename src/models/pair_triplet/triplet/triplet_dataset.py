import numpy as np
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset

from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset


class TripletCmap(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
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

    def __getitem__(self, index):
        sample1, label1 = self.data[index], self.labels[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(list(self.labels_set - {label1}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        sample2 = self.data[positive_index]
        sample3 = self.data[negative_index]

        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
            sample3 = self.transform(sample3)
        return (sample1, sample2, sample3), [], self.class_weights[label1]

    def __len__(self):
        return len(self.raw_cmap_dataset)