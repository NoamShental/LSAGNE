import math
import os
import re
import time
from dataclasses import dataclass, field
from functools import cached_property
from logging import Logger
from os import PathLike
from typing import Dict, Tuple, Iterator, Union, Collection, List, Iterable

from numpy.typing import NDArray
from torch import Tensor

from src.assertion_utils import assert_same_len
from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_data_augmentation_v1.generate_partial_augmentations_v1 import Augmentation
from src.configuration import config
import pickle
import numpy as np

from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.model_learning_parameters import AugmentationParams


class CmapAugmentation:
    def __init__(
            self,
            logger: Logger,
            clouds_to_augment: Collection[CmapCloudRef],
            offline_augmentations_folder_path: PathLike | None = None,
            cloud_ref_to_augmentations: dict[CmapCloudRef, NDArray[float]] | None = None,
            random_seed: int | None = None
    ):
        self.logger = logger
        if not offline_augmentations_folder_path and not cloud_ref_to_augmentations:
            raise ValueError('At least on source of augmentation data should be provided')
        self.cloud_ref_to_augmentation = cloud_ref_to_augmentations \
            if cloud_ref_to_augmentations \
            else self._load_cloud_ref_to_augmentations(offline_augmentations_folder_path, clouds_to_augment)
        self.cloud_ref_to_current_idx: Dict[CmapCloudRef, int] = {
            cloud_ref: 0
            for cloud_ref, augmentations in self.cloud_ref_to_augmentation.items()
        }
        for cloud_to_augment in clouds_to_augment:
            assert cloud_to_augment in self.cloud_ref_to_augmentation, f'Could not find augmentations for cloud {cloud_to_augment}'
        self.rng = np.random.default_rng(random_seed)

    def _load_cloud_ref_to_augmentations(
            self,
            offline_augmentations_folder_path: str,
            clouds_to_augment: Collection[CmapCloudRef]
    ) -> Dict[CmapCloudRef, Augmentation]:
        cloud_ref_to_augmentation: Dict[CmapCloudRef, Augmentation] = {}
        self.logger.info(f'Loading offline augmentations from "{offline_augmentations_folder_path}"...')
        for root, dirs, files in os.walk(offline_augmentations_folder_path):
            for cloud_to_augment in clouds_to_augment:
                self.logger.info(f'Loading augmentation for {cloud_to_augment}...')
                regex = re.compile(f'.*{cloud_to_augment.tissue_code}_{cloud_to_augment.perturbation}.*.pkl')
                matching_files = list(filter(regex.match, files))
                if len(matching_files) == 0:
                    raise AssertionError(f'No augmentation file for cloud {cloud_to_augment} using the pattern "{regex.pattern}".')
                if len(matching_files) > 1:
                    raise AssertionError(f'{cloud_to_augment} have more than one matching augmentation file.')
                with open(os.path.join(root, matching_files[0]), 'rb') as file:
                    current_cloud_ref_to_augmentation: dict[CmapCloudRef, Augmentation] = pickle.load(file)
                cloud_ref, augmentation = current_cloud_ref_to_augmentation.popitem()
                assert cloud_ref == cloud_to_augment
                assert len(current_cloud_ref_to_augmentation) == 0, 'Should contain augmentation for a single cloud'
                cloud_ref_to_augmentation[cloud_ref] = augmentation
                self.logger.info(f'Done augmentation for {cloud_to_augment}.')
        self.logger.info('Done loading offline augmentations.')
        return cloud_ref_to_augmentation

    @cached_property
    def supported_cloud_refs(self) -> List[CmapCloudRef]:
        return list(self.cloud_ref_to_current_idx.keys())

    def shuffle_inplace(self, *arrs: NDArray):
        n = len(arrs[0])
        idx = np.arange(n)
        self.rng.shuffle(idx)
        for arr in arrs:
            arr[:] = arr[idx]

    def next_augmentation(self, cloud_ref: CmapCloudRef, k: int) -> tuple[NDArray[bool], NDArray[float]]:
        start_idx = self.cloud_ref_to_current_idx[cloud_ref]
        mask, augmentations = self.cloud_ref_to_augmentation[cloud_ref]
        n = len(augmentations)
        if start_idx + k > n:
            prefix_amount = n - start_idx
            prefix_mask = mask[start_idx:]
            prefix_augmentations = augmentations[start_idx:]
            self.shuffle_inplace(mask, augmentations)
            self.cloud_ref_to_current_idx[cloud_ref] = 0
            suffix_amount = k - prefix_amount
            suffix_mask, suffix_augmentations = self.next_augmentation(cloud_ref, suffix_amount)
            return np.vstack([prefix_mask, suffix_mask]), np.vstack([prefix_augmentations, suffix_augmentations])
        self.cloud_ref_to_current_idx[cloud_ref] += k
        end_idx = start_idx+k
        return mask[start_idx:end_idx], augmentations[start_idx:end_idx]

    def augment_samples_inplace(
            self,
            samples: NDArray | Tensor,
            cloud_refs: NDArray[CmapCloudRef],
            mask: NDArray[bool | int] | None = None
    ) -> Union[NDArray, Tensor]:
        assert_same_len(samples, cloud_refs)
        cloud_refs = cloud_refs[mask] if mask is not None else cloud_refs
        unique_cloud_refs, counts = np.unique(cloud_refs, return_counts=True)
        original_samples = samples
        samples = samples[mask] if mask is not None else samples
        for cloud_ref, count in zip(unique_cloud_refs, counts):
            cloud_idx = cloud_refs == cloud_ref
            idx, augmentation = self.next_augmentation(cloud_ref, count)
            cloud_samples = samples[cloud_idx]
            cloud_samples[idx] = augmentation[idx]
            samples[cloud_idx] = cloud_samples
        original_samples[mask] = samples
        return original_samples


@dataclass
class CmapAugmentationParams:
    prob: float
    clouds_to_augment: Collection[CmapCloudRef]
    offline_augmentations_folder_path: PathLike | None = None
    cloud_ref_to_augmentations: dict[CmapCloudRef, NDArray[float]] | None = None

@dataclass
class AugmentorChooser:
    logger: Logger
    augmentation_params: list[AugmentationParams]
    random_seed: int | None = None

    augmentors: list[CmapAugmentation | None] = field(init=False, default_factory=list)
    probs: list[float] = field(init=False, default_factory=list)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.random_seed)
        for params in self.augmentation_params:
            self.augmentors.append(CmapAugmentation(
                logger=self.logger,
                clouds_to_augment=params.clouds_to_augment,
                offline_augmentations_folder_path=params.augmentation_path,
                random_seed=self.random_seed
            ))
            self.probs.append(params.prob)
        probs_sum = sum(self.probs)
        assert probs_sum <= 1 or math.isclose(probs_sum, 1), f'value {probs_sum} does not meet requirements'
        if not math.isclose(probs_sum, 1):
            self.probs.append(1 - probs_sum)
            self.augmentors.append(None)

    def choose_augmentor(self) -> tuple[CmapAugmentation | None, float | None]:
        idx = self.rng.choice(len(self.probs), p=self.probs)
        augmentation_rate = None if self.augmentors[idx] is None else self.augmentation_params[idx].augmentation_rate
        return self.augmentors[idx], augmentation_rate


if __name__ == '__main__':
    print('Starting to create mock augmentation dict...')
    FILE_PATH = os.path.join(config.organized_data_augmentation_folder, 'test.pkl')
    SAMPLES_PER_CLOUD = 1_000
    TISSUES_NAMES = list(config.tissue_code_to_name.keys())
    PERTURBATIONS_NAMES = config.perturbations_whitelist
    GENES_PER_SAMPLE = 5
    GENES = np.arange(977)

    cloud_to_augmentations = {}

    cmap = RawCmapDataset.load_dataset_from_disk(config.perturbations_whitelist, config.tissues_whitelist)

    start = time.time()
    for perturbation, tissue in cmap.perturbation_and_tissue_to_encoded_label.keys():
        tissue_code = config.tissue_name_to_code[tissue]
        size = (SAMPLES_PER_CLOUD, GENES_PER_SAMPLE)
        genes_to_augment = np.random.choice(GENES, size)
        fold_change = np.random.normal(0, 0.001, size)

        current_cloud_augmentations = []
        for i in range(size[0]):
            current_cloud_augmentations.append((genes_to_augment[i, :], fold_change[i, :]))

        cloud_to_augmentations[(tissue_code, perturbation)] = current_cloud_augmentations

    end = time.time()
    print(f'Preparation took {end - start} secs.')

    print('Pickling...')
    start = time.time()
    with open(FILE_PATH, 'wb') as handle:
        pickle.dump(cloud_to_augmentations, handle, protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print(f'Pickling took {end - start} secs.')

    print('Loading Pickle...')
    start = time.time()
    with open(FILE_PATH, 'rb') as handle:
        x = pickle.load(handle)
    end = time.time()
    print(f'Loading Pickle took {end - start} secs.')

    print('Done!')


