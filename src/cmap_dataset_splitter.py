from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import Logger
from typing import Dict, Collection, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset


@dataclass(frozen=True)
class SplittedCmapDataset:
    training_only: RawCmapDataset
    training_concealed: RawCmapDataset
    left_out: RawCmapDataset
    cross_validation: RawCmapDataset

    @cached_property
    def full_cmap_dataset(self) -> RawCmapDataset:
        return RawCmapDataset.merge_datasets(
            self.training_only,
            self.training_concealed,
            self.left_out,
            self.cross_validation
        )

    @staticmethod
    def _split_partials(
            logger: Logger,
            full_training_dataset: RawCmapDataset,
            cloud_ref_to_partial_training_size: Optional[Dict[CmapCloudRef, int]]
    ) -> Tuple[RawCmapDataset, RawCmapDataset]:
        not_in_training_idx_mask: NDArray[bool] = np.full(len(full_training_dataset), False)
        for cloud_ref, partial_training_size in cloud_ref_to_partial_training_size.items():
            cloud_idx = full_training_dataset.cloud_ref_to_idx[cloud_ref]
            not_in_training_idx = np.random.choice(cloud_idx, len(cloud_idx) - partial_training_size, replace=False)
            not_in_training_idx_mask[not_in_training_idx] = True
        training_only_idx_mask = ~not_in_training_idx_mask
        training_dataset, concealed_from_training_dataset = full_training_dataset.split_by_mask(training_only_idx_mask)
        return training_dataset, concealed_from_training_dataset
    
    @classmethod
    def split_cmap_dataset(
            cls,
            logger: Logger,
            original_cmap_dataset: RawCmapDataset,
            left_out_cloud_ref: CmapCloudRef,
            cross_validation_cloud_refs: Collection[CmapCloudRef],
            cloud_ref_to_partial_training_size: Dict[CmapCloudRef, int],
    ) -> SplittedCmapDataset:
        full_training_dataset, left_out_cmap_dataset = original_cmap_dataset.leave_out(left_out_cloud_ref)
        if len(left_out_cmap_dataset) == 0:
            logger.critical('No left out items, stopping this task!')
            raise AssertionError('Left out dataset is empty.')
        assert left_out_cmap_dataset.clouds_count == 1, 'Only one class should be left out'
        logger.info(f'Left out: tissue={left_out_cloud_ref.tissue}, perturbation={left_out_cloud_ref.perturbation}.')
        logger.info(f'Left out class numeric label = {left_out_cmap_dataset.original_numeric_labels_unique}')
        logger.info(f'Left out CMAP dataset:\n{left_out_cmap_dataset.summary}\n')
        logger.info(f'CMAP without left out:\n{full_training_dataset.summary}\n')

        logger.info(f'Taking cross validations {cross_validation_cloud_refs}".')
        full_training_dataset, cross_validation_dataset = full_training_dataset.leave_out(*cross_validation_cloud_refs)

        training_dataset, concealed_from_training_dataset = cls._split_partials(logger, full_training_dataset, cloud_ref_to_partial_training_size)
        assert len(full_training_dataset) == len(training_dataset) + len(concealed_from_training_dataset), "Sizes mismatch"
        logger.info(f'Using partial clouds training size reduced {len(full_training_dataset):,} --> {len(training_dataset):,}')

        return cls(
            training_only=training_dataset,
            training_concealed=concealed_from_training_dataset,
            left_out=left_out_cmap_dataset,
            cross_validation=cross_validation_dataset
        )
