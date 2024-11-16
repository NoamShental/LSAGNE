from __future__ import annotations

from dataclasses import dataclass
from typing import Sized, Optional

import torch
from torch import Tensor

from src.assertion_utils import assert_same_len
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.lsagne_model import LsagneModel
from src.samples_embedding import SamplesEmbedding


@dataclass(frozen=True)
class CmapEvaluationData(Sized):
    cmap: RawCmapDataset
    raw_samples_t: Tensor
    embedding: SamplesEmbedding

    def __post_init__(self):
        assert_same_len(self.cmap, self.raw_samples_t, self.embedding)

    @property
    def z_t(self) -> Tensor:
        return self.embedding.z_t

    def __len__(self) -> int:
        return len(self.embedding)


@dataclass(frozen=True)
class SplittedCmapEvaluationData:
    training_only: CmapEvaluationData
    training_concealed: CmapEvaluationData
    cross_validation: Optional[CmapEvaluationData] = None
    left_out: Optional[CmapEvaluationData] = None

    # todo move to some generic utils - this can be used often
    @classmethod
    def _calculate_embedding_of(cls, cmap: RawCmapDataset, model: LsagneModel) -> CmapEvaluationData:
        raw_samples_t = torch.tensor(cmap.data, device=model.device)
        embedding = model.get_embedding(raw_samples_t)
        return CmapEvaluationData(
            cmap=cmap,
            raw_samples_t=raw_samples_t,
            embedding=embedding
        )

    @classmethod
    def create_instance(
            cls,
            training_only_cmap: RawCmapDataset,
            training_concealed_cmap: RawCmapDataset,
            model: LsagneModel,
            cross_validation: Optional[RawCmapDataset] = None,
            left_out: Optional[RawCmapDataset] = None
            ) -> SplittedCmapEvaluationData:
        return cls(
            training_only=cls._calculate_embedding_of(training_only_cmap, model),
            training_concealed=cls._calculate_embedding_of(training_concealed_cmap, model),
            cross_validation=cls._calculate_embedding_of(cross_validation, model) if cross_validation else None,
            left_out=cls._calculate_embedding_of(left_out, model) if left_out else None
        )

    def add_cv_and_left_out(
            self,
            cross_validation: RawCmapDataset,
            left_out: RawCmapDataset,
            model: LsagneModel) -> SplittedCmapEvaluationData:
        return SplittedCmapEvaluationData(
            training_only=self.training_only,
            training_concealed=self.training_concealed,
            cross_validation=self._calculate_embedding_of(cross_validation, model),
            left_out=self._calculate_embedding_of(left_out, model)
        )
