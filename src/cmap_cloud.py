from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Sized, Optional

import numpy as np
from numpy.typing import NDArray

from src.assertion_utils import assert_promise_true
from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_cloud_ref_and_tag import CmapCloudRefAndTag


@dataclass(frozen=True)
class CmapCloud(CmapCloudRefAndTag, Sized):
    samples: NDArray[float]
    cmap_ids: Optional[NDArray[str]]

    def __post_init__(self):
        assert len(self.samples) > 0, f'Cloud {self.cloud_ref} has no samples!'
        assert_promise_true(
            lambda: np.all(len(self.samples) == len(self.cmap_ids)) if self.cmap_ids is not None else True
        )

    @cached_property
    def cloud_refs(self) -> NDArray[CmapCloudRef]:
        return np.full(len(self.samples), self.cloud_ref)

    def __len__(self) -> int:
        return len(self.samples)
