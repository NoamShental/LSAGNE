from dataclasses import dataclass
from typing import Sized

from torch import Tensor

from src.assertion_utils import assert_same_len


@dataclass(frozen=True)
class SamplesEmbedding(Sized):
    z_t: Tensor
    mu_t: Tensor
    log_var_t: Tensor

    def __post_init__(self):
        assert_same_len(self.z_t, self.z_t, self.log_var_t)

    def __len__(self) -> int:
        return len(self.z_t)
