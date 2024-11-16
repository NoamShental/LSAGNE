from datetime import datetime
from logging import Logger
from typing import Optional
import torch
import numpy as np


class RandomManager:
    @property
    def random_seed(self) -> Optional[int]:
        return self._random_seed

    def __init__(self, use_seed: bool, random_seed: Optional[int], logger: Optional[Logger] = None):
        self._random_seed = None
        self.log = logger.info if logger is not None else print
        if use_seed:
            if random_seed is not None:
                self._random_seed = random_seed
            else:
                self._random_seed = datetime.now().microsecond
            self.apply_seed_if_needed()

    def apply_seed_if_needed(self):
        if self._random_seed is not None:
            self.log(f'Random seed is {self._random_seed}.')
            torch.manual_seed(self._random_seed)
            np.random.seed(self._random_seed)
