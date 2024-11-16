from abc import ABC, abstractmethod

from torch import Tensor

from src.samples_embedding import SamplesEmbedding


class SamplesEmbedder(ABC):
    @abstractmethod
    def get_embedding(self, samples: Tensor) -> SamplesEmbedding:
        raise NotImplementedError()
