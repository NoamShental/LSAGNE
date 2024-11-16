from dataclasses import dataclass
from functools import cached_property
from typing import Dict, TypeVar, Callable, Collection, Generic

import numpy as np
from numpy.typing import NDArray

from src.cmap_cloud_ref import CmapCloudRef

T = TypeVar('T')


@dataclass(frozen=True, init=False)
class ClassEncoder(Generic[T]):
    class_to_encoded_label: Dict[T, int]
    encoded_label_to_class: Dict[int, T]
    np_class_to_encoded_label_vectorize: np.vectorize
    np_encoded_label_to_class_vectorize: np.vectorize

    def __init__(self, all_classes: Collection[T]):
        class_to_encoded_label: Dict[CmapCloudRef, int] = {}
        encoded_label_to_class: Dict[int, CmapCloudRef] = {}
        for i, cls in enumerate(set(all_classes)):
            class_to_encoded_label[cls] = i
            encoded_label_to_class[i] = cls
        object.__setattr__(self, 'class_to_encoded_label', class_to_encoded_label)
        object.__setattr__(self, 'encoded_label_to_class', encoded_label_to_class)

    @cached_property
    def np_class_to_encoded_label_vectorize(self) -> Callable[[NDArray[T]], NDArray[int]]:
        return np.vectorize(lambda cls: self.class_to_encoded_label[cls])

    @cached_property
    def np_encoded_label_to_class_vectorize(self) -> Callable[[NDArray[int]], NDArray[T]]:
        return np.vectorize(lambda label: self.encoded_label_to_class[label])
