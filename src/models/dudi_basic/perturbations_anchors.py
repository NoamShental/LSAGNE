from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar('T')


@dataclass(frozen=True)
class PerturbationsAnchors(Generic[T]):
    treated: T
    dmso_6: T
    dmso_24: T
