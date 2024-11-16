from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Union

from src.configuration import config
from src.perturbation import Perturbation
from src.tissue import Tissue


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class CmapCloudRef:
    tissue: Tissue
    perturbation: Perturbation

    def __init__(self, tissue: Union[str, Tissue], perturbation: Union[str, Perturbation]):
        """
        Ctor
        :param tissue: Code or full tissue, can be either str or Tissue
        :param perturbation: Can be either str or Perturbation
        """
        # Please read https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "tissue", Tissue(tissue))
        object.__setattr__(self, "perturbation", Perturbation(perturbation))

    def change_perturbation(self, perturbation: Perturbation) -> CmapCloudRef:
        return CmapCloudRef(
            tissue=self.tissue,
            perturbation=perturbation
        )

    @cached_property
    def tissue_code(self) -> str:
        return self.tissue.tissue_code

    @cached_property
    def is_dmso_6h(self) -> bool:
        return self.perturbation.is_dmso_6h

    @cached_property
    def is_dmso_24h(self) -> bool:
        return self.perturbation.is_dmso_24h

    @cached_property
    def is_not_dmso_6h_or_24h(self) -> bool:
        return not self.perturbation.is_dmso_6h and not self.perturbation.is_dmso_24h

    @cached_property
    def is_dmso_6h_or_24h(self) -> bool:
        return not self.is_not_dmso_6h_or_24h

    @cached_property
    def dmso_6h(self) -> CmapCloudRef:
        return CmapCloudRef(self.tissue, config.dmso_6h_perturbation)

    @cached_property
    def dmso_24h(self) -> CmapCloudRef:
        return CmapCloudRef(self.tissue, config.time_24h_perturbation)

    @cached_property
    def name(self) -> str:
        return f'{self.tissue.tissue_code} {self.perturbation}'

    def __lt__(self, other: CmapCloudRef):
        """
        It is a hack that lets you use __lt__
        """
        return self.name.__lt__(other.name)
