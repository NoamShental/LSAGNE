from __future__ import annotations
from functools import cached_property
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from src.configuration import config


class Perturbation(str):
    ALLOWED_PERTURBATIONS = {*config.perturbations_whitelist, config.dmso_6h_perturbation}

    def __init__(self, value: str):
        if value not in self.ALLOWED_PERTURBATIONS:
            raise AssertionError(f'Perturbation "{value}" is not part of the whitelist')

    # Add support for Pydantic parsing
    @classmethod
    def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))

    @cached_property
    def is_dmso_6h(self) -> bool:
        return self == config.dmso_6h_perturbation

    @cached_property
    def is_dmso_24h(self) -> bool:
        return self == config.time_24h_perturbation

    @classmethod
    def TIME_24H(cls) -> Perturbation:
        return cls(config.time_24h_perturbation)

    @classmethod
    def DMSO_6H(cls) -> Perturbation:
        return cls(config.dmso_6h_perturbation)
