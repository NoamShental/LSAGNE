from __future__ import annotations

from functools import cached_property
from typing import Union

from src.configuration import config


class Tissue(str):
    ALLOWED_TISSUES = set(config.tissues_whitelist)

    def __new__(cls, value: Union[str, Tissue]) -> Tissue:
        if value in config.tissue_code_to_name:
            value = config.tissue_code_to_name[value]
        if value not in cls.ALLOWED_TISSUES:
            raise AssertionError(f'Tissue "{value}" is not part of the whitelist / a short tissue code')
        return super().__new__(cls, value)

    @cached_property
    def tissue_code(self) -> str:
        return config.tissue_name_to_code[self]
