from dataclasses import dataclass, field
from logging import Logger
from pathlib import Path

import pandas as pd

from src.logger_utils import create_logger


@dataclass
class DfResultsSaver:
    logger: Logger
    max_chunk_size: int
    output_path: Path
    file_name_prefix: str
    results: list[dict] = field(init=False, default_factory=list)
    chunk_i: int = field(init=False, default=0)

    def __post_init__(self):
        self.logger.info(f'max_chunk_size = {self.max_chunk_size}')

    def flush(self):
        df = pd.DataFrame(self.results)
        self.results = []
        hdf_path = self.output_path / f'{self.file_name_prefix}_chunk_{self.chunk_i}.h5'
        self.logger.info(f'Saving chunk {self.chunk_i:,} of the results to "{hdf_path}"')
        self.chunk_i += 1
        df.to_hdf(hdf_path, key='df')

    def add_result(self, result: dict):
        self.results.append(result)
        if len(self.results) == self.max_chunk_size:
            self.flush()


# @dataclass
# class DfResultsLoader:
#     results_path: Path
#     take: int | None = None
#     logger: Logger = field(init=False, default_factory=lambda: create_logger('results loader'))
#
#     def load(self) -> pd.DataFrame:
#         results = []
#         for file_path in self.results_path.iterdir():
#             results.append(pd.read_hdf(file_path, key='df'))
#         return pd.concat(results)
