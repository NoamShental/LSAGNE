import logging
import sys
from functools import lru_cache
from logging import Logger
from os import PathLike


@lru_cache(maxsize=None)
def create_logger(name: str | None = None, level=logging.DEBUG) -> Logger:
    formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)-5.5s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def add_file_handler(logger: Logger, log_file_path:  str | PathLike):
    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)

    # Determine the level and formatter from existing handlers
    if logger.handlers:  # Check if any handlers already exist
        existing_handler = logger.handlers[0]  # Assume the first handler
        file_handler.setLevel(existing_handler.level)
        file_handler.setFormatter(existing_handler.formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

