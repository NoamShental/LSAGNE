from dataclasses import dataclass
from typing import Optional, Any

from src.random_manager import RandomManager


@dataclass
class TechnicalLearningParameters:
    device: Any
    random_manager: RandomManager
    working_directory: str
