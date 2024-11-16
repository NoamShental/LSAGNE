from abc import ABC
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, PositiveInt, model_validator

from src.cmap_data_augmentation_v1.generate_partial_augmentations_v1 import AugmentationVariace
from src.perturbation import Perturbation


class CacheParameters(BaseModel,ABC):
    alias: str | None


class CmapOrganizationParameters(CacheParameters):
    raw_cmap_folder: Path
    perturbation_to_max_size: dict[Perturbation, int] | None = None
    random_seed: int | None = None
    recreate_for_each_run: bool = False


class LeftOneOutRun(BaseModel, frozen=True):
    left_out: tuple[str, str]
    cv: tuple[str, str]
    augmented_perturbations: frozenset[Perturbation]
    cmap: str
    fold_change_factor: float = 1.0

class LeftOneOutSpecs(BaseModel):
    number_of_seeds: int
    number_of_repeats: int
    clouds: list[LeftOneOutRun]


class SlurmResources(BaseModel):
    cpu_cores: PositiveInt
    ram_gb: PositiveInt


class RunningPaths(BaseModel):
    root: Path
    code: Path


class TechnicalSpecs(BaseModel):
    conda_env_path: str
    paths: RunningPaths
    augmentation_resources: SlurmResources
    training_resources: SlurmResources
    use_gpu: Literal['a100', 'titan'] | None


class AugmentationParametersDbBuilding(CacheParameters):
    min_drug_samples_per_cellline: int
    min_cellines_perdrug: int
    min_genes_per_go: int
    max_genes_per_go: int
    use_compression: bool
    calc_beta: bool


class AugmentationParametersSamplesGeneration(CacheParameters):
    num_of_samples: int
    n_pathways: int
    n_corrpathways: int
    proba_pathway: float
    n_genes: int
    n_corrgenes: int
    proba_gene: float
    use_variance: AugmentationVariace


class AugmentationParameters(BaseModel):
    raw_augmentation_folder: Path
    drug_batch_size: int
    augmentation_rate: float
    db_building: AugmentationParametersDbBuilding
    samples_generation: list[AugmentationParametersSamplesGeneration]


class ExperimentSpecs(BaseModel):
    name: Path
    api_ver: int
    technical_specs: TechnicalSpecs
    cmap_organization_parameters: list[CmapOrganizationParameters]
    augmentation_parameters: AugmentationParameters
    left_one_out_specs: LeftOneOutSpecs
    training_parameters: dict
