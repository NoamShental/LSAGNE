from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from shutil import copyfile, rmtree
from typing import Any, Generator, Collection

import yaml
from pydantic import BaseModel

from slurm_runner.v2.experiment_specs import ExperimentSpecs, CmapOrganizationParameters, \
    CacheParameters, AugmentationParametersSamplesGeneration, LeftOneOutRun
from slurm_runner.v2.sbatch_creator import SbatchParameters, create_sbatch_file
from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.cmap_organizer import CMAP_ORGANIZER_LOG_FILE_NAME
from src.datasets_and_organizers.cmap_utils import load_and_reduce_cmap, create_cmap_from_raw
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.logger_utils import create_logger, add_file_handler
from src.os_utilities import create_dir_if_not_exists
from src.perturbation import Perturbation

SLURM_INFO_FILE_SUFFIX = 'slurm_info.json'

SUPPORTED_API_VER = 5

logger = create_logger('experiment_manager')


@dataclass(frozen=True)
class OrganizedCmap:
    cmap: RawCmapDataset
    path: Path
    params: CmapOrganizationParameters


@dataclass(frozen=True)
class AugmentationSlurmTask:
    path: Path
    slurm_job_id: str


class SlurmJobInfo(BaseModel):
    job_id: str

    @classmethod
    def load(cls, file_path: Path) -> SlurmJobInfo:
        with open(file_path) as f:
            return cls(**json.load(f))

    def dump(self, file_path: Path):
        with open(file_path, 'w') as f:
            json.dump(json.loads(self.json()), f, indent=2)


@dataclass(frozen=True)
class ExperimentPaths:
    experiments_dir: Path
    experiment_dir: Path
    cmap_cache_dir: Path
    augmentation_db_cache_dir: Path
    augmentation_samples_cache_dir: Path
    sbatch_files_dir: Path
    experiment_sbatch_outputs_dir: Path
    results_dir: Path


def create_base_file_system_structure(root: Path, experiment_name: str) -> ExperimentPaths:
    cache_dir = root / 'cache'
    experiments_dir = root / 'experiments'
    experiment_dir = experiments_dir / experiment_name
    augmentation_cache_dir = cache_dir / 'augmentation'
    paths = ExperimentPaths(
        experiments_dir=experiments_dir,
        experiment_dir=experiment_dir,
        cmap_cache_dir=cache_dir / 'cmap',
        augmentation_db_cache_dir=augmentation_cache_dir / 'db',
        augmentation_samples_cache_dir=augmentation_cache_dir / 'samples',
        sbatch_files_dir=experiment_dir / 'sbatch_files',
        experiment_sbatch_outputs_dir=experiment_dir / 'sbatch_out_err',
        results_dir=experiment_dir / 'results'
    )
    if paths.experiment_dir.exists():
        if not LSAGNE_TEST_MODE:
            logger.error(f'The experiment {paths.experiment_dir} already exists, exiting.')
            exit(1)
        rmtree(paths.experiment_dir)
    for path_name, path in asdict(paths).items():
        logger.info(f'Creating path {path_name}: "{path}" if not exists')
        create_dir_if_not_exists(path)
    add_file_handler(logger, paths.experiment_dir / 'log.txt')
    return paths


def get_profile_path() -> Path:
    parser = ArgumentParser(description='Create an LSAGNE experiment')
    parser.add_argument(
        'experiment_specs',
        type=Path,
        help='The path to yaml file containing all the experiment specs'
    )
    args = parser.parse_args()
    return args.experiment_specs


def create_cmap(
        cmap_params: CmapOrganizationParameters,
        paths: ExperimentPaths) -> OrganizedCmap:
    logger.info(f'create_cmap random_seed is {cmap_params.random_seed:,}')
    cmap_params.alias += f'_seed_{cmap_params.random_seed}'
    cmap_already_exists, organized_cmap_dir = create_cache_folder(
        cache_type='CMAP',
        root_cache_dir=paths.cmap_cache_dir,
        params=cmap_params
    )
    if cmap_already_exists:
        return OrganizedCmap(
            cmap=RawCmapDataset.load_dataset_from_disk(
                cmap_folder=organized_cmap_dir
            ),
            path=organized_cmap_dir,
            params=cmap_params
        )
    with tempfile.TemporaryDirectory(prefix='__temp__', dir=organized_cmap_dir, ignore_cleanup_errors=True) as temp_cmap_dir:
        temp_cmap_dir = Path(temp_cmap_dir)
        print(f'Temporary directory created at: {temp_cmap_dir}')
        create_cmap_from_raw(
            raw_cmap_folder=cmap_params.raw_cmap_folder,
            output_cmap=temp_cmap_dir
        )
        logger.info(f'Cmap organizer completed, the complete log can be found at "{organized_cmap_dir / CMAP_ORGANIZER_LOG_FILE_NAME}"')
        copyfile(temp_cmap_dir / CMAP_ORGANIZER_LOG_FILE_NAME, organized_cmap_dir / CMAP_ORGANIZER_LOG_FILE_NAME)
        cmap = load_and_reduce_cmap(
            logger=logger,
            full_cmap_folder=temp_cmap_dir,
            reduced_cmap_folder=organized_cmap_dir,
            perturbation_to_resize=cmap_params.perturbation_to_max_size,
            random_seed=cmap_params.random_seed
        )
        return OrganizedCmap(
            cmap=cmap,
            path=organized_cmap_dir,
            params=cmap_params
        )


def create_augmentation_db(
        experiment_specs: ExperimentSpecs,
        paths: ExperimentPaths,
        organized_cmap: OrganizedCmap
) -> AugmentationSlurmTask:
    augmentation_db_params = copy.deepcopy(experiment_specs.augmentation_parameters.db_building)
    augmentation_db_params.alias = f'{organized_cmap.params.alias}__{augmentation_db_params.alias}'
    already_exists, cache_dir = create_cache_folder(
        "Augmentation DB",
        paths.augmentation_db_cache_dir,
        augmentation_db_params,
        {
            'raw_augmentation_folder': str(experiment_specs.augmentation_parameters.raw_augmentation_folder),
            'raw_cmap_folder': str(organized_cmap.path),
            'drug_batch_size': experiment_specs.augmentation_parameters.drug_batch_size
        }
    )
    slurm_info_file = cache_dir / SLURM_INFO_FILE_SUFFIX
    if already_exists:
        return AugmentationSlurmTask(
            path=cache_dir,
            slurm_job_id=SlurmJobInfo.load(slurm_info_file).job_id
        )
    job_name = cache_dir.name
    logger.info(f'Creating sbatch job for "{job_name}", it will be put in "{cache_dir}"')
    command_param_name_to_value = {
        '--min-drug-samples-per-cellline': augmentation_db_params.min_drug_samples_per_cellline,
        '--min-cellines-perdrug': augmentation_db_params.min_cellines_perdrug,
        '--min-genes-per-go': augmentation_db_params.min_genes_per_go,
        '--max-genes-per-go': augmentation_db_params.max_genes_per_go,
        '--drug-batch-size': experiment_specs.augmentation_parameters.drug_batch_size,
        '--raw-cmap-dir': organized_cmap.path,
        '--raw-data-augmentation-dir': experiment_specs.augmentation_parameters.raw_augmentation_folder,
        '--output-dir': cache_dir
    }
    # Jinja2 template will craete None value templates to "--arg_name" without value
    if augmentation_db_params.use_compression:
        command_param_name_to_value['--use_compression'] = None
    if augmentation_db_params.calc_beta:
        command_param_name_to_value['--calc-beta'] = None
    sbatch_params = SbatchParameters(
        conda_env_path=experiment_specs.technical_specs.conda_env_path,
        code_path=experiment_specs.technical_specs.paths.code,
        sbatch_output_file=cache_dir / f'sbatch.out',
        sbatch_error_file=cache_dir / f'sbatch.err',
        cpu_cores=experiment_specs.technical_specs.augmentation_resources.cpu_cores,
        ram_mem=experiment_specs.technical_specs.augmentation_resources.ram_gb,
        job_name=f'aug_db_{job_name}',
        command=f'python {experiment_specs.technical_specs.paths.code / "src" / "cmap_data_augmentation_v1" / "generate_augmentation_db_v1.py"}',
        command_params=command_param_name_to_value
    )
    sbatch_file_path = cache_dir / 'sbatch.sh'
    create_sbatch_file(sbatch_params, sbatch_file_path)
    logger.info(f'Submitting the augmentation DB creation job "{job_name}"')
    job_id = submit_to_slurm(sbatch_file_path)
    SlurmJobInfo(job_id=job_id).dump(slurm_info_file)
    return AugmentationSlurmTask(
        path=cache_dir,
        slurm_job_id=job_id
    )


def create_augmentation_samples(
        experiment_specs: ExperimentSpecs,
        augmentation_generation_params: AugmentationParametersSamplesGeneration,
        fold_change_factor: float,
        data_augmentation_db_task: AugmentationSlurmTask,
        organized_cmap: OrganizedCmap,
        paths: ExperimentPaths,
        cloud_ref: CmapCloudRef,
) -> AugmentationSlurmTask:
    cloud_name = f'{cloud_ref.tissue_code}_{cloud_ref.perturbation}'
    augmentation_samples_params = copy.deepcopy(augmentation_generation_params)
    augmentation_samples_params.alias = f'{organized_cmap.params.alias}__{experiment_specs.augmentation_parameters.db_building.alias}__{augmentation_samples_params.alias}_fc_{fold_change_factor:.2f}'
    already_exists, cache_dir = create_cache_folder(
        "Augmentation samples",
        paths.augmentation_samples_cache_dir,
        augmentation_samples_params,{
            'raw_augmentation_folder': str(experiment_specs.augmentation_parameters.raw_augmentation_folder),
            'drug_batch_size': experiment_specs.augmentation_parameters.drug_batch_size,
            'data_augmentation_db_dir': str(data_augmentation_db_task.path)
        })
    sbatch_info_file_path = cache_dir / f'{cloud_name}.{SLURM_INFO_FILE_SUFFIX}'
    if sbatch_info_file_path.exists():
        return AugmentationSlurmTask(
            path=cache_dir,
            slurm_job_id=SlurmJobInfo.load(sbatch_info_file_path).job_id
        )
    job_name = f'{cache_dir.name}__{cloud_name}'
    logger.info(f'Creating sbatch job for "{job_name}", it will be put in "{cache_dir}"')

    command_param_name_to_value = {
        '--data-augmentation-db-folder': data_augmentation_db_task.path,
        '--tissue': cloud_ref.tissue,
        '--perturbation': cloud_ref.perturbation,
        '--drug-batch-size': experiment_specs.augmentation_parameters.drug_batch_size,
        '--use-variance': augmentation_samples_params.use_variance,
        '--num-of-samples': augmentation_samples_params.num_of_samples,
        '--output': cache_dir / f'{cloud_name}.pkl',
        '--n-pathways': augmentation_samples_params.n_pathways,
        '--n-corrpathways': augmentation_samples_params.n_corrpathways,
        '--proba-pathway': augmentation_samples_params.proba_pathway,
        '--n-genes': augmentation_samples_params.n_genes,
        '--n-corrgenes': augmentation_samples_params.n_corrgenes,
        '--proba-gene': augmentation_samples_params.proba_gene,
        '--fold-change-factor': fold_change_factor
    }
    sbatch_params = SbatchParameters(
        conda_env_path=experiment_specs.technical_specs.conda_env_path,
        code_path=experiment_specs.technical_specs.paths.code,
        sbatch_output_file=cache_dir / f'sbatch_{cloud_name}.out',
        sbatch_error_file=cache_dir / f'sbatch_{cloud_name}.err',
        cpu_cores=experiment_specs.technical_specs.augmentation_resources.cpu_cores,
        ram_mem=experiment_specs.technical_specs.augmentation_resources.ram_gb,
        job_name=f'aug_gen__{job_name}',
        command=f'python {experiment_specs.technical_specs.paths.code / "src" / "cmap_data_augmentation_v1" / "generate_partial_augmentations_v1.py"}',
        command_params=command_param_name_to_value
    )
    sbatch_file_path = cache_dir / f'sbatch_{cloud_name}.sh'
    create_sbatch_file(sbatch_params, sbatch_file_path)
    logger.info(f'Submitting the augmentation samples generation job "{job_name}"')
    prior_jobs = [data_augmentation_db_task.slurm_job_id] if data_augmentation_db_task.slurm_job_id else []
    job_id = submit_to_slurm(sbatch_file_path, prior_jobs)
    SlurmJobInfo(job_id=job_id).dump(sbatch_info_file_path)
    return AugmentationSlurmTask(
        path=cache_dir,
        slurm_job_id=job_id
    )


def create_cache_folder(
        cache_type: str,
        root_cache_dir: Path,
        params: CacheParameters,
        additional_params: dict[str, Any] | None = None
) -> tuple[bool, Path]:
    params_dict = json.loads(params.json())
    if additional_params:
        params_dict.update(additional_params)
    params_json = json.dumps(params_dict, indent=2, sort_keys=True)
    logger.info(f'Creating {cache_type} cache using the parameters: {params_json}')
    if params.alias:
        logger.info(f'{cache_type} cache defined the alias "{params.alias}", so will use that')
        cache_name = params.alias
    else:
        cache_name = hashlib.md5(params_json.encode()).hexdigest()
        logger.info(f'{cache_type} params has not defined an alias, so the hash is: {cache_name}')
    cache_dir = root_cache_dir / cache_name
    already_exists = cache_dir.exists()
    if already_exists:
        logger.info(f'{cache_type} cache "{cache_name}" already exists')
    else:
        logger.info(f'Creating new {cache_type} cache dir for: "{cache_dir}"')
        create_dir_if_not_exists(cache_dir)
        with open(cache_dir / 'params.json', 'w') as f:
            f.write(params_json)
    return already_exists, cache_dir


def submit_to_slurm(sbatch_file: Path, job_ids: list[str] = None) -> str:
    """
    Submits a job using 'sbatch' with dependencies on a list of job IDs,
    ensuring they complete successfully before starting,
    and extracts the job ID.

    Args:
        sbatch_file (str): The path to the SBATCH file.
        job_ids (list): Optional list of job IDs to create the dependency string.

    Returns:
        str: The extracted job ID.
    """
    if job_ids:
        process_args = ['sbatch', '--dependency=afterok:' + ':'.join(job_ids), sbatch_file]
    else:
        process_args = ['sbatch', sbatch_file]
    result = process_runner(process_args, capture_output=True, text=True)
    # Example output from 'sbatch': 'Submitted batch job 3895712'
    output_line = result.stdout.strip()
    logger.debug(f'Output line while submitting "{process_args}" is "{output_line}"')
    job_id = output_line.split()[-1]  # Extract the last word as the job ID
    logger.info(f'Running slurm sbatch job with args {process_args}, job_id: "{job_id}"')
    return job_id


def create_clouds_to_augment(
        augmented_perturbations: Collection[Perturbation] | None,
        left_out_cloud: CmapCloudRef,
        cmap: RawCmapDataset) -> list[str]:
    res = []
    if not augmented_perturbations:
        return res
    for augmented_perturbation in augmented_perturbations:
        for cloud_ref in cmap.perturbation_to_cloud_refs[augmented_perturbation]:
            if cloud_ref == left_out_cloud:
                continue
            res.extend([cloud_ref.tissue_code, cloud_ref.perturbation])
    return res


def run_seeds(specs: ExperimentSpecs) -> Generator[int, None, None]:
    return (i for i in range(1, specs.left_one_out_specs.number_of_seeds + 1))


def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    logger.info(f'Creating a new experiment')
    profile_path = get_profile_path()

    with open(profile_path, 'r') as file:
        data = yaml.safe_load(file)
    experiment_specs = ExperimentSpecs(**data)
    logger.info(f'CODE VERSION ==> {experiment_specs.technical_specs.paths.code}')
    assert experiment_specs.api_ver >= SUPPORTED_API_VER, 'This file is too old'
    experiment_paths = create_base_file_system_structure(experiment_specs.technical_specs.paths.root, experiment_specs.name.name)

    logger.info(f'Experiment specs file: "{profile_path}"')
    logger.info(f'Experiment specs:\n{experiment_specs.dict()}')

    logger.info(f'Creating CMAPs for the runs...')
    cmap_alias_to_seed_to_organized_cmap: dict[str, dict[int, OrganizedCmap]] = defaultdict(dict)
    for cmap_params in experiment_specs.cmap_organization_parameters:
        logger.info(f'Creating CMAP {cmap_params.alias} ==>')
        for seed in run_seeds(experiment_specs):
            _cmap_params = copy.deepcopy(cmap_params)
            _cmap_params.random_seed = seed
            organized_cmap = create_cmap(_cmap_params, experiment_paths)
            cmap_alias_to_seed_to_organized_cmap[cmap_params.alias][seed] = organized_cmap

    logger.info(f'='*50)

    left_out_run_to_seed_to_augmentation_name_to_samples_tasks: dict[
        LeftOneOutRun, dict[int, dict[str, list[AugmentationSlurmTask]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if experiment_specs.augmentation_parameters.samples_generation:
        cmap_alias_to_seed_to_augmentation_db_task: dict[str, dict[int, AugmentationSlurmTask]] = defaultdict(dict)
        logger.info(f'Creating augmentations DBs...')
        for cmap_alias, seed_to_organized_cmap in cmap_alias_to_seed_to_organized_cmap.items():
            for seed, organized_cmap in seed_to_organized_cmap.items():
                logger.info(f'Creating augmentations DB for CMAP "{organized_cmap.params.alias}" and seed {seed}')
                augmentation_db_task = create_augmentation_db(
                    experiment_specs,
                    experiment_paths,
                    organized_cmap
                )
                cmap_alias_to_seed_to_augmentation_db_task[cmap_alias][seed] = augmentation_db_task

        logger.info(f'=' * 50)

        logger.info(f'Creating augmentations samples...')
        for left_out_experiment in experiment_specs.left_one_out_specs.clouds:
            left_out_cloud = CmapCloudRef(left_out_experiment.left_out[0], left_out_experiment.left_out[1])
            for seed in run_seeds(experiment_specs):
                organized_cmap = cmap_alias_to_seed_to_organized_cmap[left_out_experiment.cmap][seed]
                for perturbation in left_out_experiment.augmented_perturbations:
                    for cloud_ref in organized_cmap.cmap.perturbation_to_cloud_refs[perturbation]:
                        if left_out_cloud == cloud_ref:
                            continue
                        for augmentation_generation_params in experiment_specs.augmentation_parameters.samples_generation:
                            augmentation_db_task = cmap_alias_to_seed_to_augmentation_db_task[left_out_experiment.cmap][seed]
                            augmentation_samples_task = create_augmentation_samples(
                                experiment_specs=experiment_specs,
                                augmentation_generation_params=augmentation_generation_params,
                                fold_change_factor=left_out_experiment.fold_change_factor,
                                data_augmentation_db_task=augmentation_db_task,
                                organized_cmap=organized_cmap,
                                paths=experiment_paths,
                                cloud_ref=cloud_ref
                            )
                            logger.info(f'Submited Slurm job "{augmentation_samples_task.slurm_job_id}" to generate augmented samples "{augmentation_samples_task.path}"')
                            left_out_run_to_seed_to_augmentation_name_to_samples_tasks[left_out_experiment][seed][augmentation_generation_params.alias].append(augmentation_samples_task)

    logger.info(f'=' * 50)

    logger.info(f'Creating training jobs...')
    all_augmentation_alias = [aug.alias for aug in experiment_specs.augmentation_parameters.samples_generation]
    for repeat in range(1, experiment_specs.left_one_out_specs.number_of_repeats + 1):
        for seed in run_seeds(experiment_specs):
            for left_out_experiment in experiment_specs.left_one_out_specs.clouds:
                left_out_cloud = CmapCloudRef(left_out_experiment.left_out[0], left_out_experiment.left_out[1])
                organized_cmap = cmap_alias_to_seed_to_organized_cmap[left_out_experiment.cmap][seed]
                training_parameters = {
                    **experiment_specs.training_parameters,
                    'left_out_cloud': left_out_experiment.left_out,
                    'cross_validation_clouds': left_out_experiment.cv,
                    'augmentation_params': []  # <-- Fill later
                }
                required_augmentation_job_ids = []
                if left_out_experiment.augmented_perturbations:
                    augmentation_alias_to_samples_tasks = {augmentation_alias:
                                                               left_out_run_to_seed_to_augmentation_name_to_samples_tasks[
                                                                   left_out_experiment][seed][augmentation_alias] for
                                                           augmentation_alias in all_augmentation_alias}
                    required_augmentation_job_ids.extend(
                        [task.slurm_job_id for tasks in augmentation_alias_to_samples_tasks.values() for task in tasks])

                    training_augmentation_params = []
                    for augmentation_alias, augmentation_samples_tasks in augmentation_alias_to_samples_tasks.items():
                        all_tasks_paths = {task.path for task in augmentation_samples_tasks}
                        assert len(all_tasks_paths) == 1
                        training_augmentation_params.append({
                            'alias': augmentation_alias,
                            'augmentation_path': str(list(all_tasks_paths)[0]),
                            'augmentation_rate': experiment_specs.augmentation_parameters.augmentation_rate,
                            'prob': 1 / len(experiment_specs.augmentation_parameters.samples_generation),
                            'clouds_to_augment': create_clouds_to_augment(
                                left_out_experiment.augmented_perturbations,
                                left_out_cloud,
                                organized_cmap.cmap
                            )
                        })
                    training_parameters['augmentation_params'] = training_augmentation_params

                experiment_name = (f'{left_out_cloud.tissue_code}'
                                   f'_{left_out_cloud.perturbation}'
                                   f'_seed_{seed}'
                                   f'_repeat_{repeat}'
                                   f'_{experiment_specs.left_one_out_specs.number_of_repeats}'
                                   f'__exp_name__{experiment_specs.name}'
                                   f'__cv__{"_".join(left_out_experiment.cv)}'
                                   f'__cmap__{organized_cmap.path.name}'
                                   f'__aug__{len(left_out_experiment.augmented_perturbations) > 0}'
                                   f'__fold_change__{left_out_experiment.fold_change_factor if left_out_experiment.augmented_perturbations else 0:.2f}'
                                   )
                logger.info(f'Creating sbatch file for "{experiment_name}"')
                command_param_name_to_value = {
                    '--job-prefix': experiment_name,
                    '--output-folder': experiment_paths.results_dir,
                    '--organized-cmap-folder': organized_cmap.path,
                    '--override-run-parameters': json.dumps(training_parameters).replace('"', "'")
                }
                sbatch_params = SbatchParameters(
                    conda_env_path=experiment_specs.technical_specs.conda_env_path,
                    code_path=experiment_specs.technical_specs.paths.code,
                    sbatch_output_file=experiment_paths.experiment_sbatch_outputs_dir / f'{experiment_name}.out',
                    sbatch_error_file=experiment_paths.experiment_sbatch_outputs_dir / f'{experiment_name}.err',
                    cpu_cores=experiment_specs.technical_specs.training_resources.cpu_cores,
                    ram_mem=experiment_specs.technical_specs.training_resources.ram_gb,
                    job_name=experiment_name,
                    command=f'python {experiment_specs.technical_specs.paths.code / "main.py"} train',
                    command_params=command_param_name_to_value,
                    gpu_type=experiment_specs.technical_specs.use_gpu
                )
                sbatch_file_path = experiment_paths.sbatch_files_dir / experiment_name
                create_sbatch_file(sbatch_params, sbatch_file_path)
                job_id = submit_to_slurm(sbatch_file_path, required_augmentation_job_ids)
                logger.info(f'Submitting the experiment "{experiment_name}", slurm job id is "{job_id}"')

    logger.info(f'All is done!')
    return


@dataclass
class ProcessRunnerMock:
    job_id: int = 0

    @property
    def stdout(self):
        return f'Submitted batch job {self.job_id}'

    def process_runner(self, args, **kwargs):
        logger.info(f'======== running external process with args {args}')
        self.job_id += 1
        return self


if __name__ == '__main__':
    LSAGNE_TEST_MODE = os.getenv('LSAGNE_TEST', 'false').lower() == 'true'
    process_runner = subprocess.run
    # use this to test the script on local machine
    if LSAGNE_TEST_MODE:
        process_runner = ProcessRunnerMock().process_runner
    main()
