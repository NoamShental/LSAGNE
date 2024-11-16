from pathlib import Path

from slurm_runner.v2.sbatch_creator import SbatchParameters, create_sbatch_file
from src.os_utilities import create_dir_if_not_exists


def main():
    root = Path('/RG/compbio/emil/augmentation_optimization_root_v5')
    sbatch_files = root / 'sbatch_files'
    sbatch_out_err = root / 'sbatch_out_err'
    results = root / 'results'
    working_dir = root / 'working_dir'
    code_path = Path('/RG/compbio/emil/code/27_7_2024')
    samples_to_augment = 1_000
    hdf_chunk_size = 10_000

    create_dir_if_not_exists(root)
    create_dir_if_not_exists(sbatch_files)
    create_dir_if_not_exists(sbatch_out_err)
    create_dir_if_not_exists(results)
    create_dir_if_not_exists(working_dir)

    for i in range(15):
        random_range = f'{i}-{i + 1}'
        output_metrics_file_location = results / f'res_{random_range}.h5'
        experiment_name = f'exp_{random_range}'

        print(f'Creating sbatch file for "{experiment_name}"')

        sbatch_params = SbatchParameters(
            conda_env_path='/RG/compbio/emil/lsagnev8-test',
            code_path=code_path,
            sbatch_output_file=sbatch_out_err / f'{experiment_name}.out',
            sbatch_error_file=sbatch_out_err / f'{experiment_name}.err',
            cpu_cores=64,
            ram_mem=64,
            job_name=experiment_name,
            command=f'python { code_path / "src/augmentation_parameters_tuning/test_augmentation_parameters.py"} server "{random_range}" "{output_metrics_file_location}" {samples_to_augment} {hdf_chunk_size}',
            command_params={},
            gpu_type=None
        )
        sbatch_file_path = sbatch_files / experiment_name
        create_sbatch_file(sbatch_params, sbatch_file_path)


if __name__ == '__main__':
    main()
