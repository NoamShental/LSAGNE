import subprocess
import string
import os
import argparse
from pathlib import Path

import pandas as pd

# Change this consts if needed
from src.configuration import config


NAME = 'emil'

# #SBATCH --exclude=gpu[1-4] - use when need specific machines (torch needs 11+ version on cuda)
RUN_SBATCH_TEMPLATE = """#!/bin/bash
### Resource allocation
#SBATCH --partition=work
#SBATCH --output="SLURM_OUTPUT_FOLDER/OUTPUT_CONST"
#SBATCH --error="SLURM_OUTPUT_FOLDER/ERROR_CONST"
#SBATCH --ntasks=1                   # Run a single task	
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=10GB
#SBATCH --qos=gpu
#SBATCH --job-name="JOB_NAME"

### Modules
module load anaconda3

### Runtime
source activate lsagnev4
export PREFECT__FLOWS__CHECKPOINTING=true
srun python CODE_PATH/main.py train -tissue-code "TUMOR_CONST" -perturbation "PERT_CONST" -repeat-index-start "START_CONST" -num-of-repeats "NUM_OF_REPEATS_CONST" -output-folder "PYTHON_OUTPUT_FOLDER" -root-organized-folder "ROOT_ORGANIZED_FOLDER" -organized-folder-name "ORGANIZED_FOLDER_NAME" -override-run-parameters "OVERRIDE_RUN_PARAMETERS" -job-prefix "JOB_PREFIX" """


def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(prog='LSAGNE')
    subparsers = parser.add_subparsers(dest='command')

    organizer = subparsers.add_parser('organize')
    organizer.add_argument('-raw-data-folder', action='store_const', const=config.raw_data_folder)
    organizer.add_argument('-organized-folder', action='store_const', const=config.organized_cmap_folder)

    trainer = subparsers.add_parser('train')
    trainer.add_argument('-perturbation', help='perturbation name', default=config.left_perturbation_name)
    trainer.add_argument('-tissue-code', help='tissue code', default=config.left_tissue_code)
    trainer.add_argument('-output-folder', default=config.output_folder_path)
    trainer.add_argument('-organized-folder-name', default=config.organized_folder_name)
    trainer.add_argument('-unique-clouds-file-name', default=config.slurm_unique_clouds_file_name)
    trainer.add_argument('-code-version', default=config.version)
    trainer.add_argument('-repeat-index-start', help='Start replay (included)', type=int, default=0)
    trainer.add_argument('-num-of-repeats', help='End replay (not include)', type=int, default=1)
    trainer.add_argument('-one-job', help='If set, run each test cloud repeats in one job', action='store_true')

    return parser.parse_args()


def _file_name_escaping(filename):
    """
    Make escaping for file names (i.e: omit '|' or '\'.
    :param filename: filename to escape
    :return: escaped filename
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


def replace_in_string(s, placeholder_to_string):
    """
    Replace list of strings in another list of strings, in the same size
    :param s: original string
    :param current_strings: list of strings to replace
    :param replace_strings: replacement strings
    :return: original string, with replacements
    """
    for placeholder, replace_string in placeholder_to_string.items():
        s = s.replace(placeholder, replace_string)
    return s


def run_sbatch(sbatch_content, sbatch_folder, test_name):
    """
    Write sbatch to file and run it
    :param sbatch_content: content of sbatch
    :param sbatch_folder: folder to put the sbatch
    :param test_name:name of file
    """
    sbatch_path = os.path.join(sbatch_folder, test_name)
    print(f'Writing sbatch file to "{sbatch_path}".')
    Path(sbatch_folder).mkdir(parents=True, exist_ok=True)
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)

    # command = 'sbatch "{}"'.format(sbatch_path)
    # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    # process.wait()


def _create_run_options(prefix: str, path: str) -> pd.DataFrame:
    options = []
    # list(range(1,10)) + list(range(10,101, 10)) + list(range(150,1001, 50)) + list(range(1000,2001, 100)) + list(range(2000,5001, 250))
    # for clouds_classifier in [200.0, 200_000.0]:
    #     for tissues_classifier in [0.0]:
    #         for distance_from_cloud_center in [0.0]:
    #             for vae_kld in [3.0, 1.0, 0.1, 0.001, 0.0001]:
    #                 for collinearity in [0.0]:
    #                     for different_directions in [0.0]:
    #                         for treatment_and_drug_vectors_distance in [0.0]:
    #                             for n_epochs in [1000, 2500]:
    #                                 for lr in [3e-3, 1e-3, 1e-4]:
    #                                     options.append({
    #                                         'lr': lr,
    #                                         'n_epochs': n_epochs,
    #                                         'loss_coef':{
    #                                             'clouds_classifier': clouds_classifier,
    #                                             'tissues_classifier': tissues_classifier,
    #                                             'distance_from_cloud_center': distance_from_cloud_center,
    #                                             'vae_kld': vae_kld,
    #                                             'treatment_vectors_collinearity': collinearity,
    #                                             'drug_vectors_collinearity': collinearity,
    #                                             'different_directions': different_directions,
    #                                             'treatment_and_drug_vectors_distance': treatment_and_drug_vectors_distance,
    #                                         },
    #                                         'warmup_reference_points_loss_coef': {
    #                                             'clouds_classifier': clouds_classifier,
    #                                             'tissues_classifier': 0.0,
    #                                             'distance_from_cloud_center': distance_from_cloud_center,
    #                                             'vae_kld': vae_kld,
    #                                             'treatment_vectors_collinearity': 0.0,
    #                                             'drug_vectors_collinearity': 0.0,
    #                                             'different_directions': 0.0,
    #                                             'treatment_and_drug_vectors_distance': 0.0,
    #                                         }
    #                                     })

    # for clouds_classifier, distance_from_cloud_center, vae_kld in [
    #     # (270, 0, 3),
    #     # (270, 1, 5),
    #     # (270, 3, 5),
    #     # (270, 9, 9),
    #     # (270, 9, 5),
    #     # (270, 27, 9),
    #     # (270, 81, 3),
    #     # (270, 81, 1),
    #     # (810, 0, 3),
    #     # (810, 3, 5),
    #     # (810, 3, 3),
    #     # (810, 81, 5),
    #     # (810, 81, 3),
    #     (810, 81, 1),
    #     # (810, 240, 9),
    #     # (810, 240, 5),
    #     # (810, 240, 3),
    #     # (810, 240, 1),
    #
    #     # (270, 81, 3)
    # ]:
    for clouds_classifier, vae_kld in [(810, 1)]:
        for distance_from_cloud_center in [81, 150, 300, 700, 1000]:
            for treatment_vectors_collinearity, drug_vectors_collinearity, treatment_and_drug_vectors_distance in [
                (150, 100, 81),
                (800, 400, 500),
                (400, 200, 200),
                (200, 100, 81),
                (600, 300, 150),
                (800, 400, 150),
                (150, 100, 500),
            ]:
                for tissues_classifier in [0.0]:
                    for different_directions in [0.0, 1.0, 2.0]:
                        for n_epochs in [3000]:
                            for lr in [3e-3]:
                                for random_seed in [17417]:
                                    options.append({
                                        'lr': lr,
                                        'n_epochs': n_epochs,
                                        'loss_coef':{
                                            'clouds_classifier': clouds_classifier,
                                            'tissues_classifier': tissues_classifier,
                                            'distance_from_cloud_center': distance_from_cloud_center,
                                            'vae_kld': vae_kld,
                                            'treatment_vectors_collinearity_using_batch_treated': treatment_vectors_collinearity,
                                            'treatment_vectors_collinearity_using_batch_control': treatment_vectors_collinearity,
                                            'drug_vectors_collinearity_using_batch_treated': drug_vectors_collinearity,
                                            'drug_vectors_collinearity_using_batch_control': drug_vectors_collinearity,
                                            'treatment_vectors_different_directions_using_anchors': different_directions,
                                            'drug_and_treatment_vectors_different_directions_using_anchors': different_directions,
                                            'treatment_and_drug_vectors_distance': treatment_and_drug_vectors_distance,
                                        },
                                        'warmup_reference_points_loss_coef': {
                                            'clouds_classifier': clouds_classifier,
                                            'tissues_classifier': 0.0,
                                            'distance_from_cloud_center': distance_from_cloud_center,
                                            'vae_kld': vae_kld,
                                            'treatment_vectors_collinearity_using_batch_treated': 0.0,
                                            'treatment_vectors_collinearity_using_batch_control': 0.0,
                                            'drug_vectors_collinearity_using_batch_treated': 0.0,
                                            'drug_vectors_collinearity_using_batch_control': 0.0,
                                            'treatment_vectors_different_directions_using_anchors': 0.0,
                                            'drug_and_treatment_vectors_different_directions_using_anchors': 0.0,
                                            'treatment_and_drug_vectors_distance': 0.0,
                                        },
                                        'random_seed': random_seed
                                    })

    df = pd.DataFrame(options)
    df.to_csv(os.path.join(path, f'{prefix}_run_options.csv'))
    return df



def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    args = parse_arguments()
    HOME_PATH = os.path.join('/RG/compbio/groupData', NAME)
    CODE_PATH = os.path.join(HOME_PATH, 'code', args.code_version)
    RESULTS_FOLDER = os.path.join(HOME_PATH, 'results')
    RUN_SLURM_OUTPUT_FOLDER = os.path.join(HOME_PATH, 'run_slurm_output', args.code_version)
    RUN_SBATCH_PATH = os.path.join(HOME_PATH, 'run_sbatch_files', args.code_version)
    PYTHON_OUTPUT_FOLDER = os.path.join(RESULTS_FOLDER, 'python_results', args.code_version)
    ROOT_ORGANIZED_FOLDER = os.path.join(HOME_PATH, 'organized_data')
    ORGANIZED_FOLDER_NAME = args.organized_folder_name
    CLOUDS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.unique_clouds_file_name)

    unique_clouds_df = pd.read_csv(CLOUDS_PATH)
    for unique_clouds_i, unique_cloud in unique_clouds_df.iterrows():
        tumor_code = str(unique_cloud['tumor_code'])
        pert = str(unique_cloud['perturbation'])
        test_name = _file_name_escaping(config.version + '_' + tumor_code + '_' + pert + '_' + str(args.repeat_index_start) + '_' + str(args.num_of_repeats))
        test_options = _create_run_options(test_name, PYTHON_OUTPUT_FOLDER)
        for test_option_i, option in test_options.iterrows():
            sub_test_name = f'{test_name}__test_option_{test_option_i}'
            out_file = sub_test_name + '.out'
            err_file = sub_test_name + '.err'
            if args.one_job:
                sbatch_text = replace_in_string(
                    RUN_SBATCH_TEMPLATE,
                    {
                        'SLURM_OUTPUT_FOLDER': RUN_SLURM_OUTPUT_FOLDER,
                        'CODE_PATH': CODE_PATH,
                        'OUTPUT_CONST': out_file,
                        'ERROR_CONST': err_file,
                        'TUMOR_CONST': tumor_code,
                        'PERT_CONST': pert,
                        'START_CONST': str(args.repeat_index_start),
                        'NUM_OF_REPEATS_CONST': str(args.num_of_repeats),
                        'PYTHON_OUTPUT_FOLDER': PYTHON_OUTPUT_FOLDER,
                        'ROOT_ORGANIZED_FOLDER': ROOT_ORGANIZED_FOLDER,
                        'ORGANIZED_FOLDER_NAME': ORGANIZED_FOLDER_NAME,
                        'JOB_NAME': sub_test_name,
                        'OVERRIDE_RUN_PARAMETERS': str(option.to_dict()),
                        'JOB_PREFIX': f'test_option_{test_option_i}'
                    })
                sbatch_text += "\n"
                # print(sbatch_text)
                run_sbatch(sbatch_text, RUN_SBATCH_PATH, sub_test_name)
            else:
                print("ERROR: have to set one run option")


if __name__ == '__main__':
    main()
