import subprocess
import string
import os
import argparse
from pathlib import Path

import pandas as pd

# Change this consts if needed
from src.configuration import config

## run this file like this: python generate_slurm_batch_files_v2.py train -one-job

RUN_SBATCH_TEMPLATE = """#!/bin/bash
### Resource allocation
#SBATCH --partition=work
#SBATCH --output="SLURM_OUTPUT_FOLDER/OUTPUT_CONST"
#SBATCH --error="SLURM_OUTPUT_FOLDER/ERROR_CONST"
#SBATCH --ntasks=1                   # Run a single task	
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=10GB
#SBATCH --job-name="JOB_NAME"

### Modules
module load anaconda3

### Runtime
source activate lsagnev6
export PREFECT__FLOWS__CHECKPOINTING=true
srun python CODE_PATH/main.py train -tissue-code "TUMOR_CONST" -perturbation "PERT_CONST" -repeat-index-start "START_CONST" -num-of-repeats "NUM_OF_REPEATS_CONST" -use-cuda -output-folder "PYTHON_OUTPUT_FOLDER" -root-organized-folder "ROOT_ORGANIZED_FOLDER" -organized-folder-name "ORGANIZED_FOLDER_NAME" -override-run-parameters "OVERRIDE_RUN_PARAMETERS" -job-prefix "JOB_PREFIX" """


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
    trainer.add_argument('-num-of-repeats', help='End replay (not include)', type=int, default=3)
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

def create_run_option(n_epochs=3000, WRM_DUR=300, RSLCT=0, edim=20, kld=0.1,
                      CLS=1000, CLST=200, COLIN=40000, DD=300, DCC=50, DST=300):
    option= {
        'n_epochs': n_epochs,
        'warmup_reference_points_duration': WRM_DUR,
        'reference_point_reselect_period': RSLCT,
        'embedding_dim': edim,
        'loss_coef':{
            'vae_mse': 1,
            'vae_kld': kld,
            'clouds_classifier': CLS,
            'tissues_classifier': CLST,
            'treatment_vectors_collinearity_using_batch_treated': COLIN,
            'treatment_vectors_collinearity_using_batch_control': COLIN,
            'drug_vectors_collinearity_using_batch_treated': COLIN,
            'drug_vectors_collinearity_using_batch_control': COLIN,
            'treatment_vectors_different_directions_using_anchors': DD,
            'drug_and_treatment_vectors_different_directions_using_anchors': DD,
            'distance_from_cloud_center': DCC,
            'treatment_and_drug_vectors_distance': DST
        },
        'warmup_reference_points_loss_coef': {
            'vae_mse': 1,
            'vae_kld': kld,
            'clouds_classifier': CLS,
            'tissues_classifier': CLST,
            'treatment_vectors_collinearity': 0,
            'drug_vectors_collinearity_using_batch_treated': 0,
            'drug_vectors_collinearity_using_batch_control': 0,
            'treatment_vectors_different_directions_using_anchors': 0,
            'drug_and_treatment_vectors_different_directions_using_anchors': 0,
            'distance_from_cloud_center': 0,
            'treatment_and_drug_vectors_distance': 0
        }
    }
    return option

def option_to_name(option):
    sstr=f'epchs{option.n_epochs}_WRM{option.warmup_reference_points_duration}_RSLCT{option.reference_point_reselect_period}_'
    sstr=sstr+f'dim{option.embedding_dim}_kld{option.loss_coef["vae_kld"]}_CLS{option.loss_coef["clouds_classifier"]}'
    sstr=sstr+f'_CLST{option.loss_coef["tissues_classifier"]}_COLIN{option.loss_coef["treatment_vectors_collinearity"]}_DD{option.loss_coef["treatment_vectors_different_directions_using_anchors"]}_DD{option.loss_coef["drug_and_treatment_vectors_different_directions_using_anchors"]}'
    sstr=sstr+f'_DCC{option.loss_coef["distance_from_cloud_center"]}_DST{option.loss_coef["treatment_and_drug_vectors_distance"]}'
    return sstr

def _create_run_options(prefix: str, path: str) -> pd.DataFrame:
    options = []

#   different distance_from_cloud_center
    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=300;DCC=50;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=300;DCC=100;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=300;DCC=200;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=300;DCC=1000;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    #   some more
    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=4000;DD=300;DCC=50;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=4000;DD=300;DCC=100;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=4000;DD=300;DCC=200;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=4000;DD=300;DCC=1000;DST=300;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    #   some more
    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=100;DCC=50;DST=100;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=100;DCC=100;DST=100;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=100;DCC=200;DST=100;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1000;CLST=200;COLIN=40000;DD=100;DCC=1000;DST=100;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))


    #   test for minimum for each option
    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=1;CLST=0;COLIN=0;DD=0;DCC=0;DST=0;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=0;CLST=1;COLIN=0;DD=0;DCC=0;DST=0;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=0;CLST=0;COLIN=1;DD=0;DCC=0;DST=0;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=0;CLST=0;COLIN=0;DD=1;DCC=0;DST=0;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=0;CLST=0;COLIN=0;DD=0;DCC=1;DST=0;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))

    n_epochs=3000;WRM_DUR=300;RSLCT=0;edim=20;kld=0.1;CLS=0;CLST=0;COLIN=0;DD=0;DCC=0;DST=1;
    options.append(create_run_option(n_epochs,WRM_DUR,RSLCT,edim,kld,CLS,CLST,COLIN,DD,DCC,DST))


    df = pd.DataFrame(options)
    # df.to_csv(os.path.join(path, f'{prefix}_run_options.csv'))
    return df



def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    args = parse_arguments()
    HOME_PATH = '/home/michaelel/lsagne'
    CODE_PATH = HOME_PATH
    RESULTS_FOLDER = os.path.join(HOME_PATH, 'slurm_results')
    RUN_SLURM_OUTPUT_FOLDER = os.path.join(HOME_PATH, 'run_slurm_output', args.code_version)
    RUN_SBATCH_PATH = os.path.join(HOME_PATH, 'run_sbatch_files', args.code_version)
    PYTHON_OUTPUT_FOLDER = os.path.join(RESULTS_FOLDER, 'python_results', args.code_version)
    ROOT_ORGANIZED_FOLDER = os.path.join(HOME_PATH, 'organized_data')
    ORGANIZED_FOLDER_NAME = args.organized_folder_name
    CLOUDS_PATH = os.path.join('/home/michaelel/lsagne/slurm_runner', args.unique_clouds_file_name)

    unique_clouds_df = pd.read_csv(CLOUDS_PATH)
    for unique_clouds_i, unique_cloud in unique_clouds_df.iterrows():
        tumor_code = str(unique_cloud['tumor_code'])
        pert = str(unique_cloud['perturbation'])
        test_name = _file_name_escaping(config.version + '_' + tumor_code + '_' + pert + '_' + str(args.repeat_index_start) + '_' + str(args.num_of_repeats))
        test_options = _create_run_options(test_name, PYTHON_OUTPUT_FOLDER)
        for test_option_i, option in test_options.iterrows():
            sub_test_name = f'{test_name}_MEL_SBTCH_{option_to_name(option)}'
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
                        'JOB_PREFIX': f'test_{test_option_i}_{sub_test_name}'
                    })
                sbatch_text += "\n"
                # print(sbatch_text)
                run_sbatch(sbatch_text, RUN_SBATCH_PATH, sub_test_name+'.sh')
            else:
                print("ERROR: have to set one run option")


if __name__ == '__main__':
    main()
