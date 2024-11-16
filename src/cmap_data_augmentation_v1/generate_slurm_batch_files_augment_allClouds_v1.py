import subprocess
import string
import os
import argparse
from pathlib import Path
import random

import pandas as pd

RUN_SBATCH_TEMPLATE = """#!/bin/bash
### Resource allocation
#SBATCH --partition=work
#SBATCH --output="SLURM_OUTPUT_FOLDER/OUTPUT_CONST"
#SBATCH --error="SLURM_OUTPUT_FOLDER/ERROR_CONST"
#SBATCH --ntasks=1                   # Run a single task	
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=16GB
#SBATCH --job-name="JOB_NAME"

### Modules
#module load anaconda3

### Runtime
source activate lsagnev7
export PREFECT__FLOWS__CHECKPOINTING=true
cd CODE_PATH
srun python generate_partial_augmentations_v1.py TUMOR_CONST PERT_CONST NUM_OF_REPEATS_CONST PYTHON_OUTPUT_FOLDER -job-prefix JOB_PREFIX """


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


def write_sbatch(sbatch_content, sbatch_folder, test_name):
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

def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    version = '0.9.7'
    # HOME_PATH = '/RG/compbio/michaelel/lsagne_v0.9.6/lsagne/'
    # HOME_PATH = 'C:/Work/Noam/CMAP/lsagne_git/lsagne/'
    HOME_PATH = '/Users/emil/MSc/lsagne-1'
    CODE_PATH = os.path.join(HOME_PATH, 'src/cmap_data_augmentation_v1')
    PYTHON_OUTPUT_FOLDER = os.path.join(HOME_PATH, 'organized_data/data_augmentation')
    CLOUDS_PATH = os.path.join(HOME_PATH, 'slurm_runner/unique_clouds_full_11x9.csv')
    RUN_SLURM_OUTPUT_FOLDER = os.path.join(HOME_PATH, 'slurm_results', version)
    RUN_SBATCH_PATH = os.path.join(HOME_PATH, 'run_sbatch_files', version)

    num_of_repeats = 100000

    unique_clouds_df = pd.read_csv(CLOUDS_PATH)
    tumor_codes = list(set([str(unique_cloud['tumor_code']) for unique_clouds_i, unique_cloud in unique_clouds_df.iterrows()]))
    for unique_clouds_i, unique_cloud in unique_clouds_df.iterrows():
        tumor_code = str(unique_cloud['tumor_code'])
        pert = '"' + str(unique_cloud['perturbation']) + '"'
        test_name = _file_name_escaping(version + '_' + tumor_code + '_' + pert.replace(" ","_") + '_' + str(num_of_repeats))
        sub_test_name = f'{test_name}_MEL_AUG_SBTCH'
        out_file = sub_test_name + '.out'
        err_file = sub_test_name + '.err'
        sbatch_text = replace_in_string(
            RUN_SBATCH_TEMPLATE,
            {
                'SLURM_OUTPUT_FOLDER': RUN_SLURM_OUTPUT_FOLDER,
                'CODE_PATH': CODE_PATH,
                'OUTPUT_CONST': out_file,
                'ERROR_CONST': err_file,
                'TUMOR_CONST': tumor_code,
                'PERT_CONST': pert,
                'NUM_OF_REPEATS_CONST': str(num_of_repeats),
                'PYTHON_OUTPUT_FOLDER': PYTHON_OUTPUT_FOLDER+'/PatialAugment_'+test_name+'.pkl',
                'JOB_NAME': sub_test_name,
                'JOB_PREFIX': f'test_{sub_test_name}'
            })
        sbatch_text += "\n"
        write_sbatch(sbatch_text, RUN_SBATCH_PATH, sub_test_name+'.sh')


if __name__ == '__main__':
    main()
