import subprocess
import string
import os
import argparse
from pathlib import Path

import pandas as pd

# Change this consts if needed
from src.configuration import config


NAME = 'emil'

# Don't touch this consts
# RUN_STRING_CONSTS = ['SLURM_OUTPUT_FOLDER', 'CODE_PATH', 'OUTPUT_CONST', 'ERROR_CONST', 'TUMOR_CONST', 'PERT_CONST',
#                      'START_CONST', 'END_CONST', 'TEST_NUMBER_CONST', 'PYTHON_OUTPUT_FOLDER', 'ORGANIZED_FOLDER']

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
srun python CODE_PATH/main.py train -tissue-code "TUMOR_CONST" -perturbation "PERT_CONST" -repeat-index-start "START_CONST" -num-of-repeats "NUM_OF_REPEATS_CONST" -output-folder "PYTHON_OUTPUT_FOLDER" -root-organized-folder "ROOT_ORGANIZED_FOLDER" -organized-folder-name "ORGANIZED_FOLDER_NAME" """


# POST_STRING_CONSTS = ['SLURM_OUTPUT_FOLDER', 'CODE_PATH', 'OUTPUT_CONST', 'ERROR_CONST', 'TUMOR_CONST', 'PERT_CONST',
#                       'START_CONST', 'END_CONST', 'TEST_NUMBER_CONST', 'PYTHON_OUTPUT_FOLDER', 'ORGANIZED_FOLDER']
# POST_SBATCH_TEMPLATE = """#!/bin/bash
# #SBATCH --partition=work
# #SBATCH --output="SLURM_OUTPUT_FOLDER/OUTPUT_CONST"
# #SBATCH --error="SLURM_OUTPUT_FOLDER/ERROR_CONST"
# #SBATCH --ntasks=1                   # Run a single task
# #SBATCH --gpus-per-task=1
# #SBATCH --cpus-per-task=8            # Number of CPU cores per task
# #SBATCH --mem=16GB
# conda activate lsagne-pytorch
# srun python CODE_PATH/main.py --post  -tumor "TUMOR_CONST" -pert "PERT_CONST" -start START_CONST -end END_CONST -test-num "TEST_NUMBER_CONST" -output "PYTHON_OUTPUT_FOLDER" -organized-folder "ORGANIZED_FOLDER" --confusion-table --semi-supervised """

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
    # trainer.add_argument('-root-organized-folder', default=config.root_organized_cmap_folder)
    trainer.add_argument('-organized-folder-name', default=config.organized_folder_name)
    trainer.add_argument('-unique-clouds-file-name', default=config.slurm_unique_clouds_file_name)
    trainer.add_argument('-code-version', default=config.version)
    trainer.add_argument('-repeat-index-start', help='Start replay (included)', type=int, default=0)
    trainer.add_argument('-num-of-repeats', help='End replay (not include)', type=int, default=1)
    trainer.add_argument('-one-job', help='If set, run each test cloud repeats in one job', action='store_true')

    # parser.add_argument('-run', help='if set, run normal test', action='store_true')
    # parser.add_argument('-post', help='if set, run post results', action='store_true')
    # parser.add_argument('-num', help='Number of output folder', dest='number', required=True)
    # parser.add_argument('-one-pert', help='If set, run or do post tests with one pert only', dest="one_pert", action='store_true')
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

def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    args = parse_arguments()
    HOME_PATH = os.path.join('/RG/compbio/groupData', NAME)
    CODE_PATH = os.path.join(HOME_PATH, 'code', args.code_version)
    RESULTS_FOLDER = os.path.join(HOME_PATH, 'results')
    RUN_SLURM_OUTPUT_FOLDER = os.path.join(HOME_PATH, 'run_slurm_output', args.code_version)
    POST_SLURM_OUTPUT_FOLDER = os.path.join(RESULTS_FOLDER, 'post_slurm_output', args.code_version)
    RUN_SBATCH_PATH = os.path.join(HOME_PATH, 'run_sbatch_files', args.code_version)
    POST_SBATCH_PATH =  os.path.join(RESULTS_FOLDER, 'post_sbatch_files', args.code_version)
    PYTHON_OUTPUT_FOLDER = os.path.join(RESULTS_FOLDER, 'python_results', args.code_version)
    ROOT_ORGANIZED_FOLDER = os.path.join(HOME_PATH, 'organized_data')
    ORGANIZED_FOLDER_NAME = args.organized_folder_name
    CLOUDS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.unique_clouds_file_name)

    unique_clouds_df = pd.read_csv(CLOUDS_PATH)
    for i in range(unique_clouds_df.shape[0]):
        row = unique_clouds_df.iloc[i]
        tumor_code = str(row['tumor_code'])
        pert = str(row['perturbation'])
        test_name = _file_name_escaping(config.version + '_' + tumor_code + '_' + pert + '_' + str(args.repeat_index_start) + '_' + str(args.num_of_repeats))
        out_file = test_name + '.out'
        err_file = test_name + '.err'
        if args.command == 'train':
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
                        'JOB_NAME': test_name
                    })
                sbatch_text += "\n"
                # print(sbatch_text)
                run_sbatch(sbatch_text, RUN_SBATCH_PATH, test_name)
            else:
                for j in range(args.start, args.end):
                    new_out_file = str(j) + '_' + out_file
                    new_err_file = str(j) + '_' + err_file
                    new_test_name = str(j) + '_' + test_name
                    sbatch_text = replace_in_string(RUN_SBATCH_TEMPLATE, RUN_STRING_CONSTS,
                                                    [RUN_SLURM_OUTPUT_FOLDER, CODE_PATH, new_out_file, new_err_file,
                                                     tumor_code, pert, str(j), str(j+1), str(i), PYTHON_OUTPUT_FOLDER,
                                                     ORGANIZED_FOLDER])
                    if args.one_pert:
                        sbatch_text += '-whitelist-pert "{}"\n'.format(pert)
                    else:
                        sbatch_text += "\n"

                    run_sbatch(sbatch_text, RUN_SBATCH_PATH, new_test_name)
        elif args.post:
            if args.one_job:
                sbatch_text = replace_in_string(POST_SBATCH_TEMPLATE, POST_STRING_CONSTS,
                                                [RUN_SLURM_OUTPUT_FOLDER, CODE_PATH, out_file, err_file, tumor,
                                                 pert, str(args.start), str(args.end), str(i), PYTHON_OUTPUT_FOLDER,
                                                 ORGANIZED_FOLDER])
                if args.one_pert:
                    sbatch_text += '-whitelist-pert "{}"\n'.format(pert)
                else:
                    sbatch_text += "\n"

                run_sbatch(sbatch_text, POST_SBATCH_PATH, test_name)
            else:
                for j in range(args.start, args.end):
                    new_out_file = str(j) + '_' + out_file
                    new_err_file = str(j) + '_' + err_file
                    new_test_name = str(j) + '_' + test_name
                    sbatch_text = replace_in_string(POST_SBATCH_TEMPLATE, POST_STRING_CONSTS,
                                                    [POST_SLURM_OUTPUT_FOLDER, CODE_PATH, new_out_file, new_err_file,
                                                     tumor, pert, str(j), str(j+1), str(i), PYTHON_OUTPUT_FOLDER,
                                                     ORGANIZED_FOLDER])
                    if args.one_pert:
                        sbatch_text += '-whitelist-pert "{}"\n'.format(pert)
                    else:
                        sbatch_text += "\n"
                    run_sbatch(sbatch_text, POST_SBATCH_PATH, new_test_name)
        else:
            print("ERROR: have to set one run option")


if __name__ == '__main__':
    main()
