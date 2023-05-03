import subprocess
import string
import os
import argparse
import pandas as pd

# Change this consts if needed
CLOUDS_PATH = 'unique_clouds.csv'
NAME = 'bashan'

# Don't touch this consts
RUN_STRING_CONSTS = ['SLURM_OUTPUT_FOLDER', 'CODE_PATH', 'OUTPUT_CONST', 'ERROR_CONST', 'TUMOR_CONST', 'PERT_CONST',
                     'START_CONST', 'END_CONST', 'TEST_NUMBER_CONST', 'PYTHON_OUTPUT_FOLDER', 'ORGANIZED_FOLDER']
RUN_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --partition=work
#SBATCH --output="SLURM_OUTPUT_FOLDER/OUTPUT_CONST"
#SBATCH --error="SLURM_OUTPUT_FOLDER/ERROR_CONST"
#SBATCH --ntasks=1                   # Run a single task	
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=20GB
srun python CODE_PATH/main.py --run -tumor "TUMOR_CONST" -pert "PERT_CONST" -start "START_CONST" -end "END_CONST" -test-num "TEST_NUMBER_CONST" -output "PYTHON_OUTPUT_FOLDER" -organized-folder "ORGANIZED_FOLDER" """


POST_STRING_CONSTS = ['SLURM_OUTPUT_FOLDER', 'CODE_PATH', 'OUTPUT_CONST', 'ERROR_CONST', 'TUMOR_CONST', 'PERT_CONST',
                      'START_CONST', 'END_CONST', 'TEST_NUMBER_CONST', 'PYTHON_OUTPUT_FOLDER', 'ORGANIZED_FOLDER']
POST_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --partition=work
#SBATCH --output="SLURM_OUTPUT_FOLDER/OUTPUT_CONST"
#SBATCH --error="SLURM_OUTPUT_FOLDER/ERROR_CONST"
#SBATCH --ntasks=1                   # Run a single task	
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=20GB
srun python CODE_PATH/main.py --post  -tumor "TUMOR_CONST" -pert "PERT_CONST" -start START_CONST -end END_CONST -test-num "TEST_NUMBER_CONST" -output "PYTHON_OUTPUT_FOLDER" -organized-folder "ORGANIZED_FOLDER" --confusion-table --semi-supervised """

def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run', help='if set, run normal test', action='store_true')
    parser.add_argument('-post', help='if set, run post results', action='store_true')
    parser.add_argument('-num', help='Number of output folder', dest='number', required=True)
    parser.add_argument('-start', help='Start replay (included)', dest='start', required=True, type=int)
    parser.add_argument('-end', help='End replay (not include)', dest='end', required=True, type=int)
    parser.add_argument('-one-pert', help='If set, run or do post tests with one pert only', dest="one_pert", action='store_true')
    parser.add_argument('-one-job', help='If set, run each test cloud repeats in one job', dest="one_job", action='store_true')
    return parser.parse_args()


def _file_name_escaping(filename):
    """
    Make escaping for file names (i.e: omit '|' or '\'.
    :param filename: filename to escape
    :return: escaped filename
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


def replace_in_string(s, current_strings, replace_strings):
    """
    Replace list of strings in another list of strings, in the same size
    :param s: original string
    :param current_strings: list of strings to replace
    :param replace_strings: replacement strings
    :return: original string, with replacements
    """
    for i in range(len(current_strings)):
        s = s.replace(current_strings[i], replace_strings[i])
    return s


def run_sbatch(sbatch_content, sbatch_folder, test_name):
    """
    Write sbatch to file and run it
    :param sbatch_content: content of sbatch
    :param sbatch_folder: folder to put the sbatch
    :param test_name:name of file
    """
    sbatch_path = os.path.join(sbatch_folder, test_name)
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)

    command = 'sbatch "{}"'.format(sbatch_path)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    args = parse_arguments()
    number = args.number
    HOME_PATH = '/RG/compbio/groupData'
    CODE_PATH = 'HOME_PATH/code/NAME/NUMBER'.replace('HOME_PATH', HOME_PATH).replace('NAME', NAME).replace('NUMBER',
                                                                                                           number)
    RESULTS_FOLDER = 'HOME_PATH/results/NAME/NUMBER'.replace('HOME_PATH', HOME_PATH).replace('NAME', NAME).replace(
        'NUMBER', number)
    RUN_SLURM_OUTPUT_FOLDER = 'RESULTS_FOLDER/run_slurm_output'.replace('RESULTS_FOLDER', RESULTS_FOLDER)
    POST_SLURM_OUTPUT_FOLDER = 'RESULTS_FOLDER/post_slurm_output'.replace('RESULTS_FOLDER', RESULTS_FOLDER)
    RUN_SBATCH_PATH = 'RESULTS_FOLDER/run_sbatch_files'.replace('RESULTS_FOLDER', RESULTS_FOLDER)
    POST_SBATCH_PATH = 'RESULTS_FOLDER/post_sbatch_files'.replace('RESULTS_FOLDER', RESULTS_FOLDER)
    PYTHON_OUTPUT_FOLDER = 'RESULTS_FOLDER/python_results'.replace('RESULTS_FOLDER', RESULTS_FOLDER)
    ORGANIZED_FOLDER = 'HOME_PATH/organized_data/NAME/NUMBER'.replace(
        'HOME_PATH', HOME_PATH).replace('NAME', NAME).replace('NUMBER', number)
    TCGA_FOLDER = 'HOME_PATH/organized_data/TCGA'

    unique_clouds_df = pd.read_csv(CLOUDS_PATH)
    for i in range(unique_clouds_df.shape[0]):
        row = unique_clouds_df.iloc[i]
        tumor = str(row['tumor'])
        pert = str(row['perturbation'])
        test_name = _file_name_escaping(str(i) + '_' + tumor + '_' + pert)
        out_file = test_name + '_out'
        err_file = test_name + '_err'
        if args.run:
            if args.one_job:
                sbatch_text = replace_in_string(RUN_SBATCH_TEMPLATE, RUN_STRING_CONSTS,
                                                [RUN_SLURM_OUTPUT_FOLDER, CODE_PATH, out_file, err_file, tumor,
                                                 pert, str(args.start), str(args.end), str(i), PYTHON_OUTPUT_FOLDER,
                                                 ORGANIZED_FOLDER])
                if args.one_pert:
                    sbatch_text += '-whitelist-pert "{}"\n'.format(pert)
                else:
                    sbatch_text += "\n"

                run_sbatch(sbatch_text, RUN_SBATCH_PATH, test_name)
            else:
                for j in range(args.start, args.end):
                    new_out_file = str(j) + '_' + out_file
                    new_err_file = str(j) + '_' + err_file
                    new_test_name = str(j) + '_' + test_name
                    sbatch_text = replace_in_string(RUN_SBATCH_TEMPLATE, RUN_STRING_CONSTS,
                                                    [RUN_SLURM_OUTPUT_FOLDER, CODE_PATH, new_out_file, new_err_file,
                                                     tumor, pert, str(j), str(j+1), str(i), PYTHON_OUTPUT_FOLDER,
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
