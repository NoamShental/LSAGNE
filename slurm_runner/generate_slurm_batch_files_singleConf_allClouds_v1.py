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
#SBATCH --mem=64GB
#SBATCH --job-name="JOB_NAME"

### Modules
#module load Anaconda3/2022.05

### Runtime
source activate /RG/compbio/emil/lsagnev8-test
export PREFECT__FLOWS__CHECKPOINTING=true
export PYTHONUNBUFFERED=1

srun python CODE_PATH/main.py train -repeat-index-start "START_CONST" -num-of-repeats "NUM_OF_REPEATS_CONST" -output-folder "PYTHON_OUTPUT_FOLDER" -root-organized-folder "ROOT_ORGANIZED_FOLDER" -organized-folder-name "ORGANIZED_FOLDER_NAME" -override-run-parameters "OVERRIDE_RUN_PARAMETERS" -job-prefix "JOB_PREFIX" """


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

def option_to_name(option):
    sstr=f'epchs{option.n_epochs}_WRM{option.warmup_reference_points_duration}_RSLCT{option.reference_point_reselect_period}_'
    sstr=sstr+f'dim{option.embedding_dim}_kld{option.loss_coef["vae_kld"]}_CLS{option.loss_coef["clouds_classifier"]}'
    # sstr=sstr+f'_CLST{option.loss_coef["tissues_classifier"]}_COLIN-T-C{option.loss_coef["treatment_vectors_collinearity_using_batch_control"]}_COLIN-T-T{option.loss_coef["treatment_vectors_collinearity_using_batch_treated"]}_COLIN-D-T{option.loss_coef["drug_vectors_collinearity_using_batch_treated"]}_COLIN-D-C{option.loss_coef["drug_vectors_collinearity_using_batch_control"]}_TD{option.loss_coef["treatment_vectors_different_directions"]}_DD{option.loss_coef["drug_and_treatment_vectors_different_directions"]}_T12D{option.loss_coef["treatment_and_drug_vectors_distance_p1_p2_loss"]}_T12P{option.loss_coef["treatment_and_drug_vectors_distance_p1_p2_to_treated_loss"]}'
    # sstr=sstr+f'_DCC6{option.loss_coef["distance_from_cloud_center_6h"]}_DCC24{option.loss_coef["distance_from_cloud_center_24h"]}'
    return sstr

def create_run_options(prefix: str, path: str, CV_TISSUE:str , CV_DRUG:str, TUMOR_CONST:str, PERT_CONST:str) -> pd.DataFrame:
    options = []
    #single option
    options.append({                                          #############
        'left_out_cloud':[TUMOR_CONST,PERT_CONST],
        'cross_validation_clouds': [CV_TISSUE,CV_DRUG],
        'perturbations_equivalence_sets': [['isonicotinohydroxamic-acid','raloxifene'],['vorinostat','trichostatin-a']], 'perturbations_equivalence_losses_coefs':{'clouds_classifier': 0,'treatment_vectors_different_directions_using_anchors': 0,'treatment_vectors_different_directions_using_batch': 0},
        'clouds_to_augment': [],
        'partial_cloud_training': [],
        'n_epochs': 5000,                                     #
        'lr':0.0001,
        'warmup_reference_points_duration': 300,
        'reference_point_reselect_period': 4300,
        'embedding_dim': 20,
        'max_radius': 0.1,
        'loss_coef':{
            'clouds_classifier': 5000,
            'tissues_classifier': 5000,
            'distance_from_cloud_center_6h': 10000,
            'distance_from_cloud_center_dmso_24h': 10000,
            'distance_from_cloud_center_24h_without_dmso_24h': 10000,
            'max_radius_limiter': 10000,
            'vae_kld': 0.01,
            'treatment_vectors_collinearity_using_batch_treated': 10000,
            'treatment_vectors_collinearity_using_batch_control': 10000,
            'drug_vectors_collinearity_using_batch_treated': 10000,
            'drug_vectors_collinearity_using_batch_control': 10000,
            'treatment_vectors_different_directions_using_batch': 10000,
            'drug_and_treatment_vectors_different_directions_using_batch': 10000,
            'treatment_and_drug_vectors_distance_p1_p2_loss': 10000,
            'treatment_and_drug_vectors_distance_p1_p2_to_treated_loss': 10000
        },
        'warmup_reference_points_loss_coef': {
            'clouds_classifier': 5000,
            'tissues_classifier': 5000,
            'distance_from_cloud_center_6h': 0,
            'distance_from_cloud_center_dmso_24h': 0,
            'distance_from_cloud_center_24h_without_dmso_24h': 0,
            'max_radius_limiter': 0,
            'vae_kld': 0.01,
            'treatment_vectors_collinearity_using_batch_treated': 0,
            'treatment_vectors_collinearity_using_batch_control': 0,
            'drug_vectors_collinearity_using_batch_treated': 0,
            'drug_vectors_collinearity_using_batch_control': 0,
            'treatment_vectors_different_directions_using_batch': 0,
            'drug_and_treatment_vectors_different_directions_using_batch': 0,
            'treatment_and_drug_vectors_distance_p1_p2_loss': 0,
            'treatment_and_drug_vectors_distance_p1_p2_to_treated_loss': 0
        },
        'trim_treated_clouds_ratio_to_keep':0.85, 
        'trim_untreated_clouds_and_time_24h_ratio_to_keep':0.5, 
        'clouds_trimming_epochs':[1000]
    })
    # for clouds_classifier in [500,1000, 2000, 4000]:                                                      # 4
        # for tissues_classifier in [2000]:                                                                 # 1
            # for distance_from_cloud_center in [200, 500,1000, 2000, 4000]:                                # 5
                # for vae_kld in [0.1,0.01]:                                                                # 2
                    # for collinearity in [500, 1000, 2000, 4000, 10000]:                                   # 5
                        # for different_directions in [500, 1000, 2000]:                                    # 3
                            # for treatment_and_drug_vectors_distance in [200, 500, 1000]:                  # 3
                                # for distance_from_drug_vector_predicted in [500, 1000, 2000, 4000,10000]: # 5
                                    # for n_epochs in [3000]:                                               # 1
                                        # for RSLCT in [0,300]:                                             # 3
                                                # options.append({                                          #############
                                                # 'n_epochs': n_epochs,                                     # 
                                                # 'warmup_reference_points_duration': 300,
                                                # 'reference_point_reselect_period': RSLCT,
                                                # 'embedding_dim': 20,
                                                # 'loss_coef':{
                                                    # 'clouds_classifier': clouds_classifier,
                                                    # 'tissues_classifier': tissues_classifier,
                                                    # 'distance_from_cloud_center': distance_from_cloud_center,
                                                    # 'vae_kld': vae_kld,
                                                    # 'treatment_vectors_collinearity': collinearity/10,
                                                    # 'drug_vectors_collinearity': collinearity,
                                                    # 'different_directions': different_directions,
                                                    # 'treatment_and_drug_vectors_distance': treatment_and_drug_vectors_distance,
                                                # },
                                                # 'warmup_reference_points_loss_coef': {
                                                    # 'clouds_classifier': clouds_classifier,
                                                    # 'tissues_classifier': 0.0,
                                                    # 'distance_from_cloud_center': distance_from_cloud_center,
                                                    # 'vae_kld': vae_kld,
                                                    # 'treatment_vectors_collinearity': 0.0,
                                                    # 'drug_vectors_collinearity': 0.0,
                                                    # 'different_directions': 0.0,
                                                    # 'treatment_and_drug_vectors_distance': 0.0,
                                                # }
                                            # })
    df = pd.DataFrame(options)
    return df


def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    version = 'baseline'
    # HOME_PATH = '/RG/compbio/michaelel/lsagne_v0.9.3/lsagne/'
    HOME_PATH = '/RG/compbio/emil/'
    CODE_PATH = os.path.join(HOME_PATH, 'code', version)
    RESULTS_FOLDER = os.path.join(HOME_PATH, 'slurm_results')
    RUN_SLURM_OUTPUT_FOLDER = os.path.join(HOME_PATH, 'slurm_results')
    RUN_SBATCH_PATH = os.path.join(HOME_PATH, 'run_sbatch_files', version)
    PYTHON_OUTPUT_FOLDER = os.path.join(RESULTS_FOLDER, 'python_results', version)
    ROOT_ORGANIZED_FOLDER = os.path.join(HOME_PATH, 'organized_data')
    ORGANIZED_FOLDER_NAME = "cmap"
    CLOUDS_PATH = os.path.join(CODE_PATH, 'slurm_runner/unique_clouds_full_7x8.csv')
    repeat_index_start = 1
    num_of_repeats = 3

    unique_clouds_df = pd.read_csv(CLOUDS_PATH)
    tumor_codes = list(set([str(unique_cloud['tumor_code']) for unique_clouds_i, unique_cloud in unique_clouds_df.iterrows()]))
    for unique_clouds_i, unique_cloud in unique_clouds_df.iterrows():
        tumor_code = str(unique_cloud['tumor_code'])
        non_tumor_code = tumor_codes.copy()
        non_tumor_code.remove(tumor_code)
        non_tumor_code = random.choice(non_tumor_code)
        pert = str(unique_cloud['perturbation'])
        if version=='':
            test_name = _file_name_escaping(tumor_code + '_' + pert + '_' + str(repeat_index_start) + '_' + str(num_of_repeats))
        else:
            test_name = _file_name_escaping(version + '_' + tumor_code + '_' + pert + '_' + str(repeat_index_start) + '_' + str(num_of_repeats))
        test_options = create_run_options(test_name, PYTHON_OUTPUT_FOLDER,non_tumor_code,pert,tumor_code,pert)
        for test_option_i, option in test_options.iterrows():
            sub_test_name = f'{test_name}_MEL_SBTCH_{option_to_name(option)}'
            out_file = sub_test_name + '.out'
            err_file = sub_test_name + '.err'
            sbatch_text = replace_in_string(
                RUN_SBATCH_TEMPLATE,
                {
                    'SLURM_OUTPUT_FOLDER': RUN_SLURM_OUTPUT_FOLDER,
                    'CODE_PATH': CODE_PATH,
                    'OUTPUT_CONST': out_file,
                    'ERROR_CONST': err_file,
                    'START_CONST': str(repeat_index_start),
                    'NUM_OF_REPEATS_CONST': str(num_of_repeats),
                    'PYTHON_OUTPUT_FOLDER': PYTHON_OUTPUT_FOLDER,
                    'ROOT_ORGANIZED_FOLDER': ROOT_ORGANIZED_FOLDER,
                    'ORGANIZED_FOLDER_NAME': ORGANIZED_FOLDER_NAME,
                    'JOB_NAME': sub_test_name,
                    'OVERRIDE_RUN_PARAMETERS': str(option.to_dict()),
                    'JOB_PREFIX': f'test_{test_option_i}_{sub_test_name}'
                })
            sbatch_text += "\n"
            #print(RUN_SBATCH_PATH)
            write_sbatch(sbatch_text, RUN_SBATCH_PATH, sub_test_name+'.sh')


if __name__ == '__main__':
    main()
