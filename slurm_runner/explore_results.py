import os
import pickle
from os import PathLike

import pandas as pd
import torch
from torch.types import Device

from src.configuration import config
from src.logger_utils import create_logger
from src.pipeline_utils import choose_device
from src.training_summary import TrainingSummary


def merge_h5_chunks(root_dir, output_file):
    # Initialize an empty list to hold dataframes
    dataframes = []

    # Walk through the directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5') and 'results' in file:
                # Construct full file path
                file_path = os.path.join(root, file)
                print(file_path)
                # Read the dataframe and append to the list
                df = pd.read_hdf(file_path)
                dataframes.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Write the merged dataframe to a tab-separated text file
    merged_df.to_csv(output_file, sep='\t', index=False)

def load_training_data(training_summary_path: PathLike, device: Device = None) -> tuple[TrainingSummary, Device]:
    logger = create_logger()
    with open(training_summary_path, "rb") as file:
        training_summary: TrainingSummary = pickle.load(file)
    if not device:
        device = choose_device(training_summary.params.use_cuda, logger)
    torch.set_default_dtype(config.torch_numeric_precision_type)
    # Necessary when the task result is loaded from pickled file
    training_summary.model.load_state_dict(training_summary.model_state_dict)
    training_summary.model.to(device)
    training_summary.model.eval()
    return training_summary, device


if __name__ == '__main__':
    # Define the root directory and output file path
    root_dir = '/RG/compbio/emil/augmentation_optimization_root_v4/results/merged_results.tsv'
    output_file = '/Users/emil/MSc/lsagne-1/output/from_cloud/augmentation_optimization_v4_0-1/merged_results.tsv'

    # training_results, _ = load_training_data('/Users/emil/MSc/lsagne-1/output/experiments/exp_4/exp_4_modules/A549_wortmannin_1_2_experiment_4_baseline-0-A549-wortmannin/_cache/train basic model')


    df = pd.read_csv(output_file, sep='\t')

    gp = df.groupby(['n_pathways', 'n_corrpathways', 'n_genes', 'n_corrgenes', 'use_variance', 'reduction_size'])
    gp_mean = gp[['full_cmap_svm_1_accuracy', 'reduced_cmap_svm_1_accuracy', 'augmented_svm_1_accuracy']].mean().reset_index()
    x = gp_mean[['full_cmap_svm_1_accuracy', 'reduced_cmap_svm_1_accuracy', 'augmented_svm_1_accuracy']].nunique().reset_index()



    gp = df.groupby(['n_pathways', 'n_corrpathways', 'n_genes', 'n_corrgenes', 'use_variance', 'reduction_size', 'tissue', 'perturbation'])
    # gp_mean = gp[['full_cmap_svm_1_accuracy', 'reduced_cmap_svm_1_accuracy', 'augmented_svm_1_accuracy']].mean().reset_index()
    # x = gp_mean[['full_cmap_svm_1_accuracy', 'reduced_cmap_svm_1_accuracy', 'augmented_svm_1_accuracy']].().reset_index()
    # stats = x[['full_cmap_svm_1_accuracy', 'reduced_cmap_svm_1_accuracy', 'augmented_svm_1_accuracy']].describe()





    print('Done!')