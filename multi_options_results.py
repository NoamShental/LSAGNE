import os
import pandas as pd
import numpy as np
import os.path
from os import path

if __name__ == '__main__':
    version = '0.83.0'
    root_dir = f'./output/{version}'

    experiments_df = pd.read_csv(os.path.join(root_dir, f'{version}_HT29_geldanamycin_0_1_run_options.csv'))

    experiments_ids = experiments_df.iloc[:, 0]

    # experiments_id_to_results = {}

    full_df = None

    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            experiment_id = int(dir.split('_')[2].split('-')[0])
            retry_num = int(dir.split('_')[2].split('-')[1])

            svm_score_on_predicted_file_path = os.path.join(root_dir, dir, 'svm_score_on_predicted.csv')
            score_z_closest_file_path = os.path.join(root_dir, dir, 'score-z-closest.txt')

            if not path.exists(svm_score_on_predicted_file_path) or not path.exists(score_z_closest_file_path):
                # this experiment failed
                print(f'Experiment "{dir}" failed, ignoring...')
                continue



            experiment_results = pd.read_csv(svm_score_on_predicted_file_path)
            experiment_results.drop(columns=experiment_results.columns[0], axis=1, inplace=True)
            experiment_results.rename(columns={
                # experiment_results.columns[0]: 'id',
                experiment_results.columns[-1]: f'{experiment_id}_{retry_num}'
            }, inplace=True)

            with open(score_z_closest_file_path, 'r') as reader:
                line = reader.readline()
                split_line = line.split(' ')
                tissue_name = str.join(' ', split_line[:-2])
                perturbation = split_line[-2]
                score = float(split_line[-1])
                mean = experiment_results.iloc[:, -1].astype(float).mean()
                experiment_results.loc[len(experiment_results.index)] = ['MEAN', 'MEAN', mean]
                experiment_results.loc[len(experiment_results.index)] = ['LEFT OUT', 'LEFT OUT', score]
                experiment_results.loc[len(experiment_results.index)] = ['TOTAL', 'TOTAL', (mean + score) / 2]

            if full_df is not None:
                full_df = pd.merge(full_df, experiment_results, on=['tissue', 'perturbation'])
            else:
                full_df = experiment_results
            # experiments_id_to_results[experiment_id] = experiment_results

            # single_experiment_path = os.path.join(root_dir, dir)
        break


    x = full_df.iloc[:, 2:].to_numpy().astype(float)
    s = np.argsort(x[-1])[::-1]
    ss = x[:, s]

    for i in s:
        idx = i + 2
        column_name = full_df.columns[idx]
        full_column = full_df.loc[:,column_name]
        all_left_in = full_column.iloc[:-3]
        mean_left_in = full_column.iloc[-3]
        left_out = full_column.iloc[-2]
        final_score = full_column.iloc[-1]
        print(f'{column_name} --> final score = {final_score},  left-in mean={mean_left_in}, '
              f'left_in_min={all_left_in.min()} left-out={left_out}')

    # print(full_df.columns[s+2])