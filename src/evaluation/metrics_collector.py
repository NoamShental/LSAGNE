import os
import pandas as pd
import numpy as np

from src.evaluation.statistics_utils import calculate_detailed_statistics


def collect_scores(root_dir, file_name):
    """
    :param str root_dir: A directory where every sub directory is a valid run of the algorithm
    :param str file_name: Score file name, which will be found in the directory of each run. This file contains single line
                      "<TISSUE CODE> <PERTURBATION> <SCORE>"
    :return:
        - Dataframe
        - Statistics of the scores
    """
    all_runs = os.listdir(root_dir)
    tissue_codes = []
    perturbations = []
    display_names = []
    scores = []
    run_names = []
    for run in all_runs:
        file_path = os.path.join(root_dir, run, file_name)
        # file_path = f'{root_dir}/{run}/{file_name}'
        if not os.path.isfile(file_path):
            continue
        run_name = run.split('-')[0]
        run_names.append(run_name)
        with open(file_path, 'r') as reader:
            line = reader.readline()
            split_line = line.split(' ')
            tissue_code = split_line[0]
            perturbation = split_line[-2]
            score = split_line[-1]
            # tissue_code, perturbation, score = line.split(' ')
            tissue_codes.append(tissue_code)
            perturbations.append(perturbation)
            display_names.append(f'{tissue_code} {perturbation}')
            scores.append(float(score))

    df = pd.DataFrame({
        'tissue_code': tissue_codes,
        'perturbation': perturbations,
        'display name': display_names,
        'score': scores,
        'run name': run_names,
    })

    detailed_statistics = calculate_detailed_statistics(scores)

    return df, detailed_statistics


def collect_cdists(root_dir, cdist_file_name, metric):
    all_runs = os.listdir(root_dir)
    df = pd.DataFrame()
    for run in all_runs:
        file_path = os.path.join(root_dir, run, cdist_file_name)
        if not os.path.isfile(file_path):
            continue
        splitted_run = run.split('-')
        run_name = splitted_run[0]
        tissue_code = splitted_run[1]
        perturbation = '-'.join(splitted_run[2:]) if len(splitted_run) > 3 else splitted_run[2]
        cdists = pd.read_csv(file_path,index_col=0)
        nearest = cdists[cdists.columns[5:]].loc[metric].idxmin()
        cdists.rename(columns={nearest: 'p_n'}, inplace=True)
        for corr_name in ['r_r', 'r_p', 'p_p', 'r_t', 'p_t', 'p_n']:
            notes = nearest if corr_name == 'p_n' else ''
            df = df.append({
                'tissue_code': tissue_code,
                'perturbation': perturbation,
                'display name': f'{tissue_code} {perturbation}',
                'corr_name': corr_name,
                'metric': metric,
                'run name': run_name,
                'score': cdists[corr_name][metric],
                'notes': notes
            }, ignore_index=True)
    return df
