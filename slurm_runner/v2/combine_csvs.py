from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np


def combine_csv_with_filename(dir_path: Path):
  csvs = []

  for filename in dir_path.glob('*.csv'):
    df = pd.read_csv(filename, sep='\t')
    df['originating_file'] = filename.name
    csvs.append(df)

  frame = pd.concat(csvs, axis=0, ignore_index=True)
  frame.sort_values(by=['rep', 'originating_file'], inplace=True)
  return frame


def combine_csv_with_filename_main():
    p = Path('/Users/emil/MSc/lsagne-1/output/from_cloud/results_csv/without')
    combined_df = combine_csv_with_filename(p)
    combined_df.to_csv(p / f'{p.name}.csv')


def combine_csv_results(dir_path: Path):
    REPS = 8
    fc_to_rep_to_res = defaultdict(lambda: defaultdict(lambda: [None, None]))

    for filename in dir_path.glob('*.csv'):
        name_parts = filename.name[:-4].split('_')
        if name_parts[0] == 'with':
            fc = name_parts[2]
        else:
            fc = 'NONE'
        run = int(name_parts[-1])
        df = pd.read_csv(filename, sep='\t')
        for _, row in df.iterrows():
            rep = row['rep']
            res = row['real_cloud_svm_es_left_out']
            fc_to_rep_to_res[fc][rep][run-1] = res

    df = defaultdict(list)
    for i in range(REPS):
        df['seed'].append(i+1)
        df['seed'].append(i+1)
    for fc, rep_to_res in fc_to_rep_to_res.items():
        for i in range(REPS):
            df[fc].extend(rep_to_res[i+1])
    df = pd.DataFrame(df)
    df = df[['seed', 'NONE', '1.0', '1.5', '2.0']]
    return df

if __name__ == '__main__':
    # combine_csv_with_filename_main()
    p = Path('/Users/emil/MSc/lsagne-1/output/from_cloud/results_csv/res_1')
    combined_df = combine_csv_results(p)
    combined_df.to_csv(p / f'{p.name}.csv')
