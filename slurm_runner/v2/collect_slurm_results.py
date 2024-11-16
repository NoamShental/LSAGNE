import argparse
import os.path
import re
import sys
from collections import Counter
from dataclasses import dataclass
from glob import glob
from io import StringIO
from os.path import join, isdir
from pathlib import Path

import pandas as pd
import torch
from natsort import natsorted

from src.augmentation_parameters_tuning.test_augmentation_parameters import extract_svms_from_trained_model, \
    get_svm_results
from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset


@dataclass(frozen=True)
class ResultCollectionParameters:
    full_cmap: Path
    runs_root: Path
    output: Path

    def __post_init__(self):
        assert self.full_cmap.is_dir()
        assert self.runs_root.is_dir()
        # assert self.output.is_file()

    @classmethod
    def create_using_args(cls):
        parser = argparse.ArgumentParser(description="Generating ...")
        parser.add_argument(
            "--full-cmap",
            type=Path,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--runs-root",
            type=Path,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="..."
        )
        args = parser.parse_args()
        return cls(
            full_cmap=args.full_cmap,
            runs_root=args.runs_root,
            output=args.output
        )


def parse_svm_result_df(df_str: list[str]):
    left_out = None
    cv = None

    def parse_line(l: str, name: str) -> dict:
        absolute_svm_acc, equivalence_sets_svm_acc = re.split(r"\s+", l.rstrip())[-2:]
        return {
            f'{name}_absolute_svm_acc': float(absolute_svm_acc),
            f'{name}_equivalence_sets_svm_acc': float(equivalence_sets_svm_acc),
        }

    for line in df_str:
        if cv and left_out:
            break
        if line.endswith('Left Out\n'):
            left_out = parse_line(line.replace('Left Out\n', ''), 'left_out')
        elif line.endswith('CV\n'):
            cv = parse_line(line.replace('CV\n', ''), 'cv')
    return {
        **cv,
        **left_out
    }

def extract_rep(exp: str, cell_line: str, drug: str) -> int | None:
    pattern = r'\d+-' + re.escape(f'{cell_line}-{drug}')  # Create pattern
    match = re.search(pattern, exp)

    if match:
        return int(match.group(0).split('-')[0])  # Get number before the '-'
    raise RuntimeError(f'Cloud not parse "{exp}"')


@torch.no_grad()
def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    params = ResultCollectionParameters.create_using_args()
    print('Loading full CMAP')
    full_cmap = RawCmapDataset.load_dataset_from_disk(cmap_folder=params.full_cmap)
    print('Starting to collect...')
    runs = natsorted([r for r in params.runs_root.glob('*') if r.is_dir()])
    print(f'Found {len(runs):,} runs.')
    full_res = []
    for i, run in enumerate(runs):
        try:
            _split = run.name.split('_')
            cell_line = _split[0]
            drug = _split[1]
            seed = int(_split[3])
            rep = int(_split[5])
            is_augmented = _split[30] == 'True'
            fold_change = float(_split[35].split('-')[0])
        except Exception as e:
            print(f'Could not parse the experiment {run}, error: {e}')
            continue
        print(f'Collecting #{i:,} {cell_line} {drug} {rep}')
        with open(os.path.join(run, 'log.txt')) as file:
            lines = [line.rstrip() for line in file]
        if 'Flow run SUCCESS: all reference tasks succeeded' not in lines[-1]:
            print(f'Run {cell_line} {drug} {rep} has not finished or has failed...')
            continue
        svm_results = sorted((run / 'svm_results').glob('**/epoch_*/svm_1/summary'))
        res = []
        for svm_result in svm_results:
            epoch_path = svm_result.parent.parent
            is_best = epoch_path.parent.name == 'best_epoch'
            epoch = int(epoch_path.name.split('_')[1])
            with open(svm_result) as f:
                df_str = f.readlines()[3:]
            res.append({
                'epoch': epoch,
                'is_best': is_best,
                **parse_svm_result_df(df_str)
            })
        res_df = pd.DataFrame(res)
        best_epoch_res = res_df.loc[res_df['is_best']].squeeze()
        last_epoch_res = res_df.loc[res_df['epoch'].idxmax()].squeeze()
        trained_model = extract_svms_from_trained_model(run / '_cache' / 'train basic model')
        left_out_cloud_ref = CmapCloudRef(tissue=cell_line, perturbation=drug)
        full_cloud_samples = full_cmap.cloud_ref_to_samples[left_out_cloud_ref]
        svm_results = get_svm_results(trained_model, full_cloud_samples, left_out_cloud_ref, trained_model.summary.params.perturbations_equivalence_sets)
        full_res.append({
            'cell': cell_line,
            'drug': drug,
            'rep': rep,
            'seed': seed,
            'is_augmented': is_augmented,
            'fold_change': fold_change,
            'real_cloud_svm_absolute_left_out': svm_results.svm_1.absolute_svm_accuracy,
            'real_cloud_svm_es_left_out': svm_results.svm_1.equivalence_sets_svm_accuracy,
            'best_epoch': best_epoch_res['epoch'],
            'best_epoch_absolute_left_out': best_epoch_res['left_out_absolute_svm_acc'],
            'best_epoch_es_left_out': best_epoch_res['left_out_equivalence_sets_svm_acc'],
            'best_epoch_absolute_cv': best_epoch_res['cv_absolute_svm_acc'],
            'best_epoch_es_cv': best_epoch_res['cv_equivalence_sets_svm_acc'],
            'last_epoch': last_epoch_res['epoch'],
            'last_epoch_absolute_left_out': last_epoch_res['left_out_absolute_svm_acc'],
            'last_epoch_es_left_out': last_epoch_res['left_out_equivalence_sets_svm_acc'],
            'last_epoch_absolute_cv': last_epoch_res['cv_absolute_svm_acc'],
            'last_epoch_es_cv': last_epoch_res['cv_equivalence_sets_svm_acc'],
            'max_training_perturbation_cloud': max([len(samples) for cloud, samples in trained_model.summary.cmap_datasets.training_only.cloud_ref_to_samples.items() if cloud.perturbation == left_out_cloud_ref.perturbation]),
            'min_training_perturbation_cloud': min([len(samples) for cloud, samples in trained_model.summary.cmap_datasets.training_only.cloud_ref_to_samples.items() if cloud.perturbation == left_out_cloud_ref.perturbation]),
            'evaluation_cloud_size': len(full_cloud_samples),
            'perdiction_hist': Counter(svm_results.svm_1.svm_prediction)
        })
    full_res_df = pd.DataFrame(full_res)
    full_res_df.to_csv(params.output, sep='\t', index=False)
    # bestdf = full_res_df.groupby(['Cell','Drug'])['LOMax_Left_Out'].max()
    # bestdf.reset_index()
    # bestdf.to_csv(sys.argv[2].replace('.csv','_best.csv'),sep='\t')

if __name__ == '__main__':
    main()
