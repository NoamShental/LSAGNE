import ast

from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.drug_mixture_test import drug_mixture_test
from src.parameters_parser import ParameterParser
from src.perturbation import Perturbation
from src.tissue import Tissue

print("Loading libraries...")

import argparse
import json
import os
import time

from src import flows
from src.configuration import config
import numpy as np


from src.evaluation.evaluation_printer import print_scores_bar_catplot
from src.os_utilities import create_dir_if_not_exists

PLOT_MODE = 'plot'
TRAIN_MODE = 'train'
AUGMENTATION_TEST_MODE = 'augmentation-test'
DRUG_MIXTURE_TEST_MODE = 'drug-mixture-test'

def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(prog='LSAGNE')
    subparsers = parser.add_subparsers()
    plot = subparsers.add_parser(PLOT_MODE)
    plot.set_defaults(which=PLOT_MODE)
    plot.add_argument('--full-status', help='If set, plot and save catplot on full status', action='store_true')
    plot.add_argument('--flow-name', default=config.flow_name)

    trainer = subparsers.add_parser(TRAIN_MODE)
    trainer.set_defaults(which=TRAIN_MODE)
    trainer.add_argument('--output-folder', default=config.output_folder_path)
    trainer.add_argument('--job-prefix', default='',)
    trainer.add_argument('--organized-cmap-folder', default=config.organized_cmap_folder)
    # TODO: delete, augmentation dir is passed in parameters
    trainer.add_argument('--organized-augmentation-folder', default=config.organized_data_augmentation_folder)
    trainer.add_argument('--override-run-parameters', type=str, default=None)
    trainer.add_argument('--flow-name', default=config.flow_name)
    trainer.add_argument('--repeat-index-start', help='Start replay (included)', type=int, default=0)
    trainer.add_argument('--num-of-repeats', help='End replay (not include)', type=int, default=1)
    trainer.add_argument('--one-job', help='If set, run each test cloud repeats in one job', action='store_true')
    trainer.add_argument('--use-cuda', help='If set, run each test using CUDA', action='store_true')

    augmentation_test = subparsers.add_parser(AUGMENTATION_TEST_MODE)
    augmentation_test.set_defaults(which=AUGMENTATION_TEST_MODE)
    augmentation_test.add_argument('--ratio', help='<Required> How much each cloud should be augmented', type=float, required=True)
    augmentation_test.add_argument('--model', action='append', help='<Required> Path to learned model', required=True)
    augmentation_test.add_argument('--clouds-to-augment', required=True)
    augmentation_test.add_argument('--results-path', required=True)
    augmentation_test.add_argument('--root-organized-folder', default=config.root_organized_data_folder)
    augmentation_test.add_argument('--organized-data-augmentation-folder-name', default=config.organized_data_augmentation_folder_name)

    drug_mixture_test = subparsers.add_parser(DRUG_MIXTURE_TEST_MODE)
    drug_mixture_test.set_defaults(which=DRUG_MIXTURE_TEST_MODE)
    drug_mixture_test.add_argument('--add-24h', help='Adding 24h to mixture', action='store_true')
    drug_mixture_test.add_argument('--model', help='<Required> Path to learned model', type=str, required=True)
    drug_mixture_test.add_argument('--clouds', required=True)
    drug_mixture_test.add_argument('--perturbations', required=True)
    drug_mixture_test.add_argument('--mix-size', type=int, required=True)
    drug_mixture_test.add_argument('-cmap-folder')
    drug_mixture_test.add_argument('-aux-cmap-folder')
    drug_mixture_test.add_argument('-aux-tissue-to-trained-tissue')
    return parser.parse_args()

def update_configuration(args):
    if args.which == TRAIN_MODE:
        config.organized_cmap_folder = args.organized_cmap_folder
        config.organized_data_augmentation_folder = args.organized_augmentation_folder
        config.output_folder_path = args.output_folder
        config.use_cuda = args.use_cuda
    if args.which in [TRAIN_MODE, PLOT_MODE]:
        config.flow_name = args.flow_name
    if args.which == AUGMENTATION_TEST_MODE:
        config.root_organized_data_folder = args.root_organized_folder
        config.organized_data_augmentation_folder_name = args.organized_data_augmentation_folder_name


def do_test():
    el = 0
    cloud_refs = []
    encoded_labels = []
    perturbations = []
    for p in config.perturbations_whitelist:
        for t in config.tissues_whitelist:
            for i in range(10):
                cloud_refs.append(CmapCloudRef(t, p))
                encoded_labels.append(el)
                perturbations.append(Perturbation(p))
            el += 1
    cloud_refs = np.array(cloud_refs)
    encoded_labels = np.array(encoded_labels)
    perturbations = np.array(perturbations)

    loops = 100_000

    start_time = time.time()
    for i in range(loops):
        cloud_ref = np.random.choice(cloud_refs)
        x = cloud_refs == cloud_ref
    cloud_ref_r = (time.time() - start_time) / loops
    print(f'cloud_ref ==> {cloud_ref_r}')

    start_time = time.time()
    for i in range(loops):
        cloud_label = np.random.choice(encoded_labels)
        x = encoded_labels == cloud_label
    cloud_label_r = (time.time() - start_time) / loops
    print(f'cloud_label ==> {cloud_label_r}')

    print(f'ratio 1 ==> {cloud_ref_r / cloud_label_r}')

    start_time = time.time()
    for i in range(loops):
        p = np.random.choice(perturbations)
        x = perturbations == p
    perturbation_r = (time.time() - start_time) / loops
    print(f'perturbation ==> {perturbation_r}')

    print(f'ratio 2 ==> {perturbation_r / cloud_label_r}')

    print("=" * 50)
    print("UNIQUE")

    start_time = time.time()
    for i in range(loops):
        x = set(cloud_refs)
    cloud_ref_r = (time.time() - start_time) / loops
    print(f'cloud_ref ==> {cloud_ref_r}')

    start_time = time.time()
    for i in range(loops):
        x = np.unique(encoded_labels)
    cloud_label_r = (time.time() - start_time) / loops
    print(f'cloud_label_r ==> {cloud_label_r}')

    start_time = time.time()
    for i in range(loops):
        x = np.unique(perturbations)
    perturbation_r = (time.time() - start_time) / loops
    print(f'perturbation_r ==> {perturbation_r}')

    print(f'ratio 1 ==> {cloud_ref_r / cloud_label_r}')
    print(f'ratio 2 ==> {perturbation_r / cloud_label_r}')








if __name__ == '__main__':
    # do_test()
    # exit()
    # TODO do alias for common types
    print("Parsing arguments...")
    args = parse_arguments()
    update_configuration(args)

    print(f"Running Mode {args.which}...")

    if args.which == PLOT_MODE:
        flow_name = args.flow_name
        results_dir = os.path.join('.', 'output', flow_name)
        prints_dir = os.path.join('.', 'prints', flow_name)
        create_dir_if_not_exists(prints_dir)
        print_scores_bar_catplot(results_dir=results_dir,
                                 score_file_name='score-z-closest.txt',
                                 title="Scores by predicting only 'len' closest",
                                 draw_file_name='len_closest_scores_bar_catplot',
                                 drawing_dir=prints_dir)

        # print_scores_flat(
        #     results_dir=results_dir,
        #     score_file_name='score-z-closest.txt',
        #     title="Scores by predicting only 'len' closest",
        #     draw_file_name='len_closest_scores_bar_catplot',
        #     drawing_dir=prints_dir,
        #     y_axis_field='run name'
        # )

        file_name = 'correlation_distances.csv'
        metric = 'median'

        # print_scores_flat(
        #     results_dir='C:\\files_from_server\\runs',
        #     score_file_name='score-z-closest.txt',
        #     title="VCAP-wortmannin scores by predicting only 'len' closest",
        #     draw_file_name='VCAP_wortmannin_len_closest_scores',
        #     drawing_dir=prints_dir,
        #     y_axis_field='run name'
        # )

        # print_cdist_bar_catplot(results_dir=results_dir,
        #                         cdist_file_name=file_name,
        #                         metric=metric,
        #                         title=f'CDIST using {file_name}',
        #                         draw_file_name='corr',
        #                         drawing_dir=prints_dir)

        # print_cdist_by_field(f'./output/{flow_name}', file_name=file_name, metric=metric)

        exit(0)

    if args.which == TRAIN_MODE:
        flow_name = config.flow_name
        i = 0
        for i in range(args.repeat_index_start, args.repeat_index_start + args.num_of_repeats):
            flow, params = flows.create_basic_model_flow(
                flow_name=flow_name,
                use_cuda=True,
                override_parameters=json.loads(args.override_run_parameters.replace("'", "\"")) if args.override_run_parameters else None
            )
            run_name = '-'.join([
                f'{i}' if not args.job_prefix else f'{args.job_prefix}-{i}',
                params.left_out_cloud.tissue_code,
                params.left_out_cloud.perturbation
            ])
            state = flow.run(parameters={
                'run_name': run_name
            })
            flow.get_tasks(tags=['logger'])[0].close_logger()

    if args.which == AUGMENTATION_TEST_MODE:
        raise NotImplementedError("Please fix the test before usage")
        clouds_to_augment = ast.literal_eval(args.clouds_to_augment)
        augmentation_test(args.ratio, args.model, clouds_to_augment,
                          os.path.join(args.results_path, 'AugTest_' + config.organized_data_augmentation_folder_name + '.csv'), 0)

    if args.which == DRUG_MIXTURE_TEST_MODE:
        if args.cmap_folder:
            cmap_folder = args.cmap_folder
        else:
            cmap_folder = config.organized_cmap_folder
        test_cmap = RawCmapDataset.load_dataset_from_disk(cmap_folder)
        if args.aux_cmap_folder:
            aux_cmap = RawCmapDataset.load_dataset_from_disk(cmap_folder=args.aux_cmap_folder)
            # add to config
            for tissue_str in aux_cmap.info_df['tumor'].unique():
                if tissue_str in test_cmap.tissues_unique:
                    continue
                tissue_code = tissue_str.split(' ')[0]
                config.tissue_name_to_code[tissue_str] = tissue_code
                config.tissue_code_to_name[tissue_code] = tissue_str
                Tissue.ALLOWED_TISSUES.add(tissue_str)
            for perturbation_str in aux_cmap.info_df['perturbation'].unique():
                if perturbation_str in test_cmap.perturbations_unique:
                    continue
                Perturbation.ALLOWED_PERTURBATIONS.add(perturbation_str)
        else:
            aux_cmap = None
        aux_tissue_to_trained_tissue = None
        if args.aux_tissue_to_trained_tissue:
            aux_tissue_to_trained_tissue = {
                Tissue(t1): Tissue(t2)
                for t1, t2 in ast.literal_eval(args.aux_tissue_to_trained_tissue)}
        else:
            aux_tissue_to_trained_tissue = None
        perturbations = ast.literal_eval(args.perturbations)

        combined_cmap = test_cmap.merge_datasets(aux_cmap) if aux_cmap else test_cmap
        parameters_parser = ParameterParser(combined_cmap)
        drug_mixture_test(
            add_24h=args.add_24h,
            model_path=args.model,
            clouds=parameters_parser.parse_cloud_refs('clouds', ast.literal_eval(args.clouds)),
            perturbations=perturbations,
            mix_size=args.mix_size,
            test_cmap=test_cmap,
            aux_cmap=aux_cmap,
            aux_tissue_to_trained_tissue=aux_tissue_to_trained_tissue
        )


    print('All Done. Exiting main thread')
