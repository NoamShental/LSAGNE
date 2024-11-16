import logging
import pickle
import sys
from logging import Logger
from os import path
from typing import List, Dict, Set, Tuple

import pandas as pd
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_data_augmentation_v1.augment_cmap import CmapAugmentation
from src.configuration import config
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.svm_utils import train_linear_svm_using_training_and_predicted_non_trained
from src.training_summary import TrainingSummary


def create_label_to_samples(labels: NDArray[int], samples: NDArray[float]) -> Dict[int, NDArray[float]]:
    label_to_samples = {}
    for label in np.unique(labels):
        mask = labels == label
        label_to_samples[label] = samples[mask]
    return label_to_samples


def create_logger(level=logging.DEBUG) -> Logger:
    formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def train_and_calculate_svm(
        svm_training_z: NDArray[float],
        svm_training_z_original_numeric_labels: NDArray[int],
        random_seed: int
) -> LinearSVC:
    linear_svm_classifier, _, _ = train_linear_svm_using_training_and_predicted_non_trained(
        non_trained_predicted_clouds=[],
        training_z=svm_training_z,
        training_z_original_numeric_labels=svm_training_z_original_numeric_labels,
        random_seed=random_seed
    )
    return linear_svm_classifier


def parse_cloud_refs_to_augment(
        clouds_strs: List[str],
        cmap_dataset: RawCmapDataset,
        mask: NDArray[bool],
        all_literal: str = "all") -> List[CmapCloudRef]:
    all_cloud_refs = []
    for encoded_label in np.unique(cmap_dataset.encoded_labels[mask]):
        p, t = cmap_dataset.encoded_label_to_perturbation_and_tissue[encoded_label]
        all_cloud_refs.append(CmapCloudRef(t, p))

    clouds_refs = set()
    if len(clouds_strs) % 2 != 0:
        raise AssertionError(f'len is odd.')
    for i in range(0, len(clouds_strs), 2):
        tissue_code = clouds_strs[i]
        perturbation = clouds_strs[i + 1]
        if tissue_code == all_literal and perturbation == all_literal:
            return all_cloud_refs
        if tissue_code == all_literal:
            clouds_refs.update([cloud_ref for cloud_ref in all_cloud_refs
                                if cloud_ref.perturbation == perturbation])
            continue
        if perturbation == all_literal:
            clouds_refs.update([cloud_ref for cloud_ref in all_cloud_refs
                                if cloud_ref.tissue_code == tissue_code])
            continue
        cloud_ref = CmapCloudRef(tissue_code, perturbation)
        if cloud_ref not in all_cloud_refs:
            raise AssertionError(f'Cloud {cloud_ref} is not possible for this CMAP')
        clouds_refs.add(cloud_ref)
    return list(clouds_refs)


def do_test(
        logger: Logger,
        ratio: float,
        clouds_strs: List[str],
        loaded_training_data: TrainingSummary,
        all_common_sample_ids: Set[str],
        random_seed: int) -> Tuple[Dict[CmapCloudRef, float], Dict[CmapCloudRef, int]]:
    train_cmap = loaded_training_data.updated_train_cmap_dataset
    common_samples_mask = np.isin(train_cmap.samples_cmap_id,
                                  np.array(list(all_common_sample_ids)))
    assert len(all_common_sample_ids) == common_samples_mask.sum()
    common_samples = train_cmap.data[common_samples_mask]
    common_encoded_numeric_labels = train_cmap.encoded_labels[common_samples_mask]

    clouds_to_augment = parse_cloud_refs_to_augment(clouds_strs, train_cmap, common_samples_mask)

    samples_t = torch.tensor(common_samples)
    model = loaded_training_data.model
    model.eval()
    z_t = model(samples_t).z_t
    z = z_t.detach().numpy()

    svm = train_and_calculate_svm(z, common_encoded_numeric_labels, random_seed)

    logger.info(f'Loading augmentations from {config.organized_data_augmentation_folder}...')
    cmap_augmentor = CmapAugmentation(
        logger,
        config.organized_data_augmentation_folder,
        clouds_to_augment,
        train_cmap.perturbation_and_tissue_to_encoded_label
    )

    cloud_ref_to_svm_accuracy = {}

    for cloud_ref in clouds_to_augment:
        encoded_label = train_cmap.perturbation_and_tissue_to_encoded_label[(cloud_ref.perturbation, cloud_ref.tissue)]
        mask = common_encoded_numeric_labels == encoded_label
        original_cloud_samples = common_samples[mask]
        original_cloud_size = len(original_cloud_samples)
        take_count = int(ratio * original_cloud_size)
        augmented_cloud_samples = original_cloud_samples[:take_count].copy()
        cloud_encoded_labels = common_encoded_numeric_labels[mask][:take_count]
        cmap_augmentor.augment_samples_inplace(augmented_cloud_samples, cloud_encoded_labels)
        assert np.any(original_cloud_samples[:take_count] != augmented_cloud_samples)
        # FIXME remove all .detach().numpy()
        augmented_cloud_z = model(torch.tensor(augmented_cloud_samples)).z_t.detach().numpy()
        svm_predict = svm.predict(augmented_cloud_z)
        svm_accuracy = accuracy_score(cloud_encoded_labels, svm_predict)
        cloud_ref_to_svm_accuracy[cloud_ref] = svm_accuracy

    return cloud_ref_to_svm_accuracy


def augmentation_test(ratio: float, model_paths: List[str], clouds_strs: List[str], results_filename: str, random_seed: int = 0):
    logger: Logger = create_logger()
    logger.info(f'Starting augmentation test...')

    logger.info(f'Loading full CMAP...')
    full_cmap = RawCmapDataset.load_dataset_from_disk(config.perturbations_whitelist, config.tissues_whitelist)

    model_path_to_loaded_training_data: Dict[str, TrainingSummary] = {}
    for model_path in model_paths:
        logger.info(f'Loading model {model_path}...')
        with open(model_path, "rb") as file:
            loaded_training_data = pickle.load(file)
            model_path_to_loaded_training_data[model_path] = loaded_training_data
        logger.info(f'Model {model_path} has {len(loaded_training_data.updated_train_cmap_dataset):,} samples.')
    logger.info(f'All models loaded')

    all_common_sample_ids = set.intersection(
        *[set(loaded_training_data.updated_train_cmap_dataset.samples_cmap_id)
          for loaded_training_data in list(model_path_to_loaded_training_data.values())])

    logger.info(f'All common samples len = {len(all_common_sample_ids):,}.')

    model_path_cloud_ref_to_svm_accuracy = {}

    for model_path, loaded_training_data in model_path_to_loaded_training_data.items():
        logger.info(f'Testing model "{model_path}"...')
        cloud_ref_to_svm_accuracy = do_test(
            logger=logger,
            ratio=ratio,
            clouds_strs=clouds_strs,
            loaded_training_data=loaded_training_data,
            all_common_sample_ids=all_common_sample_ids,
            random_seed=random_seed
        )
        model_path_cloud_ref_to_svm_accuracy[model_path] = cloud_ref_to_svm_accuracy

    all_cloud_refs = set().union(*[list(cloud_ref_to_svm_accuracy.keys()) for cloud_ref_to_svm_accuracy in model_path_cloud_ref_to_svm_accuracy.values()])

    logger.info('Calculate original cloud size...')
    df_rows = []
    for cloud_ref in all_cloud_refs:
        encoded_label = full_cmap.perturbation_and_tissue_to_encoded_label[(cloud_ref.perturbation, cloud_ref.tissue)]
        df_rows.append({
            'perturbation': cloud_ref.perturbation,
            'tissue': cloud_ref.tissue,
            'original size': full_cmap.encoded_label_to_idx_mask[encoded_label].sum()
        })
    df = pd.DataFrame(df_rows).set_index(['perturbation', 'tissue'])

    logger.info('Calculate cloud size after filtering...')
    df_rows = []
    train_cmap = model_path_to_loaded_training_data[next(iter(model_path_to_loaded_training_data))].updated_train_cmap_dataset
    common_samples_mask = np.isin(train_cmap.samples_cmap_id,
                                  np.array(list(all_common_sample_ids)))
    common_encoded_numeric_labels = train_cmap.encoded_labels[common_samples_mask]
    for cloud_ref in all_cloud_refs:
        encoded_label = train_cmap.perturbation_and_tissue_to_encoded_label[(cloud_ref.perturbation, cloud_ref.tissue)]
        df_rows.append({
            'perturbation': cloud_ref.perturbation,
            'tissue': cloud_ref.tissue,
            'filtered size': (common_encoded_numeric_labels == encoded_label).sum()
        })
    df['filtered size'] = pd.DataFrame(df_rows).set_index(['perturbation', 'tissue'])

    for model_path, cloud_ref_to_svm_accuracy in model_path_cloud_ref_to_svm_accuracy.items():
        df_rows = []
        for cloud_ref in all_cloud_refs:
            accuracy = cloud_ref_to_svm_accuracy[cloud_ref] if cloud_ref in cloud_ref_to_svm_accuracy else None
            df_rows.append({
                'perturbation': cloud_ref.perturbation,
                'tissue': cloud_ref.tissue,
                f'{model_path} accuracy': accuracy,
            })
        if df is None:
            df = pd.DataFrame(df_rows).set_index(['perturbation', 'tissue'])
        else:
            current_experiment_results = pd.DataFrame(df_rows).set_index(['perturbation', 'tissue'])
            df[current_experiment_results.columns] = current_experiment_results

    df = df.sort_values(['perturbation', 'tissue'])
    df[config.organized_data_augmentation_folder_name] = df.iloc[: , -(len(df. columns)-2):].mean(axis=1)
    df.to_csv(results_filename)