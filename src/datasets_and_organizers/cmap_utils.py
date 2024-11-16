from logging import Logger
from pathlib import Path
from shutil import copyfile

import joblib

from src.configuration import config
from src.datasets_and_organizers.cmap_organizer import CmapOrganizer
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.os_utilities import create_dir_if_not_exists
from src.perturbation import Perturbation
import numpy as np


def load_and_reduce_cmap(
        logger: Logger,
        full_cmap_folder: Path,
        reduced_cmap_folder: Path,
        perturbation_to_resize: dict[Perturbation, int] | None,
        random_seed: int | None = None
) -> RawCmapDataset:
    create_dir_if_not_exists(reduced_cmap_folder)
    rng = np.random.default_rng(random_seed)
    cmap = RawCmapDataset.load_dataset_from_disk(
        cmap_folder=full_cmap_folder
    )
    # TODO DRY
    if not perturbation_to_resize:
        logger.info('There is not need to reduce any CMAP clouds')
        perturbation_to_resize = {}
    logger.info('Starting CMAP reduction clouds')
    mask = np.zeros(len(cmap), bool)
    for cloud_ref, cloud_idx in cmap.cloud_ref_to_idx.items():
        if cloud_ref.perturbation not in perturbation_to_resize:
            mask[cloud_idx] = True
            continue
        resize = perturbation_to_resize[cloud_ref.perturbation]
        if len(cloud_idx) < resize:
            resize = len(cloud_idx)
        logger.info(f'Reducing cloud {cloud_ref} from {len(cloud_idx):,} to {resize: ,}')
        cloud_idx = rng.choice(cloud_idx, resize, replace=False)
        mask[cloud_idx] = True
    logger.info(f'Reducing CMAP completed')
    reduced_cmap = RawCmapDataset(data_df=cmap.data_df[mask], info_df=cmap.info_df[mask], scaler=cmap.scaler)
    logger.info(f'In total cmap was reduced from {len(cmap):,} to {len(reduced_cmap):,}')
    reduced_cmap.data_df.to_hdf(reduced_cmap_folder / config.data_file_name, key='df')
    reduced_cmap.info_df.to_csv(reduced_cmap_folder / config.information_file_name)
    scaler_path = reduced_cmap_folder / 'cmap_scaler'
    joblib.dump(reduced_cmap.scaler, scaler_path)
    unique_clouds_file_name = 'unique_clouds.csv'
    copyfile(full_cmap_folder / unique_clouds_file_name, reduced_cmap_folder / unique_clouds_file_name)
    return reduced_cmap


def create_cmap_from_raw(
        raw_cmap_folder: Path,
        output_cmap: Path
):
    cmap_organizer = CmapOrganizer(
        raw_cmap_folder=raw_cmap_folder,
        organized_cmap_folder=output_cmap,
        perturbations_whitelist=config.perturbations_whitelist,
        untreated_labels=config.untreated_labels,
        untreated_times=config.untreated_times,
        tissues_whitelist=config.tissues_whitelist,
        perturbation_times=config.perturbation_times,
        min_treat_conc=config.min_treat_conc,
        untreated_labels_times=config.cmap_organizer_untreated_labels_times,
        data_file_name=config.data_file_name,
        information_file_name=config.information_file_name,
        min_samples_per_cloud=config.min_samples_per_cloud,
        unique_clouds_file_name=config.unique_clouds_file_name
    )
    cmap_organizer.organize_data()

