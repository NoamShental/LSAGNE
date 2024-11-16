import os
import unittest
from pathlib import Path

import numpy as np

from src.augmentation_parameters_tuning.test_augmentation_parameters import create_augmentation_db
from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_data_augmentation_v1.augment_cmap import CmapAugmentation
from src.cmap_data_augmentation_v1.generate_partial_augmentations_v1 import AugmentationGenerationParameters, \
    generate_augmentations
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.logger_utils import create_logger
from src.os_utilities import create_dir_if_not_exists


class MyTestCase(unittest.TestCase):
    def test_something(self):
        project_root = Path(os.getcwd()).parent.parent
        raw_data_augmentation_dir = project_root / 'raw_data_folder' / 'data_augmentation'
        cmap_dir = project_root / 'organized_data' / 'cmap'
        cmap = RawCmapDataset.load_dataset_from_disk(cmap_folder=cmap_dir)
        test_data_dir = project_root / 'output' / 'test_data'
        aug_db_dir = test_data_dir / 'aug_db_dir'
        cloud_ref = CmapCloudRef('A375', 'geldanamycin')
        aug_samples_dir = test_data_dir / 'aug_samples_dir'
        create_dir_if_not_exists(aug_samples_dir)
        aug_samples = aug_samples_dir / f'{cloud_ref.tissue_code}_{cloud_ref.perturbation}.pkl'
        num_of_samples = 1_000
        if not aug_db_dir.exists():
            augmentation_db_args = create_augmentation_db(
                data_augmentation_dir=raw_data_augmentation_dir,
                cmap_dir=cmap_dir,
                perturbations=[cloud_ref.perturbation],
                output=aug_db_dir
            )
        aug_gen_args = AugmentationGenerationParameters(
            data_augmentation_db_folder=aug_db_dir,
            drug_batch_size=15,
            # tissue=cloud_ref.tissue,
            # perturbation=cloud_ref.perturbation,
            tissue=cloud_ref.tissue,
            perturbation=cloud_ref.perturbation,
            num_of_samples=num_of_samples,
            output=aug_samples,
            # output=None,
            n_pathways=5,
            n_corrpathways=5,
            proba_pathway=1.0,
            n_genes=10,
            n_corrgenes=5,
            proba_gene=1.0,
            use_variance='perDrugMax'
        )
        cloud_ref_to_augmentations = generate_augmentations(aug_gen_args)
        logger = create_logger()
        augmentation_generator = CmapAugmentation(
            logger=logger,
            clouds_to_augment=[cloud_ref],
            # cloud_ref_to_augmentations=cloud_ref_to_augmentations,
            offline_augmentations_folder_path=aug_samples_dir,
            random_seed=0
        )
        samples = cmap.cloud_ref_to_samples[cloud_ref]
        samples_to_augment = np.resize(samples, (num_of_samples, samples.shape[1]))
        samples_before = samples_to_augment.copy()
        augmentation_generator.augment_samples_inplace(
            samples=samples_to_augment,
            cloud_refs=np.repeat(cloud_ref, num_of_samples)
        )
        self.assertGreater(np.count_nonzero(np.abs(samples_before - samples_to_augment) > 0), 0)
        # samples = generate_augmented_samples(
        #     aug_gen_args=aug_gen_args,
        #     cloud_ref=cloud_ref,
        #     samples=cmap.cloud_ref_to_samples[cloud_ref],
        #     size=num_of_samples,
        #     random_seed=0
        # )
if __name__ == '__main__':
    unittest.main()
