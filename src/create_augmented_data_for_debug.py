import dataclasses
import tempfile
from pathlib import Path

from src.cmap_data_augmentation_v1.generate_augmentation_db_v1 import AugmentationDbCreationParameters, \
    generate_augmentation_db
from src.cmap_data_augmentation_v1.generate_partial_augmentations_v1 import generate_augmentations, \
    AugmentationGenerationParameters
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.logger_utils import create_logger
from src.perturbation import Perturbation


def main():
    with tempfile.TemporaryDirectory() as aug_db_dir:
        aug_db_dir = Path(aug_db_dir)
        logger.info(f'Loading cmap from {full_cmap_dir}')
        full_cmap = RawCmapDataset.load_dataset_from_disk(cmap_folder=full_cmap_dir)
        logger.info(f'Creating DB at {aug_db_dir}')
        db_gen_args = AugmentationDbCreationParameters(
            raw_data_augmentation_dir=raw_data_augmentation_dir,
            raw_cmap_dir=full_cmap_dir,
            min_drug_samples_per_cellline=6,
            min_cellines_perdrug=6,
            min_genes_per_go=4,
            max_genes_per_go=50,
            drug_batch_size=15,
            use_compression=False,
            calc_beta=False,
            output_dir=aug_db_dir,
            use_drugs=perturbations_to_augment
        )
        generate_augmentation_db(db_gen_args)
        aug_gen_parameters_template = AugmentationGenerationParameters(
            data_augmentation_db_folder=aug_db_dir,
            drug_batch_size=15,
            # tissue=cloud_ref.tissue,
            # perturbation=cloud_ref.perturbation,
            tissue=None,
            perturbation=None,
            num_of_samples=num_of_samples,
            output=None,
            n_pathways=5,
            n_corrpathways=3,
            proba_pathway=1.0,
            n_genes=5,
            n_corrgenes=3,
            proba_gene=1.0,
            use_variance='perDrugMax'
        )
        for p in perturbations_to_augment:
            logger.info(f'Creating augmentations at {aug_dir} for {p}')
            for cloud_ref in full_cmap.perturbation_to_cloud_refs[p]:
                aug_gen_args = dataclasses.replace(
                    aug_gen_parameters_template,
                    tissue=cloud_ref.tissue,
                    perturbation=cloud_ref.perturbation,
                    output=aug_dir / f'{cloud_ref.tissue_code}_{cloud_ref.perturbation}.pkl'
                )
                generate_augmentations(aug_gen_args)


if __name__ == '__main__':
    full_cmap_dir = Path('/Users/emil/MSc/lsagne-1/organized_data/cmap')
    aug_dir = Path('/Users/emil/MSc/lsagne-1/organized_data/data_augmentation')
    raw_data_augmentation_dir = Path('/Users/emil/MSc/lsagne-1/raw_data_folder/data_augmentation')
    num_of_samples = 10_000
    perturbations_to_augment = [
        Perturbation('trichostatin-a'),
        # Perturbation('geldanamycin')
    ]
    logger = create_logger('aug_gen_debug')
    main()
