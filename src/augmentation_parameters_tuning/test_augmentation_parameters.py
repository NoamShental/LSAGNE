import argparse
import dataclasses
import pickle
from collections import defaultdict
from dataclasses import dataclass
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Generator, Collection, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.types import Device

from src.augmentation_parameters_tuning.df_results_utils import DfResultsSaver
from src.cmap_cloud import CmapCloud
from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_data_augmentation_v1.augment_cmap import CmapAugmentation as AugmentationGenerator
from src.cmap_data_augmentation_v1.generate_augmentation_db_v1 import AugmentationDbCreationParameters, \
    generate_augmentation_db
from src.cmap_data_augmentation_v1.generate_partial_augmentations_v1 import AugmentationGenerationParameters, \
    generate_augmentations, AugmentationVariace
from src.cmap_evaluation_data import SplittedCmapEvaluationData
from src.configuration import config
from src.datasets_and_organizers.cmap_utils import create_cmap_from_raw, load_and_reduce_cmap
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.logger_utils import create_logger
from src.models.cmap_cloud_tag import CmapCloudTag
from src.models.dudi_basic.embedded_anchors_and_vectors import EmbeddedAnchorsAndVectors
from src.models.dudi_basic.lsagne_model import LsagneModel
from src.models.dudi_basic.multi_device_data import AnchorPointsLookup
from src.models.svm_utils import perform_svm_accuracy_evaluation, AllCmapSvmEvaluationResults, \
    CmapSvmCloudPredictionResults
from src.os_utilities import create_dir_if_not_exists
from src.perturbation import Perturbation
from src.pipeline_utils import choose_device
from src.samples_embedder import SamplesEmbedder
from src.training_summary import TrainingSummary


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class ModelRef:
    path: PathLike
    alias: str


@dataclass(frozen=True)
class AugmentationTestArgs:
    logger: Logger
    n_pathways: Collection[int]
    n_corrpathways: Collection[int]
    n_genes: Collection[int]
    n_corrgenes: Collection[int]
    use_variance: Collection[AugmentationVariace]
    random_seed_range: tuple[int, int]
    raw_data_augmentation_dir: Path
    raw_cmap_dir: Path
    working_dir: Path
    model_paths: Collection[ModelRef]
    augmentation_size: int
    reduction_sizes: Collection[int]


@dataclass(frozen=True)
class SvmResults:
    svm_1: CmapSvmCloudPredictionResults
    # svm_2: CmapSvmCloudPredictionResults


@dataclass(frozen=True)
class ExperimentResult:
    model_ref: ModelRef
    cloud_ref: CmapCloudRef
    n_pathways: int
    n_corrpathways: int
    n_genes: int
    n_corrgenes: int
    use_variance: str
    aug_gen_drug_batch_size: int
    reduction_size: int
    random_seed: int
    aug_db_cmap_dir: PathLike
    min_drug_samples_per_cellline: int
    min_cellines_perdrug: int
    min_genes_per_go: int
    max_genes_per_go: int
    aug_db_drug_batch_size: int
    aug_db_use_compression: bool
    calc_beta: bool
    aug_db_dir: PathLike
    # the SVMs are calculated on the post training data of each trained model
    # FULL = use the full CMAP samples on the SVM
    full_cmap_svm_results: SvmResults
    # REDUCED = use the reduced CMAP samples on the SVM, this allows us to detect "poor selection" of reduced CMAP
    reduced_cmap_svm_results: SvmResults
    # AUGMENTED = use the augmented samples on the SVM
    augmented_svm_results: SvmResults

    @staticmethod
    def calculate_confusion_vector(predictions: NDArray[CmapCloudRef]) -> dict[CmapCloudRef, int]:
        unique_clouds, counts = np.unique(predictions, return_counts=True)
        return {
            cloud: count for cloud, count in zip(unique_clouds, counts)
        }

    def _add_svm_res_to_dict(
            self,
            res: dict[str, Any],
            svm_results: SvmResults,
            name_prefix: str
    ) -> None:
        # res[f'{name_prefix}_svm_1_results'] = svm_results.svm_1.svm_prediction
        res[f'{name_prefix}_svm_1_results'] = self.calculate_confusion_vector(svm_results.svm_1.svm_prediction)
        res[f'{name_prefix}_svm_1_accuracy'] = svm_results.svm_1.absolute_svm_accuracy
        res[f'{name_prefix}_svm_1_es_accuracy'] = svm_results.svm_1.equivalence_sets_svm_accuracy
        # res[f'{name_prefix}_svm_2_results'] = svm_results.svm_2.svm_prediction
        # res[f'{name_prefix}_svm_2_accuracy'] = svm_results.svm_2.absolute_svm_accuracy
        # res[f'{name_prefix}_svm_2_es_accuracy'] = svm_results.svm_2.equivalence_sets_svm_accuracy

    def to_dict(self) -> dict:
        res = dataclasses.asdict(self)
        res.pop('model_ref')
        res['model_name'] = self.model_ref.alias
        res['model_path'] = self.model_ref.path
        res.pop('cloud_ref')
        res['tissue'] = self.cloud_ref.tissue_code
        res['perturbation'] = self.cloud_ref.perturbation
        res.pop('full_cmap_svm_results')
        self._add_svm_res_to_dict(res, self.full_cmap_svm_results, 'full_cmap')
        res.pop('reduced_cmap_svm_results')
        self._add_svm_res_to_dict(res, self.reduced_cmap_svm_results, 'reduced_cmap')
        res.pop('augmented_svm_results')
        self._add_svm_res_to_dict(res, self.augmented_svm_results, 'augmented')
        return res


def load_training_data(training_summary_path: PathLike, device: Device = None) -> tuple[TrainingSummary, Device]:
    with open(training_summary_path, "rb") as file:
        training_summary: TrainingSummary = pickle.load(file)
    if not device:
        device = choose_device(training_summary.params.use_cuda)
    torch.set_default_dtype(config.torch_numeric_precision_type)
    # Necessary when the task result is loaded from pickled file
    training_summary.model.load_state_dict(training_summary.model_state_dict)
    training_summary.model.to(device)
    training_summary.model.eval()
    return training_summary, device


@torch.no_grad()
def samples_embedder(embedder:SamplesEmbedder, samples: NDArray[float], device) -> NDArray[float]:
    mu_embedder = lambda x: embedder.get_embedding(x).mu_t
    samples_t = torch.tensor(samples, device=device)
    return mu_embedder(samples_t).cpu().numpy()


@dataclass(frozen=True)
class TrainedModel:
    model: LsagneModel
    left_out_cloud: CmapCloudRef
    cv_clouds: list[CmapCloudRef]
    svm_with_post_training_results: AllCmapSvmEvaluationResults
    summary: TrainingSummary


def extract_svms_from_trained_model(
        training_summary_path: PathLike,
        perturbations_equivalence_sets: Collection[set[Perturbation]] | None = None
) -> TrainedModel:
    """
    Do svm training on the model original trained CMAP that is extracted
    """
    training_summary, device = load_training_data(training_summary_path)
    if perturbations_equivalence_sets is None:
        perturbations_equivalence_sets = training_summary.params.perturbations_equivalence_sets
    model = training_summary.model
    left_out_cloud = training_summary.params.left_out_cloud
    cv_clouds = training_summary.params.cross_validation_clouds
    datasets = training_summary.cmap_datasets
    splitted_evaluation_data = SplittedCmapEvaluationData.create_instance(
        training_only_cmap=datasets.training_only,
        training_concealed_cmap=datasets.training_concealed,
        model=model,
        cross_validation=datasets.cross_validation,
        left_out=datasets.left_out
    )
    anchor_points = training_summary.anchor_points
    original_space_anchor_points_lookup = AnchorPointsLookup.create_from_anchor_points(anchor_points, device)
    embedded_anchors_and_vectors = EmbeddedAnchorsAndVectors.create(
        original_space_anchor_points_lookup=original_space_anchor_points_lookup,
        embedder=model
    )
    all_svm_results = perform_svm_accuracy_evaluation(
        splitted_evaluation_data=splitted_evaluation_data,
        embedded_anchors_and_vectors=embedded_anchors_and_vectors,
        predicted_cloud_max_size=training_summary.params.predicted_cloud_max_size,
        random_seed=training_summary.params.random_manager.random_seed,
        # perturbations_equivalence_sets=training_summary.params.perturbations_equivalence_sets
        perturbations_equivalence_sets=perturbations_equivalence_sets
    )
    return TrainedModel(
        model=model,
        left_out_cloud=left_out_cloud,
        cv_clouds=cv_clouds,
        svm_with_post_training_results=all_svm_results,
        summary=training_summary
    )


def create_augmentation_db(
        data_augmentation_dir: Path,
        cmap_dir: Path,
        perturbations: list[Perturbation],
        output: Path,
        drug_batch_size=15
):
    args = AugmentationDbCreationParameters(
        raw_data_augmentation_dir=data_augmentation_dir,
        raw_cmap_dir=cmap_dir,
        min_drug_samples_per_cellline=6,
        min_cellines_perdrug=6,
        min_genes_per_go=4,
        max_genes_per_go=50,
        drug_batch_size=drug_batch_size,
        use_compression=False,
        calc_beta=False,
        output_dir=output,
        use_drugs=perturbations
    )
    generate_augmentation_db(args)
    return args


def generate_augmentation_parameters_template(
        data_augmentation_db: Path,
        num_of_samples: int,
        drug_batch_size = 15
) -> Generator[AugmentationGenerationParameters, None, None]:
    for n_pathways in [1,5,3,7]:
        for n_corrpathways in [0,3,5,7]:
            for n_genes in [5,7,10,15]:
                for n_corrgenes in [0,3,5,7]:
                    # for use_variance in ['perDrugMax', 'perDrugCelline']:
                    for use_variance in ['perDrugMax']:
                        yield AugmentationGenerationParameters(
                            data_augmentation_db_folder=data_augmentation_db,
                            drug_batch_size=drug_batch_size,
                            # tissue=cloud_ref.tissue,
                            # perturbation=cloud_ref.perturbation,
                            tissue=None,
                            perturbation=None,
                            num_of_samples=num_of_samples,
                            output=None,
                            n_pathways=n_pathways,
                            n_corrpathways=n_corrpathways,
                            proba_pathway=1.0,
                            n_genes=n_genes,
                            n_corrgenes=n_corrgenes,
                            proba_gene=1.0,
                            use_variance=use_variance,
                        )


def generate_augmented_samples(
        aug_gen_args: AugmentationGenerationParameters,
        cloud_ref: CmapCloudRef,
        samples: NDArray[float],
        size: int,
        random_seed: int
):
    augmentation_generator = AugmentationGenerator(
        logger=logger,
        clouds_to_augment=[cloud_ref],
        cloud_ref_to_augmentations={
            cloud_ref: generate_augmentations(aug_gen_args)[cloud_ref]
        },
        random_seed=random_seed
    )
    samples_to_augment = np.resize(samples, (size, samples.shape[1]))
    augmentation_generator.augment_samples_inplace(
        samples=samples_to_augment,
        cloud_refs=np.repeat(cloud_ref, size)
    )
    return samples_to_augment


def calculate_accuracy(
        results: NDArray[CmapCloudRef],
        truth: CmapCloudRef
) -> float:
    return (results == truth).mean()


def get_svm_results(
        trained_model: TrainedModel,
        samples: NDArray[float],
        cloud_ref: CmapCloudRef,
        perturbations_equivalence_sets: Collection[set[Perturbation]]
) -> SvmResults:
    embedded_samples = trained_model.model.get_embedding(
        torch.tensor(samples, device=trained_model.model.device)).z_t.cpu().numpy()
    svm_1_predictions = trained_model.svm_with_post_training_results.svm_1.svm.predict(embedded_samples)
    # svm_2_predictions = trained_model.svm_with_post_training_results.svm_2.svm.predict(embedded_samples)
    cmap_cloud = CmapCloud(
        cloud_ref=cloud_ref,
        tag=CmapCloudTag.AUGMENTED,
        samples=embedded_samples,
        cmap_ids=np.full(len(samples), 'AUGMENTED SAMPLE'))
    svm_1_results = CmapSvmCloudPredictionResults.create(
        cmap_cloud=cmap_cloud,
        svm_prediction=svm_1_predictions,
        perturbations_equivalence_sets=perturbations_equivalence_sets,
        calculate_confusion_df=False
    )
    # svm_2_results = CmapSvmCloudPredictionResults.create(
    #     cmap_cloud=cmap_cloud,
    #     svm_prediction=svm_2_predictions,
    #     perturbations_equivalence_sets=perturbations_equivalence_sets,
    #     calculate_confusion_df=False
    # )
    return SvmResults(
        svm_1=svm_1_results,
        # svm_2=svm_2_results
    )


def calculate_svm_on_cmap(
        cmap: RawCmapDataset,
        model_ref_to_trained_model: dict[ModelRef, TrainedModel],
        perturbations_equivalence_sets: Collection[set[Perturbation]]
) -> dict[ModelRef, dict[CmapCloudRef, SvmResults]]:
    res = defaultdict(dict)
    for model_ref, trained_model in model_ref_to_trained_model.items():
        for cloud_ref, samples in cmap.cloud_ref_to_samples.items():
            if cloud_ref.is_dmso_6h:
                continue
            res[model_ref][cloud_ref] = get_svm_results(trained_model, samples, cloud_ref, perturbations_equivalence_sets)
    return res


@torch.no_grad()
def main(
        trained_models: list[ModelRef],
        augmentation_size: int,
        reduction_sizes: list[int],
        random_seed_range: tuple[int, int],
        working_dir: Path,
        perturbations_equivalence_sets: Collection[set[Perturbation]],
        results_chunk_size: int
):
    logger.info('Loading models...')
    model_ref_to_model: dict[ModelRef, TrainedModel] = {}
    for model_ref in trained_models:
        logger.info(f'Loading model "{model_ref.path}"...')
        model_ref_to_model[model_ref] = extract_svms_from_trained_model(model_ref.path, perturbations_equivalence_sets)
    logger.info('All models loaded')

    experiment_root = Path(working_dir)
    cmaps_root = experiment_root / 'cmaps'
    aug_db_root = experiment_root / 'aug_db'
    full_cmap_dir = cmaps_root / 'full_cmap'
    output_dir = experiment_root / 'output'
    create_dir_if_not_exists(output_dir)

    if not full_cmap_dir.exists():
        logger.info('Creating full CMAP...')
        create_cmap_from_raw(
            raw_cmap_folder=raw_cmap_dir,
            output_cmap=full_cmap_dir
        )
    full_cmap = RawCmapDataset.load_dataset_from_disk(cmap_folder=full_cmap_dir)
    model_ref_to_cloud_ref_to_non_augmented_full_cmap_svm_results = calculate_svm_on_cmap(
        full_cmap,
        model_ref_to_model,
        perturbations_equivalence_sets
    )
    results_saver = DfResultsSaver(
        logger=logger,
        max_chunk_size=results_chunk_size,
        output_path=output_dir,
        file_name_prefix=f'results_rand_{random_seed_range[0]}-{random_seed_range[1]}'
    )

    # TODO: calculate all the options
    ALL_COMBINATIONS = (
        (len(full_cmap.perturbations_unique) - 2) * # DMSO + 24h
        len(reduction_sizes) *
        3 ** 4 * 2
    )
    counter = 1

    for p in full_cmap.perturbations_unique:
        if p.is_dmso_6h or p.is_dmso_24h:
            continue
        for reduction_size in reduction_sizes:
            for random_seed in range(*random_seed_range):
                experiment_name = f'pert={p}_newSize={reduction_size}_seed={random_seed}'
                logger.info(f'Starting experiment "{experiment_name}"')
                reduced_cmap_dir = cmaps_root / experiment_name
                logger.info(f'Creating reduced CMAP in "{reduced_cmap_dir}"')
                reduced_cmap = load_and_reduce_cmap(
                    logger=logger,
                    full_cmap_folder=full_cmap_dir,
                    reduced_cmap_folder=reduced_cmap_dir,
                    perturbation_to_resize={
                        p: reduction_size
                    },
                    random_seed=random_seed
                )
                model_ref_to_cloud_ref_to_non_augmented_reduced_cmap_svm_results = calculate_svm_on_cmap(reduced_cmap, model_ref_to_model, perturbations_equivalence_sets)
                aug_db_dir = aug_db_root / experiment_name
                logger.info(f'Creating aug DB in "{aug_db_dir}"')
                augmentation_db_args = create_augmentation_db(
                    data_augmentation_dir=raw_data_augmentation_dir,
                    cmap_dir=reduced_cmap_dir,
                    perturbations=[p],
                    output=aug_db_dir
                )
                for aug_gen_parameters_template in generate_augmentation_parameters_template(
                        data_augmentation_db=aug_db_dir,
                        num_of_samples=augmentation_size
                ):
                    logger.info(f'Generating {aug_gen_parameters_template.num_of_samples:,} samples using n_pathways ; n_corrpathways={aug_gen_parameters_template.n_corrpathways} ; proba_pathway={aug_gen_parameters_template.proba_pathway} ; n_genes={aug_gen_parameters_template.n_genes} ; n_corrgenes={aug_gen_parameters_template.n_corrgenes} ; proba_gene={aug_gen_parameters_template.proba_gene} ; use_variance={aug_gen_parameters_template.use_variance}')
                    logger.info(f'### main counter: {counter:,} / {ALL_COMBINATIONS:,}')
                    counter += 1
                    for cloud_ref in full_cmap.perturbation_to_cloud_refs[p]:
                        aug_gen_args = dataclasses.replace(
                            aug_gen_parameters_template,
                            tissue=cloud_ref.tissue,
                            perturbation=cloud_ref.perturbation
                        )
                        samples = generate_augmented_samples(
                            aug_gen_args=aug_gen_args,
                            cloud_ref=cloud_ref,
                            samples=reduced_cmap.cloud_ref_to_samples[cloud_ref],
                            size=augmentation_size,
                            random_seed=random_seed
                        )
                        logger.info(f'Generated samples for cloud {cloud_ref}')
                        for model_ref, trained_model in model_ref_to_model.items():
                            if cloud_ref in [trained_model.left_out_cloud, *trained_model.cv_clouds]:
                                logger.info(f'Skipping evaluating results for model "{model_ref.alias}"')
                                continue
                            logger.info(f'Evaluating results for model "{model_ref.alias}"')
                            experiment_results = ExperimentResult(
                                model_ref=model_ref,
                                cloud_ref=cloud_ref,
                                n_pathways=aug_gen_parameters_template.n_pathways,
                                n_corrpathways=aug_gen_parameters_template.n_corrpathways,
                                n_genes=aug_gen_parameters_template.n_genes,
                                n_corrgenes=aug_gen_parameters_template.n_corrgenes,
                                use_variance=aug_gen_parameters_template.use_variance,
                                aug_gen_drug_batch_size=aug_gen_parameters_template.drug_batch_size,
                                reduction_size=reduction_size,
                                random_seed=random_seed,
                                aug_db_cmap_dir=augmentation_db_args.raw_cmap_dir,
                                min_drug_samples_per_cellline=augmentation_db_args.min_drug_samples_per_cellline,
                                min_cellines_perdrug=augmentation_db_args.min_cellines_perdrug,
                                min_genes_per_go=augmentation_db_args.min_genes_per_go,
                                max_genes_per_go=augmentation_db_args.max_genes_per_go,
                                aug_db_drug_batch_size=augmentation_db_args.drug_batch_size,
                                aug_db_use_compression=augmentation_db_args.use_compression,
                                calc_beta=augmentation_db_args.calc_beta,
                                aug_db_dir=augmentation_db_args.output_dir,
                                full_cmap_svm_results= model_ref_to_cloud_ref_to_non_augmented_full_cmap_svm_results[model_ref][cloud_ref],
                                reduced_cmap_svm_results=model_ref_to_cloud_ref_to_non_augmented_reduced_cmap_svm_results[model_ref][cloud_ref],
                                augmented_svm_results=get_svm_results(trained_model, samples, cloud_ref, perturbations_equivalence_sets),
                            )
                            results_saver.add_result(experiment_results.to_dict())
        # break
    results_saver.flush()
    logger.info('Done!')


def parse_args():
    parser = argparse.ArgumentParser(description="Get running mode argument.")
    parser.add_argument(
        "running_mode",
        choices=["local", "server", "slurm"],
        help="Running mode (local, server, or slurm)",
        default="local"
    )
    parser.add_argument(
        "random_seed_range",
        type=str,
        help="Random seed range, e.g. '1-100'",
        default="0-99"
    )
    parser.add_argument(
        "working_dir",
        type=Path,
        help="Location of the working dir",
        # required=True
    )
    parser.add_argument(
        "augmentation_size",
        type=int,
        help="Augmentation size",
    )
    parser.add_argument(
        "result_chunk_size",
        type=int,
        help="Result chunk size",
    )
    args = parser.parse_args()
    return args.running_mode, tuple(int(r) for r in args.random_seed_range.split('-')), args.working_dir, args.augmentation_size, args.result_chunk_size


if __name__ == '__main__':
    # df = pd.read_hdf('/Users/emil/MSc/lsagne-1/output/aug_test_params_root/results.h5', key='df')

    logger: Logger = create_logger()
    running_mode, random_seed_range, working_dir, augmentation_size, result_chunk_size = parse_args()

    match running_mode:
        case "local":
            raw_data_augmentation_dir = Path('/Users/emil/MSc/lsagne-1/raw_data_folder/data_augmentation')
            raw_cmap_dir = Path('/Users/emil/MSc/lsagne-1/raw_data_folder/cmap')
            reduction_sizes = [6]
            model_paths = [
                ModelRef(
                    path='/Users/emil/MSc/lsagne-1/output/experiments/exp_4/exp_4_modules/A549_wortmannin_1_2_experiment_4_baseline-0-A549-wortmannin/_cache/train basic model',
                    alias='model_1'
                ),
                ModelRef(
                    path='/Users/emil/MSc/lsagne-1/output/experiments/exp_4/exp_4_modules/A549_wortmannin_1_2_experiment_4_baseline-0-A549-wortmannin/_cache/train basic model',
                    alias='model_2'
                )
            ]
        case "server":
            raw_data_augmentation_dir = Path('/RG/compbio/emil/data_augmentation')
            raw_cmap_dir = Path('/RG/compbio/emil/cmap')
            reduction_sizes = [6, 12, 25, 50, 100]
            # reduction_sizes = [6]
            model_paths = [
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_A375_geldanamycin_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-2-A375-geldanamycin/_cache/train basic model',
                    alias='model_3'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_A375_raloxifene_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-1-A375-raloxifene/_cache/train basic model',
                    alias='model_8'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_A549_isonicotinohydroxamic-acid_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-3-A549-isonicotinohydroxamic-acid/_cache/train basic model',
                    alias='model_25'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_A549_trichostatin-a_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-1-A549-trichostatin-a/_cache/train basic model',
                    alias='model_32'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_HA1E_trichostatin-a_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-1-HA1E-trichostatin-a/_cache/train basic model',
                    alias='model_53'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_HA1E_vorinostat_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-3-HA1E-vorinostat/_cache/train basic model',
                    alias='model_58'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_HEPG2_geldanamycin_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-1-HEPG2-geldanamycin/_cache/train basic model',
                    alias='model_80'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_HEPG2_raloxifene_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-2-HEPG2-raloxifene/_cache/train basic model',
                    alias='model_84'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_HT29_trichostatin-a_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-3-HT29-trichostatin-a/_cache/train basic model',
                    alias='model_103'
                ),
                ModelRef(
                    path='/RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/test_0_baseline_MCF7_trichostatin-a_1_3_MEL_SBTCH_epchs5000_WRM300_RSLCT4300_dim20_kld0.01_CLS5000-2-MCF7-trichostatin-a/_cache/train basic model',
                    alias='model_120'
                ),
            ]
        case _:
            raise RuntimeError('Bug in parser')

    main(
        trained_models=model_paths,
        augmentation_size=augmentation_size,
        reduction_sizes=reduction_sizes,
        random_seed_range=random_seed_range,
        working_dir=working_dir,
        perturbations_equivalence_sets=[
            {
                Perturbation("isonicotinohydroxamic-acid"),
                Perturbation("raloxifene")
            },
            {
                Perturbation("vorinostat"),
                Perturbation("trichostatin-a")
            }
        ],
        results_chunk_size=result_chunk_size
    )
