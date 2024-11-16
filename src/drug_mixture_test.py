import pickle
from dataclasses import dataclass
from itertools import combinations
from logging import Logger
from typing import List, Dict, Union, Iterable, Callable, Optional, Set

import torch
from numpy.typing import NDArray

from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_clouds_utils import find_center_of_cloud_np
from src.configuration import config
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.logger_utils import create_logger
from src.models.dudi_basic.embedded_anchors_and_vectors import EmbeddedAnchorsAndVectors
from src.models.dudi_basic.multi_device_data import AnchorPointsLookup
from src.models.predicted_clouds_calculator import predicted_cloud_calculator, PredictedCloud
from src.perturbation import Perturbation
from src.pipeline_utils import choose_device
from src.tissue import Tissue
from src.training_summary import TrainingSummary


@dataclass(frozen=True)
class DrugMixtureDoTriangleStage:
    perturbation: Perturbation
    start_samples: NDArray[float]
    samples_dmso_6h: NDArray[float]
    samples_dmso_24h: NDArray[float]
    predicted: PredictedCloud


@dataclass(frozen=True)
class DrugMixtureAddDrugStage:
    perturbation: Perturbation
    start_samples: NDArray[float]
    predicted: NDArray[float]
    drug_vector: NDArray[float]


@dataclass(frozen=True)
class DrugMixtureResult:
    cloud_ref: CmapCloudRef
    mix: Iterable[Perturbation]
    add_24h: bool
    stages: List[Union[DrugMixtureDoTriangleStage, DrugMixtureAddDrugStage]]


def do_test(
        logger: Logger,
        drug_mixture_result: DrugMixtureResult,
        decode_to_original_space: Callable[[NDArray[float]], NDArray[float]]
):
    logger.info(f"Michael is doing some weird experiment on "
                f"cloud={drug_mixture_result.cloud_ref} with mix {drug_mixture_result.mix} :)")
    decoded_samples = decode_to_original_space(drug_mixture_result.stages[-1].predicted)
    pass


@torch.no_grad()
def drug_mixture_test(
    add_24h: bool,
    model_path: str,
    clouds: List[CmapCloudRef],
    perturbations: List[Perturbation],
    mix_size: int,
    test_cmap: RawCmapDataset,
    aux_cmap: Optional[RawCmapDataset],
    aux_tissue_to_trained_tissue: Optional[Dict[str, str]]
):
    if aux_tissue_to_trained_tissue:
        assert aux_cmap
    logger: Logger = create_logger()
    logger.info(f'Starting drug mixture test...')

    logger.info(f'Loading model {model_path}...')
    with open(model_path, "rb") as file:
        training_summary: TrainingSummary = pickle.load(file)
    device = choose_device(training_summary.params.use_cuda, logger)
    torch.set_default_dtype(config.torch_numeric_precision_type)
    # Necessary when the task result is loaded from pickled file
    training_summary.model.load_state_dict(training_summary.model_state_dict)
    training_summary.model.to(device)
    model = training_summary.model
    model.eval()

    logger.info('Embedding all the samples...')
    all_perturbations = set(perturbations).union({Perturbation.DMSO_6H(), Perturbation.TIME_24H()})
    cloud_ref_to_encoded_samples: Dict[CmapCloudRef, NDArray[float]] = {}
    clouds_to_encode: Set[CmapCloudRef] = set()
    for cloud_ref in clouds:
        if cloud_ref not in test_cmap.unique_cloud_refs and aux_cmap and cloud_ref not in aux_cmap.unique_cloud_refs:
            raise AssertionError(f'{cloud_ref} not in the given CMAPs')
        clouds_to_encode.add(cloud_ref)
        for perturbation in all_perturbations:
            if aux_tissue_to_trained_tissue and cloud_ref.tissue in aux_tissue_to_trained_tissue:
                cloud_ref_ = CmapCloudRef(aux_tissue_to_trained_tissue[cloud_ref.tissue], perturbation)
            else:
                cloud_ref_ = cloud_ref.change_perturbation(perturbation)
            clouds_to_encode.add(cloud_ref_)
    for cloud_ref in clouds_to_encode:
            if cloud_ref in cloud_ref_to_encoded_samples \
                    or cloud_ref not in test_cmap.unique_cloud_refs \
                    and cloud_ref not in aux_cmap.unique_cloud_refs:
                continue
            if cloud_ref in test_cmap.unique_cloud_refs:
                samples = test_cmap.cloud_ref_to_samples[cloud_ref]
            else:
                samples = aux_cmap.cloud_ref_to_samples[cloud_ref]
            samples_t = torch.tensor(samples, device=device)
            cloud_ref_to_encoded_samples[cloud_ref] = model.get_embedding(samples_t).z_t.cpu().numpy()

    perturbation_mix = list(combinations(perturbations, mix_size))

    on_device_original_space_anchor_points: AnchorPointsLookup = AnchorPointsLookup.create_from_anchor_points(
        training_summary.anchor_points,
        device
    )
    embedded_anchors_and_vectors: EmbeddedAnchorsAndVectors = EmbeddedAnchorsAndVectors.create(
        original_space_anchor_points_lookup=on_device_original_space_anchor_points,
        embedder=model
    )

    cloud_ref_to_predicted_drug_vector: Dict[CmapCloudRef, NDArray[float]] = {}

    for cloud_ref in clouds:
        if cloud_ref.is_dmso_6h_or_24h:
            logger.warning(f'Cloud {cloud_ref} is dmso 6h or 24h, skipping...')
            continue
        for mix in perturbation_mix:
            if Perturbation.DMSO_6H() in mix or Perturbation.TIME_24H() in mix:
                logger.warning(f'Mix {mix} contains dmso 6h or 24h, skipping...')
                continue
            mixture_result = DrugMixtureResult(
                cloud_ref=cloud_ref,
                mix=mix,
                add_24h=add_24h,
                stages=[]
            )
            current_samples = cloud_ref_to_encoded_samples[cloud_ref]
            for i, perturbation in enumerate(mix):
                cloud_ref_with_current_perturbation = cloud_ref.change_perturbation(perturbation)
                if aux_tissue_to_trained_tissue and cloud_ref.tissue in aux_tissue_to_trained_tissue:
                    cloud_ref_with_current_perturbation = CmapCloudRef(
                        aux_tissue_to_trained_tissue[cloud_ref.tissue],
                        cloud_ref_with_current_perturbation.perturbation
                    )
                if cloud_ref_with_current_perturbation not in cloud_ref_to_predicted_drug_vector:
                    triangle_prediction = predicted_cloud_calculator(
                        cloud_ref=cloud_ref_with_current_perturbation,
                        dmso_6h_cloud_z=cloud_ref_to_encoded_samples[cloud_ref_with_current_perturbation.dmso_6h],
                        dmso_24h_cloud_z=cloud_ref_to_encoded_samples[cloud_ref_with_current_perturbation.dmso_24h],
                        embedded_anchors_and_vectors=embedded_anchors_and_vectors
                    )
                    predicted_cloud_reference = find_center_of_cloud_np(triangle_prediction.predicted_z)
                    cloud_ref_to_predicted_drug_vector[cloud_ref_with_current_perturbation] = \
                        predicted_cloud_reference - triangle_prediction.dmso_24h_reference_z

                perturbation_drug_vector = cloud_ref_to_predicted_drug_vector[cloud_ref_with_current_perturbation]
                predicted_samples = current_samples + perturbation_drug_vector
                mixture_result.stages.append(
                    DrugMixtureAddDrugStage(
                        perturbation=perturbation,
                        start_samples=current_samples,
                        predicted=predicted_samples,
                        drug_vector=perturbation_drug_vector
                    )
                )
                current_samples = predicted_samples
            if add_24h:
                if aux_tissue_to_trained_tissue and cloud_ref.tissue in aux_tissue_to_trained_tissue:
                    cloud_ref_ = CmapCloudRef(aux_tissue_to_trained_tissue[cloud_ref.tissue], cloud_ref.perturbation)
                else:
                    cloud_ref_ = cloud_ref
                dmso_6h_center = embedded_anchors_and_vectors.cloud_ref_to_cloud_center[cloud_ref_.dmso_6h].cpu().numpy()
                dmso_24h_center = embedded_anchors_and_vectors.cloud_ref_to_cloud_center[cloud_ref_.dmso_24h].cpu().numpy()
                dmso_6h_to_dmso_24_reference = dmso_24h_center - dmso_6h_center
                predicted_samples = current_samples + dmso_6h_to_dmso_24_reference
                mixture_result.stages.append(
                    DrugMixtureAddDrugStage(
                        perturbation=Perturbation.TIME_24H(),
                        start_samples=current_samples,
                        predicted=predicted_samples,
                        drug_vector=dmso_6h_to_dmso_24_reference
                    )
                )

            do_test(
                logger,
                mixture_result,
                lambda samples: model.vae.decode(torch.tensor(samples, device=device)).cpu().numpy()
            )
