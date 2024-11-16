import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Collection, Sized

import numpy as np
import torch
from numpy.typing import NDArray

from src.assertion_utils import assert_same_len
from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.embedded_anchors_and_vectors import EmbeddedAnchorsAndVectors
from src.multi_dim_arithmetics import calculate_nearest_points_on_2_lines_np, calculate_nearest_points_on_2_lines_t
from src.utils import take_closest_points_to_reference, TensorOrNumpyType, calculate_cloud_ref_to_samples


@dataclass(frozen=True)
class PredictedCloud(Sized):
    cloud_ref: CmapCloudRef
    predicted_z: NDArray[float]
    dmso_6h_reference_z: NDArray[float]
    dmso_24h_reference_z: NDArray[float]
    closest_dmso_6h_points_z: NDArray[float]
    closest_dmso_24h_points_z: NDArray[float]
    anchor_treatment_vector_z: NDArray[float]
    anchor_drug_vector_z: Optional[NDArray[float]]

    def __post_init__(self):
        assert_same_len(self.closest_dmso_6h_points_z, self.closest_dmso_24h_points_z, self.predicted_z)

    def __len__(self):
        return len(self.predicted_z)


def predict_triangle_cloud(
        batch_tissue_untreated_6h_samples: TensorOrNumpyType,
        batch_tissue_untreated_24h_samples: TensorOrNumpyType,
        treatment_vector: TensorOrNumpyType,
        drug_vector: TensorOrNumpyType
) -> Tuple[TensorOrNumpyType, TensorOrNumpyType]:
    min_len = min(len(batch_tissue_untreated_24h_samples), len(batch_tissue_untreated_6h_samples))
    drug_vectors_line_x0 = batch_tissue_untreated_24h_samples[:min_len]
    drug_vectors_line_x = drug_vector

    treatment_vectors_line_x0 = batch_tissue_untreated_6h_samples[:min_len]
    treatment_vectors_line_x = treatment_vector

    if isinstance(batch_tissue_untreated_6h_samples, torch.Tensor):
        calculate_nearest_points_on_2_lines = calculate_nearest_points_on_2_lines_t
    elif isinstance(batch_tissue_untreated_6h_samples, np.ndarray):
        calculate_nearest_points_on_2_lines = calculate_nearest_points_on_2_lines_np
    else:
        raise TypeError(f'Unsupported type passed! {type(batch_tissue_untreated_6h_samples)}')

    p1, p2 = calculate_nearest_points_on_2_lines(
        l1=(drug_vectors_line_x0, drug_vectors_line_x),
        l2=(treatment_vectors_line_x0, treatment_vectors_line_x)
    )

    return p1, p2


@torch.no_grad()
def predicted_cloud_calculator(
        cloud_ref: CmapCloudRef,
        dmso_6h_cloud_z: NDArray[float],
        dmso_24h_cloud_z: NDArray[float],
        embedded_anchors_and_vectors: EmbeddedAnchorsAndVectors,
        predicted_cloud_max_size: Optional[int] = None,
        ) -> PredictedCloud:

    dmso_6h_reference_z = embedded_anchors_and_vectors.cloud_ref_to_cloud_center[cloud_ref.dmso_6h].cpu().numpy()
    dmso_24h_reference_z = embedded_anchors_and_vectors.cloud_ref_to_cloud_center[cloud_ref.dmso_24h].cpu().numpy()
    predicted_cloud_max_size = math.inf if predicted_cloud_max_size is None else predicted_cloud_max_size
    predicted_cloud_size = min(predicted_cloud_max_size, len(dmso_6h_cloud_z), len(dmso_24h_cloud_z))

    closest_samples_to_dmso_6h_reference_z = take_closest_points_to_reference(
        all_points=dmso_6h_cloud_z,
        reference=dmso_6h_reference_z,
        points_count=predicted_cloud_size
    )
    closest_samples_to_dmso_24h_reference_z = take_closest_points_to_reference(
        all_points=dmso_24h_cloud_z,
        reference=dmso_24h_reference_z,
        points_count=predicted_cloud_size
    )
    anchor_treatment_vector_z = embedded_anchors_and_vectors.perturbation_to_anchor_treatment_vector[cloud_ref.perturbation].cpu().numpy()
    anchor_drug_vector_z = embedded_anchors_and_vectors.perturbation_to_anchor_drug_vector[cloud_ref.perturbation].cpu().numpy()
    closest_to_lines_tuple = predict_triangle_cloud(
        batch_tissue_untreated_6h_samples=closest_samples_to_dmso_6h_reference_z,
        batch_tissue_untreated_24h_samples=closest_samples_to_dmso_24h_reference_z,
        treatment_vector=anchor_treatment_vector_z,
        drug_vector=anchor_drug_vector_z
    )
    predicted_z = np.mean(closest_to_lines_tuple, axis=0)

    return PredictedCloud(
        cloud_ref=cloud_ref,
        predicted_z=predicted_z,
        dmso_6h_reference_z=dmso_6h_reference_z,
        dmso_24h_reference_z=dmso_24h_reference_z,
        closest_dmso_6h_points_z=closest_samples_to_dmso_6h_reference_z,
        closest_dmso_24h_points_z=closest_samples_to_dmso_24h_reference_z,
        anchor_treatment_vector_z=anchor_treatment_vector_z,
        anchor_drug_vector_z=anchor_drug_vector_z,
    )


@torch.no_grad()
def predicted_clouds_calculator(
        all_cloud_refs: Collection[CmapCloudRef],
        training_cmap_dataset: RawCmapDataset,
        embedded_anchors_and_vectors: EmbeddedAnchorsAndVectors,
        training_z: NDArray[float],
        predicted_cloud_max_size: Optional[int] = None,
        cloud_ref_to_skip: Collection[CmapCloudRef] = None
        ) -> Dict[CmapCloudRef, PredictedCloud]:
    """
    This method calculates all the non-DMSO clouds given in all_cloud_refs.
    We assume that for each such cloud, a DMSO-6H and DMSO-24H are present in the training set.
    During prediction, if the cloud is trained, we take the reference vectors, if not, we calculate a new cloud center
    :param all_cloud_refs:
    :param training_cmap_dataset:
    :param model:
    :param embedded_anchors_and_vectors:
    :param training_z:
    :param predicted_cloud_max_size:
    :param non_trained_cloud_refs:
    :param non_trained_cloud_ref_to_z:
    :param cloud_ref_to_skip:
    :return:
    """
    if cloud_ref_to_skip is None:
        cloud_ref_to_skip = set()

    training_cloud_ref_to_samples = calculate_cloud_ref_to_samples(training_cmap_dataset, training_z)

    cloud_ref_to_predicted_cloud: Dict[CmapCloudRef, PredictedCloud] = {}

    for cloud_ref in all_cloud_refs:
        if cloud_ref in cloud_ref_to_skip or cloud_ref.is_dmso_6h or cloud_ref.is_dmso_24h:
            continue

        cloud_ref_to_predicted_cloud[cloud_ref] = predicted_cloud_calculator(
            cloud_ref=cloud_ref,
            dmso_6h_cloud_z=training_cloud_ref_to_samples[cloud_ref.dmso_6h],
            dmso_24h_cloud_z=training_cloud_ref_to_samples[cloud_ref.dmso_24h],
            embedded_anchors_and_vectors=embedded_anchors_and_vectors,
            predicted_cloud_max_size=predicted_cloud_max_size
        )

    return cloud_ref_to_predicted_cloud
