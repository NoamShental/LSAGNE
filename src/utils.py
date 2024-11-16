import time
from functools import wraps
from typing import Dict, TypeVar, Optional, Collection, Set

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from src.assertion_utils import assert_same_len
from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.lookup_tensor import LookupTensor
from src.perturbation import Perturbation

TensorOrNumpyType = TypeVar('TensorOrNumpyType', Tensor, np.ndarray)


def check_if_in_same_perturbations_equivalence_set(
        cloud_ref_1: CmapCloudRef,
        cloud_ref_2: CmapCloudRef,
        perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]]
) -> bool:
    if cloud_ref_1 == cloud_ref_2:
        return True
    if not perturbations_equivalence_sets or cloud_ref_1.tissue != cloud_ref_2.tissue:
        return False
    # At this point:
    # 1. The cloud refs are not the same
    # 2. The tissue is the same
    # 3. At least one equivalence set is defined
    for perturbations_equivalence_set in perturbations_equivalence_sets:
        if cloud_ref_1.perturbation in perturbations_equivalence_set and \
                cloud_ref_2.perturbation in perturbations_equivalence_set:
            return True
    return False


def calculate_cloud_ref_to_samples(cmap: RawCmapDataset, samples: TensorOrNumpyType) -> Dict[CmapCloudRef, TensorOrNumpyType]:
    assert_same_len(cmap, samples)
    return {
        cloud_ref: samples[cmap.cloud_ref_to_idx[cloud_ref]] for cloud_ref in cmap.unique_cloud_refs
    }


def calculate_non_trained_reference_vector(
        trained_cloud_ref_to_reference_vector: LookupTensor[CmapCloudRef],
        non_trained_cloud_ref: CmapCloudRef
) -> NDArray[float]:
    """
    Calculates the predicted reference vector for non-trained node.
    This is especially useful in order to calculate alpha model prediction.
    :param trained_cloud_ref_to_reference_vector: All the known reference vectors, might even include the reference
    vector for the current non_trained_cloud_ref.
    :param non_trained_cloud_ref: The cloud to remove (if exists) from the trained_cloud_ref_to_reference_vector
    and use other reference vectors in order to calculate the predicted reference vector for.
    :return: Calculates predicted reference vector for non_trained_cloud_ref using other reference vectors
    taken from trained_cloud_ref_to_reference_vector.
    """
    other_tissues_treatment_vectors = np.array([
        reference_vector_t.cpu().numpy()
        for cloud_ref, reference_vector_t in trained_cloud_ref_to_reference_vector.items()
        if non_trained_cloud_ref.perturbation == cloud_ref.perturbation and
           cloud_ref.tissue != non_trained_cloud_ref.tissue
    ])
    mean_vector = np.mean(other_tissues_treatment_vectors, axis=0)
    vector_used_for_prediction_direction = mean_vector / np.linalg.norm(mean_vector, 2)
    vector_used_for_prediction_magnitude = np.linalg.norm(other_tissues_treatment_vectors, 2,
                                                                    axis=1).mean()
    return vector_used_for_prediction_direction * vector_used_for_prediction_magnitude


def take_closest_points_to_reference(
        all_points: NDArray[float],
        reference: NDArray[float],
        points_count: int
) -> NDArray[float]:
    """
    Take points_count closest points from all_points which are the nearest to reference point
    """
    points_distance_from_reference: NDArray[float] = np.linalg.norm(all_points - reference, axis=1)
    partitioned_points_idx = np.argpartition(points_distance_from_reference, points_count - 1)
    selected_idx = partitioned_points_idx[:points_count]
    assert all(points_distance_from_reference[partitioned_points_idx[points_count:]] >= max(points_distance_from_reference[selected_idx]))
    closest_points = all_points[selected_idx]
    return closest_points


def timeit(func):
    @wraps(func)  # Preserves function metadata
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # High-precision timer
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper

