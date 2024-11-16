from typing import Tuple

import torch
from numpy import ndarray
from scipy.spatial.distance import cdist
import numpy as np
from torch import Tensor


def find_center_of_cloud_np(cloud_points: ndarray) -> ndarray:
    cloud_l2_dists = cdist(cloud_points, cloud_points)
    mean_distances = np.mean(cloud_l2_dists, axis=1)
    center_idx = np.argmin(mean_distances)
    return cloud_points[center_idx]


def find_center_of_cloud_t(cloud_points_t: Tensor) -> Tuple[Tensor, Tensor]:
    center_idx_t = find_idx_for_center_of_cloud_t(cloud_points_t)
    return cloud_points_t[center_idx_t], center_idx_t


# torch.combinations(torch.arange(10), 2)
def find_idx_for_center_of_cloud_t(cloud_points_t: Tensor) -> Tensor:
    cloud_l2_dists_t = torch.cdist(cloud_points_t, cloud_points_t)
    mean_distances_t = torch.mean(cloud_l2_dists_t, dim=1)
    center_idx_t = torch.argmin(mean_distances_t)
    return center_idx_t
