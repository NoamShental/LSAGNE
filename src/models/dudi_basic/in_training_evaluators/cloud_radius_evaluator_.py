from typing import Dict, Tuple

import pandas as pd
import torch
from torch import Tensor

from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.lookup_tensor import LookupTensor


@torch.no_grad()
def evaluate_cloud_radii(
        cmap: RawCmapDataset,
        z_t: Tensor,
        cloud_ref_to_cloud_center_t: LookupTensor[CmapCloudRef]
) -> Tuple[pd.DataFrame, Dict[CmapCloudRef, Tensor]]:
    """
    Calculates the L2 distances of each sample from it's cloud center reference sample, a.k.a. radius.
    Calculates the max, min, mean, median of all the radii of each cloud.
    """
    cloud_ref_to_distances_from_radius_t: Dict[CmapCloudRef, Tensor] = {}
    original_label = []
    display_name = []
    tissue = []
    perturbation = []
    max = []
    min = []
    mean = []
    std = []
    median = []
    q_5 = []
    q_25 = []
    q_75 = []
    q_95 = []
    cloud_size = []
    for cloud_ref in cmap.unique_cloud_refs:
        cloud_samples_z_t = z_t[cmap.cloud_ref_to_idx[cloud_ref]]
        cloud_center_reference_z_t = cloud_ref_to_cloud_center_t[cloud_ref]
        dist_to_center_t = torch.linalg.norm(cloud_samples_z_t - cloud_center_reference_z_t, 2, dim=1)
        cloud_ref_to_distances_from_radius_t[cloud_ref] = dist_to_center_t
        cloud_size.append(len(cloud_samples_z_t))
        original_label.append(cmap.cloud_ref_to_original_label[cloud_ref])
        display_name.append(cmap.cloud_ref_to_display_name[cloud_ref])
        tissue.append(cloud_ref.tissue)
        perturbation.append(cloud_ref.perturbation)
        std_, mean_ = torch.std_mean(dist_to_center_t)
        max.append(float(dist_to_center_t.max().cpu()))
        min.append(float(dist_to_center_t.min().cpu()))
        std.append(float(std_.cpu()))
        mean.append(float(mean_.cpu()))
        median.append(float(dist_to_center_t.median().cpu()))
        q_5.append(float(torch.quantile(dist_to_center_t, 0.05).cpu()))
        q_25.append(float(torch.quantile(dist_to_center_t, 0.25).cpu()))
        q_75.append(float(torch.quantile(dist_to_center_t, 0.75).cpu()))
        q_95.append(float(torch.quantile(dist_to_center_t, 0.95).cpu()))
    clouds_radius_df = pd.DataFrame({
        'original_label': original_label,
        'display_name': display_name,
        'tissue': tissue,
        'perturbation': perturbation,
        'cloud_size': cloud_size,
        'std': std,
        'min': min,
        'q_5': q_5,
        'q_25': q_25,
        'mean': mean,
        'median': median,
        'q_75': q_75,
        'q_95': q_95,
        'max': max
    })
    clouds_radius_df.sort_values(by='median', ascending=False, inplace=True)

    return clouds_radius_df, cloud_ref_to_distances_from_radius_t
