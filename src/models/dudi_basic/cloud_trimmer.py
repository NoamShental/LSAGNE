from typing import Dict

import torch
from torch import Tensor
import pandas as pd

from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset


@torch.no_grad()
def trim_clouds(
        cmap: RawCmapDataset,
        cloud_ref_to_distances_from_radius_t: Dict[CmapCloudRef, Tensor],
        request_trimming_cloud_ref_to_keep_ratio: Dict[CmapCloudRef, float]) -> RawCmapDataset:
    if len(cmap) == 0:
        return cmap

    data_df_after_filter_before_stack = []
    info_df_after_filter_before_stack = []

    for cloud_ref, distances_from_radius_t in cloud_ref_to_distances_from_radius_t.items():
        cmap_mask = cmap.cloud_ref_to_idx_mask[cloud_ref]
        assert cmap_mask.sum() == len(distances_from_radius_t)
        if cloud_ref in request_trimming_cloud_ref_to_keep_ratio:
            ratio_to_keep = request_trimming_cloud_ref_to_keep_ratio[cloud_ref]
            cloud_size = len(distances_from_radius_t)
            # Take the top cloud_size * ratio_to_keep samples from the current cloud
            close_enough = torch.topk(distances_from_radius_t, int(cloud_size * ratio_to_keep), largest=False)
            idx_to_take = close_enough.indices.cpu()
            data_df_after_filter_before_stack.append(cmap.data_df[cmap_mask].iloc[idx_to_take])
            info_df_after_filter_before_stack.append(cmap.info_df[cmap_mask].iloc[idx_to_take])
        else:
            data_df_after_filter_before_stack.append(cmap.data_df[cmap_mask])
            info_df_after_filter_before_stack.append(cmap.info_df[cmap_mask])

    new_data_df = pd.concat(data_df_after_filter_before_stack)
    new_info_df = pd.concat(info_df_after_filter_before_stack)

    # TODO handle the scaler
    return RawCmapDataset(
        data_df=new_data_df,
        info_df=new_info_df,
        scaler=cmap.scaler
    )
