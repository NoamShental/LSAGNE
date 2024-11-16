import os
from itertools import combinations_with_replacement
from logging import Logger

import pandas as pd
import torch
from torch import Tensor

from src.assertion_utils import assert_same_len
from src.dataframe_utils import df_to_str
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.os_utilities import create_dir_if_not_exists
from src.write_to_file_utils import write_str_to_file


class CloudsDistancesEvaluator:
    def __init__(self, logger: Logger, working_directory: str, sub_folder: str):
        self.logger = logger
        self.directory_path = os.path.join(working_directory, sub_folder)
        create_dir_if_not_exists(self.directory_path)

    def evaluate(self, cmap: RawCmapDataset, z_t: Tensor, i_epoch: int):
        assert_same_len(cmap, z_t)
        self.logger.info(f'Performing {self.__class__.__name__}')
        display_name_1 = []
        display_name_2 = []
        tissue_1 = []
        tissue_2 = []
        mean_distances = []
        max_distances = []
        min_distances = []
        median_distances = []

        cloud_pairs = combinations_with_replacement(cmap.unique_cloud_refs, 2)
        cloud_pairs = list(cloud_pairs)
        self.logger.debug(f'There are {len(cloud_pairs):,} pairs')
        for i, (cloud_ref_1, cloud_ref_2 )in enumerate(cloud_pairs):
            if i % 500 == 0 and i > 0:
                self.logger.debug(f'Finished {i} / {len(cloud_pairs):,}')
            if cloud_ref_1 == cloud_ref_2:
                idx = cmap.cloud_refs == cloud_ref_1
                relevant_distance_matrix_entries = torch.pdist(z_t[idx])
            else:
                idx_1_t = cmap.cloud_refs == cloud_ref_1
                idx_2_t = cmap.cloud_refs == cloud_ref_2
                relevant_distance_matrix_entries = torch.cdist(z_t[idx_1_t], z_t[idx_2_t])

            display_name_1.append(cmap.cloud_ref_to_display_name[cloud_ref_1])
            display_name_2.append(cmap.cloud_ref_to_display_name[cloud_ref_2])
            tissue_1.append(cloud_ref_1.tissue)
            tissue_2.append(cloud_ref_2.tissue)
            mean_distances.append(float(relevant_distance_matrix_entries.mean().cpu()))
            max_distances.append(float(relevant_distance_matrix_entries.max().cpu()))
            min_distances.append(float(relevant_distance_matrix_entries.min().cpu()))
            median_distances.append(float(relevant_distance_matrix_entries.median().cpu()))
        clouds_distances_df = pd.DataFrame({
            'display_name_1': display_name_1,
            'tissue_1': tissue_1,
            'display_name_2': display_name_2,
            'tissue_2': tissue_2,
            'mean_distance': mean_distances,
            'max_distance': max_distances,
            'min_distance': min_distances,
            'median_distance': median_distances
        })
        clouds_distances_df.sort_values(by='mean_distance', ascending=True, inplace=True)

        same_pairs_mask = clouds_distances_df['display_name_1'] == clouds_distances_df['display_name_2']
        same_cloud_distances_df = clouds_distances_df.loc[same_pairs_mask]
        different_cloud_distances_df = clouds_distances_df.loc[~same_pairs_mask]

        file_name = f"{i_epoch:05d}.txt"

        # same cloud distances
        df_same_cloud_distances_str = df_to_str(same_cloud_distances_df,
                                          ['display_name_1', 'mean_distance', 'max_distance', 'min_distance', 'median_distance'])
        df_same_cloud_distances_str = f"""
            Note: only distinct samples pairs are taken (min distance should be > 0).\n
            Max is the max L2 distance between two samples, a.k.a the diameter of the cloud.\n
            Mean is the mean L2 distance between two samples in the cloud.\n
            {"=" * 100}\n
            \n{df_same_cloud_distances_str}
        """
        folder_path = os.path.join(self.directory_path, 'intra_cloud_distances')
        write_str_to_file(folder_path, file_name, df_same_cloud_distances_str)

        # distinct clouds distances
        df_different_cloud_distances_str = df_to_str(different_cloud_distances_df,
                                          ['display_name_1', 'display_name_2', 'mean_distance', 'max_distance', 'min_distance', 'median_distance'])
        df_different_cloud_distances_str = f"""
            Distances between different clouds. Sorted by mean_distance:\n
            {"=" * 100}\n
            \n{df_different_cloud_distances_str}
        """
        folder_path = os.path.join(self.directory_path, 'inter_clouds_distances')
        write_str_to_file(folder_path, file_name, df_different_cloud_distances_str)

        # top mean distances by tissue
        lines = []
        for tissue in cmap.tissues_unique:
            df = clouds_distances_df.loc[clouds_distances_df['tissue_1'] == tissue]
            lines.append(f'=' * 50)
            lines.append(f'For tissue {tissue.tissue_code}, closest pairs:')
            lines.append(f'=' * 50)
            pairs = df.head(12)
            for pair in pairs.iterrows():
                pair = pair[1]
                lines.append(f'{pair["display_name_1"]} <-> {pair["display_name_2"]} : {pair["mean_distance"]}')

        folder_path = os.path.join(self.directory_path, 'top_mean_distances_by_tissue')
        lines = '\n'.join(lines)
        str_to_write = f"""
            Top mean distances between two clouds, grouped by tissue, i.e. the top distance between tissue <t> to other
            cloud. We wish to see that the lowest distance is between a class to itself, and then to other perturbations
            of the same tissue, and then to biologically similar classes.\n
            {"=" * 100}\n\n{lines}
        """
        write_str_to_file(folder_path, file_name, str_to_write)
