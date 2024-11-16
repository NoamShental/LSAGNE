from typing import List, Union, Dict, Iterable

from src.cmap_cloud_ref import CmapCloudRef
from src.configuration import config
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset


class ParameterParser:
    def __init__(
            self,
            cmap_dataset: RawCmapDataset,
    ):
        self.cmap_dataset = cmap_dataset

    def parse_cloud_refs(
            self,
            param_name: str,
            clouds_strs: List[str]
    ) -> List[CmapCloudRef]:
        clouds_refs = set()
        if len(clouds_strs) % 2 != 0:
            raise AssertionError(f'{param_name} len is odd.')

        for i in range(0, len(clouds_strs), 2):
            tissue_code = clouds_strs[i]
            perturbation = clouds_strs[i + 1]
            clouds_refs.update(self.create_cloud_refs(tissue_code, perturbation))
        return list(clouds_refs)

    def create_cloud_refs(
            self,
            tissue_code: str,
            perturbation: str,
            all_literal: str = "all"
    ) -> List[CmapCloudRef]:
        if tissue_code == all_literal and perturbation == all_literal:
            cloud_refs = [cloud_ref for cloud_ref in self.cmap_dataset.unique_cloud_refs
                          if cloud_ref.tissue in config.tissues_whitelist
                          and cloud_ref.perturbation in config.perturbations_whitelist]
        elif tissue_code == all_literal:
            cloud_refs = [cloud_ref for cloud_ref in self.cmap_dataset.unique_cloud_refs
                          if cloud_ref.perturbation == perturbation and cloud_ref.tissue in config.tissues_whitelist]
        elif perturbation == all_literal:
            cloud_refs = [cloud_ref for cloud_ref in self.cmap_dataset.unique_cloud_refs
                          if cloud_ref.tissue_code == tissue_code
                          and cloud_ref.perturbation in config.perturbations_whitelist]
        else:
            return [CmapCloudRef(tissue_code, perturbation)]
        return [cloud_ref for cloud_ref in cloud_refs if cloud_ref.is_not_dmso_6h_or_24h]

    def parse_partial_clouds_requests(
            self,
            param_name: str,
            clouds_strs_with_size: List[Union[int, str]], # [[<tissue code>, <perturbation>, <size>]]
            clouds_to_exclude: Iterable[CmapCloudRef]
    ) -> Dict[CmapCloudRef, int]:
        cloud_ref_to_training_size: Dict[CmapCloudRef, int] = {}
        if len(clouds_strs_with_size) % 3 != 0:
            raise AssertionError(f'{param_name} size is not divided by 3.')
        for i in range(0, len(clouds_strs_with_size), 3):
            tissue_code = clouds_strs_with_size[i]
            perturbation = clouds_strs_with_size[i + 1]
            cloud_size = clouds_strs_with_size[i + 2]
            cloud_refs = self.create_cloud_refs(tissue_code, perturbation)
            for cloud_ref in cloud_refs:
                if cloud_ref in clouds_to_exclude:
                    continue
                if cloud_ref in cloud_ref_to_training_size:
                    existing_size = cloud_ref_to_training_size[cloud_ref]
                    if existing_size != cloud_size:
                        print(f'WARNING: {cloud_ref} size change {existing_size} -> {cloud_size}')
                cloud_ref_to_training_size[cloud_ref] = cloud_size

        for cloud_ref, training_size in cloud_ref_to_training_size.copy().items():
            cmap_cloud_size = len(self.cmap_dataset.cloud_ref_to_idx[cloud_ref])
            if cmap_cloud_size <= training_size:
                print(f"WARNING: {cloud_ref} desired partial training size is {training_size}, but cmap contains "
                      f"only {cmap_cloud_size} samples, this cloud will be trained as regular training.")
                cloud_ref_to_training_size.pop(cloud_ref)

        return cloud_ref_to_training_size
