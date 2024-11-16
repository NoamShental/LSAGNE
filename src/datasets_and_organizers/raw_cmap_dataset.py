# import future due to
# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import os
from collections import defaultdict
from functools import cached_property
from os import PathLike
from typing import Tuple, Dict, List, Optional, Set, Callable, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset

from src.cmap_cloud_ref import CmapCloudRef
from src.configuration import config
from src.perturbation import Perturbation
from src.tissue import Tissue


# TODO create CMAP Metadata for functions that need cloud refs data
class RawCmapDataset(Dataset[Tuple[NDArray, int]]):
    @classmethod
    def merge_datasets(cls, *datasets: RawCmapDataset) -> RawCmapDataset:
        if len(datasets) == 0:
            raise Exception('At least one dataset is needed to merge.')
        data_dfs = [dataset.data_df for dataset in datasets]
        info_dfs = [dataset.info_df for dataset in datasets]
        return RawCmapDataset(
            data_df=pd.concat(data_dfs),
            info_df=pd.concat(info_dfs),
            scaler=None
        )

    @classmethod
    def load_dataset_from_disk(
            cls,
            perturbations_whitelist: Optional[List[str]] = None,
            tissues_whitelist: Optional[List[str]] = None,
            cmap_folder: PathLike | None = None
    ):
        if not cmap_folder:
            cmap_folder = config.organized_cmap_folder
        data_path = os.path.join(cmap_folder, config.data_file_name)
        info_path = os.path.join(cmap_folder, config.information_file_name)
        scaler_path = os.path.join(cmap_folder, 'cmap_scaler')
        data_df: pd.DataFrame = pd.read_hdf(data_path, 'df')
        # check that all the loaded df is of the desired type, if not, convert it to the desired type
        df_dtypes = np.unique(data_df.dtypes.to_numpy())
        assert len(df_dtypes) == 1 and df_dtypes[0] == np.float64, 'All the CMAP df should be float64 type!'
        data_df = data_df.astype(config.np_numeric_precision_type)
        df_dtypes = np.unique(data_df.dtypes.to_numpy())
        assert len(df_dtypes) == 1 and df_dtypes[0] == config.np_numeric_precision_type, 'Numeric type should change.'
        info_types = {'inst_id': str,
                      'perturbation': str,
                      'tumor': str,
                      'classifier_labels': str,
                      'pert_time_label': str,
                      'numeric_labels': np.int64}
        info_df: pd.DataFrame = pd.read_csv(info_path, dtype=info_types)
        if perturbations_whitelist and tissues_whitelist:
            whitelist_tissues_idx = info_df['tumor'].isin(tissues_whitelist)
            whitelist_perturbations_idx = info_df['perturbation'].isin(perturbations_whitelist + config.untreated_labels)
            whitelist_idx = whitelist_tissues_idx & whitelist_perturbations_idx
            info_df = info_df.loc[whitelist_idx]
            data_df = data_df.loc[info_df['inst_id']]
        info_df.set_index('inst_id', inplace=True, drop=True)
        # scaler = joblib.load(scaler_path)
        scaler = None
        return cls(data_df, info_df, scaler)

    @staticmethod
    def apply_func_on_numpy(func: Callable[[Any], Any], use_as_ctor: bool = True) -> Callable[[NDArray], NDArray]:
        otype = [func] if use_as_ctor else None
        return np.vectorize(func, otype)

    def identify_sample_idx_and_id(self, sample: NDArray[float]) -> Tuple[int, str]:
        idxs = np.where((self.data == sample).all(axis=1))[0]
        if len(idxs) > 1:
            raise AssertionError("Found multiple idx for same sample")
        sample_idx = idxs[0]
        return sample_idx, self.samples_cmap_id[sample_idx]

    @cached_property
    def encoded_labels_to_balanced_class_weights(self) -> Dict[int, float]:
        # return np.ones((len(self.encoded_labels_unique)), dtype=config.np_numeric_precision_type)
        return compute_class_weight('balanced',
                                    classes=self.encoded_labels_unique,
                                    y=self.encoded_labels).astype(config.np_numeric_precision_type)

    @cached_property
    def cloud_ref_to_balanced_class_weights(self) -> Dict[CmapCloudRef, float]:
        return {
            self.encoded_label_to_cloud_ref[encoded_label]: class_weight
            for encoded_label, class_weight in self.encoded_labels_to_balanced_class_weights.items()
        }

    @cached_property
    def encoded_tissues_to_balanced_class_weights(self) -> Dict[int, float]:
        # return np.ones((len(self.encoded_labels_unique)), dtype=config.np_numeric_precision_type)
        return compute_class_weight('balanced',
                                    classes=self.encoded_tissues_unique,
                                    y=self.encoded_tissues).astype(config.np_numeric_precision_type)

    @cached_property
    def tissue_to_balanced_class_weights(self) -> Dict[Tissue, float]:
        return {
            self.encoded_tissue_to_tissue[encoded_tissue]: class_weight
            for encoded_tissue, class_weight in self.encoded_tissues_to_balanced_class_weights.items()
        }

    @cached_property
    def encoded_label_to_perturbation_and_tissue(self) -> Dict[int, Tuple[str, str]]:
        encoded_label_to_perturbation_and_tissue = {}
        for encoded_label in self.encoded_labels_unique:
            idx = self.encoded_label_to_idx_mask[encoded_label]
            perturbation = self.perturbations[idx][0]
            tissue = self.tissues[idx][0]
            encoded_label_to_perturbation_and_tissue[encoded_label] = (perturbation, tissue)
        return encoded_label_to_perturbation_and_tissue

    @cached_property
    def encoded_label_to_cloud_ref(self) -> Dict[int, CmapCloudRef]:
        return {
            encoded_label: CmapCloudRef(tissue, perturbation)
            for encoded_label, (perturbation, tissue) in self.encoded_label_to_perturbation_and_tissue.items()
        }

    @cached_property
    def encoded_label_to_encoded_perturbation_and_tissue(self):
        encoded_label_to_encoded_perturbation_and_tissue = {}
        for encoded_label, (perturbation, tissue) in self.encoded_label_to_perturbation_and_tissue.items():
            encoded_perturbation = self.perturbation_encoder.transform([perturbation])[0]
            encoded_tissue = self.tissue_encoder.transform([tissue])[0]
            encoded_label_to_encoded_perturbation_and_tissue[encoded_label] = (encoded_perturbation, encoded_tissue)
        return encoded_label_to_encoded_perturbation_and_tissue

    @cached_property
    def encoded_perturbation_to_encoded_tissues(self):
        encoded_perturbation_to_encoded_tissues = defaultdict(list)
        for encoded_label, (encoded_perturbation, encoded_tissue) in self.encoded_label_to_encoded_perturbation_and_tissue.items():
            encoded_perturbation_to_encoded_tissues[encoded_perturbation].append(encoded_tissue)
        return encoded_perturbation_to_encoded_tissues

    @cached_property
    def encoded_tissue_to_encoded_perturbations(self):
        encoded_tissues_to_encoded_perturbation = defaultdict(list)
        for _, (encoded_perturbation, encoded_tissue) in self.encoded_label_to_encoded_perturbation_and_tissue.items():
            encoded_tissues_to_encoded_perturbation[encoded_tissue].append(encoded_perturbation)
        return encoded_tissues_to_encoded_perturbation

    @cached_property
    def data_dim(self):
        return self.data_df.shape[1]

    @cached_property
    def gene_names(self) -> NDArray[str]:
        return np.array(self.data_df.columns)

    @cached_property
    def display_names(self):
        return self.info_df['classifier_labels'].to_numpy()

    @cached_property
    def data(self) -> NDArray[float]:
        return self.data_df.to_numpy()

    @cached_property
    def is_dmso_6h_mask(self) -> NDArray[bool]:
        return np.isin(self.perturbations, config.untreated_labels)

    @cached_property
    def is_dmso_24h_mask(self) -> NDArray[bool]:
        return self.perturbations == config.time_24h_perturbation

    @cached_property
    def not_dmso_6h_or_24h_mask(self) -> NDArray[bool]:
        return ~self.is_dmso_6h_mask & ~self.is_dmso_24h_mask

    @cached_property
    def is_treated_without_time(self) -> NDArray[bool]:
        treated_without_time_labels = config.perturbations_whitelist.copy()
        treated_without_time_labels.remove(config.time_24h_perturbation)
        return np.isin(self.perturbations, config.treated_without_time_labels)

    @cached_property
    def info(self) -> NDArray[Dict]:
        return np.array(list(self.info_df.to_dict('index').values()))

    @cached_property
    def original_numeric_labels(self) -> NDArray[int]:
        return self.info_df['numeric_labels'].values

    @cached_property
    def original_numeric_labels_unique(self) -> NDArray[int]:
        return np.unique(self.original_numeric_labels)

    @cached_property
    def label_encoder(self):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.original_numeric_labels)
        return label_encoder

    def encode_original_labels(self, original_numeric_labels: NDArray[int]) -> NDArray[int]:
        return self.label_encoder.transform(original_numeric_labels)

    @cached_property
    def encoded_labels(self) -> NDArray[int]:
        return self.encode_original_labels(self.original_numeric_labels)

    @cached_property
    def unique_cloud_refs(self) -> List[CmapCloudRef]:
        cloud_refs = []
        for perturbation, tissue in self.perturbation_and_tissue_to_encoded_label.keys():
            cloud_refs.append(CmapCloudRef(tissue, perturbation))
        return cloud_refs

    @cached_property
    def cloud_refs(self) -> NDArray[CmapCloudRef]:
        return self.apply_func_on_numpy(lambda encoded_label: self.encoded_label_to_cloud_ref[encoded_label], False)(self.encoded_labels)

    @cached_property
    def perturbation_to_cloud_refs(self) -> dict[Perturbation, list[CmapCloudRef]]:
        res = defaultdict(list)
        for cloud_ref in self.unique_cloud_refs:
            res[cloud_ref.perturbation].append(cloud_ref)
        return res

    @cached_property
    def tissue_to_cloud_refs(self) -> dict[Tissue, list[CmapCloudRef]]:
        res = defaultdict(list)
        for cloud_ref in self.unique_cloud_refs:
            res[cloud_ref.tissue].append(cloud_ref)
        return res

    @cached_property
    def encoded_label_to_cloud_size(self) -> Dict[int, int]:
        return {
            encoded_label: self.encoded_label_to_idx_mask[encoded_label].sum()
            for encoded_label in self.encoded_labels_unique
        }

    @cached_property
    def encoded_dmso_6h_clouds_labels(self) -> NDArray[int]:
        encoded_labels = []
        for encoded_label, (perturbation, tissue) in self.encoded_label_to_perturbation_and_tissue.items():
            if perturbation in config.untreated_labels:
                encoded_labels.append(encoded_label)
        return np.array(encoded_labels)

    @cached_property
    def encoded_dmso_24h_clouds_labels(self) -> NDArray[int]:
        encoded_labels = []
        for encoded_label, (encoded_perturbation, encoded_tissue) in self.encoded_label_to_encoded_perturbation_and_tissue.items():
            if encoded_perturbation == self.encoded_dmso_24h_perturbation:
                encoded_labels.append(encoded_label)
        return np.array(encoded_labels)

    @cached_property
    def encoded_dmso_24h_perturbation(self) -> int:
        return self.perturbation_encoder.transform([config.time_24h_perturbation])[0]

    @cached_property
    def dmso_24h_perturbation(self) -> Perturbation:
        return Perturbation(config.time_24h_perturbation)

    @cached_property
    def encoded_labels_unique(self):
        return np.unique(self.encoded_labels)

    @cached_property
    def original_labels_unique(self):
        return np.unique(self.original_numeric_labels)

    @cached_property
    def clouds_count(self):
        return len(self.original_labels_unique)

    @cached_property
    def tissues_count(self):
        return len(self.tissues_unique)

    @cached_property
    def perturbations(self) -> NDArray[Perturbation]:
        return self.apply_func_on_numpy(Perturbation)(self.info_df['perturbation'].values)

    @cached_property
    def perturbations_unique(self) -> NDArray[Perturbation]:
        return self.apply_func_on_numpy(Perturbation)(np.unique(self.perturbations))

    @cached_property
    def control_encoded_perturbations(self):
        return self.perturbation_encoder.transform(config.untreated_labels)

    @cached_property
    def tissues(self) -> NDArray[Tissue]:
        return self.apply_func_on_numpy(Tissue)(self.info_df['tumor'].values)

    @cached_property
    def tissues_unique(self) -> NDArray[Tissue]:
        return self.apply_func_on_numpy(Tissue)(np.unique(self.tissues))

    @cached_property
    def perturbation_encoder(self):
        perturbation_encoder = preprocessing.LabelEncoder()
        perturbation_encoder.fit(self.perturbations)
        return perturbation_encoder

    @cached_property
    def encoded_perturbations(self):
        return self.perturbation_encoder.transform(self.perturbations)

    @cached_property
    def encoded_perturbations_unique(self):
        return np.unique(self.encoded_perturbations)

    @cached_property
    def perturbation_to_encoded_perturbation(self) -> Dict[Perturbation, int]:
        encoded_perturbations = self.perturbation_encoder.transform(self.perturbations_unique)
        return {
            perturbation: encoded_perturbation for perturbation, encoded_perturbation in
            zip(self.perturbations_unique, encoded_perturbations)
        }

    @cached_property
    def tissue_encoder(self):
        tissue_encoder = preprocessing.LabelEncoder()
        tissue_encoder.fit(self.tissues)
        return tissue_encoder

    @cached_property
    def encoded_tissues(self) -> NDArray[int]:
        return self.tissue_encoder.transform(self.tissues)

    @cached_property
    def encoded_tissues_unique(self) -> NDArray[int]:
        return np.unique(self.encoded_tissues)

    @cached_property
    def tissue_to_encoded_tissues(self) -> Dict[str, int]:
        encoded_tissues = self.tissue_encoder.transform(self.tissues_unique)
        return {
            tissue: encoded_tissue for tissue, encoded_tissue in zip(self.tissues_unique, encoded_tissues)
        }

    @cached_property
    def encoded_label_to_idx_mask(self) -> Dict[int, NDArray[bool]]:
        result = {}
        for label in self.encoded_labels_unique:
            result[label] = self.encoded_labels == label
        return result

    @cached_property
    def cloud_ref_to_idx_mask(self) -> Dict[CmapCloudRef, NDArray[bool]]:
        return {
            self.encoded_label_to_cloud_ref[encoded_label]: idx_mask
            for encoded_label, idx_mask in self.encoded_label_to_idx_mask.items()
        }

    @cached_property
    def cloud_ref_to_samples(self) -> Dict[CmapCloudRef, NDArray[float]]:
        return {
            cloud_ref: self.data[idx_mask]
            for cloud_ref, idx_mask in self.cloud_ref_to_idx_mask.items()
        }

    @cached_property
    def encoded_label_to_idx(self):
        return {encoded_label: idx_mask.nonzero()[0] for encoded_label, idx_mask in self.encoded_label_to_idx_mask.items()}

    @cached_property
    def cloud_ref_to_idx(self):
        return {
            self.encoded_label_to_cloud_ref[encoded_label]: idx
            for encoded_label, idx in self.encoded_label_to_idx.items()
        }

    @cached_property
    def original_label_to_idx(self):
        result = {}
        for label in self.original_labels_unique:
            result[label] = (self.original_numeric_labels == label).nonzero()[0]
        return result

    @cached_property
    def perturbation_to_idx_mask(self):
        result = {}
        for perturbation in np.unique(self.perturbations):
            result[perturbation] = self.perturbations == perturbation
        return result

    @cached_property
    def tissue_to_idx_mask(self):
        result = {}
        for tissue in np.unique(self.tissues):
            result[tissue] = self.tissues == tissue
        return result

    @cached_property
    def encoded_perturbation_to_idx(self):
        return {self.perturbation_encoder.transform(perturbation)[0]: idx for perturbation, idx in
                self.perturbation_to_idx_mask.items()}

    @cached_property
    def untreated_to_idx(self):
        result = {}
        for tissue in np.unique(self.tissues):
            result[tissue] = ((self.tissues == tissue) & self.is_dmso_6h_mask)
        return result

    @cached_property
    def encoded_tissues_to_idx_mask(self):
        return {self.tissue_to_encoded_tissues[tissue]: idx_mask for tissue, idx_mask in self.tissue_to_idx_mask.items()}

    @cached_property
    def encoded_tissues_to_idx(self):
        return {encoded_tissue: idx_mask.nonzero()[0] for encoded_tissue, idx_mask in self.encoded_tissues_to_idx_mask.items()}

    @cached_property
    def encoded_label_to_display_name(self) -> Dict[int, str]:
        numeric_label_to_str = {}
        for encoded_label in self.encoded_labels_unique:
            idx = self.encoded_label_to_idx_mask[encoded_label]
            string_label = self.display_names[idx][0]
            numeric_label_to_str[encoded_label] = string_label
        return numeric_label_to_str

    @cached_property
    def cloud_ref_to_display_name(self) -> Dict[CmapCloudRef, str]:
        return {cloud_ref: self.encoded_label_to_display_name[encoded_label]
                for cloud_ref, encoded_label in self.cloud_ref_to_encoded_label.items()}

    @cached_property
    def original_label_to_display_name(self) -> Dict[int, str]:
        numeric_label_to_str = {}
        for original_label in self.original_labels_unique:
            idx = self.original_label_to_idx[original_label]
            string_label = self.display_names[idx][0]
            numeric_label_to_str[original_label] = string_label
        return numeric_label_to_str

    @cached_property
    def cloud_ref_to_display_name(self) -> Dict[CmapCloudRef, str]:
        return {cloud_ref: self.encoded_label_to_display_name[encoded_label]
                for cloud_ref, encoded_label in self.cloud_ref_to_encoded_label.items()}

    @cached_property
    def samples_cmap_id(self) -> NDArray[str]:
        return self.info_df.index.to_numpy()

    @cached_property
    def encoded_perturbation_to_perturbation_str(self) -> Dict[int, str]:
        perturbations_str = self.perturbation_encoder.inverse_transform(self.encoded_perturbations_unique)
        return {encoded_perturbation: perturbation for encoded_perturbation, perturbation in zip(self.encoded_perturbations_unique, perturbations_str)}

    @cached_property
    def encoded_tissue_to_tissue(self) -> Dict[int, Tissue]:
        tissue_str = self.tissue_encoder.inverse_transform(self.encoded_tissues_unique)
        return {encoded_tissue: Tissue(tissue) for encoded_tissue, tissue in
                zip(self.encoded_tissues_unique, tissue_str)}

    @cached_property
    def encoded_perturbation_and_tissue_to_encoded_label(self) -> Dict[Tuple[int, int], int]:
        return {(encoded_perturbation, encoded_tissue): encoded_label
                for encoded_label, (encoded_perturbation, encoded_tissue) in
                self.encoded_label_to_encoded_perturbation_and_tissue.items()}

    @cached_property
    def perturbation_and_tissue_to_encoded_label(self) -> Dict[Tuple[str, str], int]:
        return {(self.encoded_perturbation_to_perturbation_str[encoded_perturbation], self.encoded_tissue_to_tissue[encoded_tissue]): encoded_label
                for encoded_label, (encoded_perturbation, encoded_tissue) in
                self.encoded_label_to_encoded_perturbation_and_tissue.items()}

    @cached_property
    def time_24h_cloud_refs(self) -> Set[CmapCloudRef]:
        return {cloud_ref for cloud_ref in self.unique_cloud_refs
                if cloud_ref.perturbation not in [config.dmso_6h_perturbation]}

    @cached_property
    def non_dmso_cloud_refs(self) -> Set[CmapCloudRef]:
        return {cloud_ref for cloud_ref in self.unique_cloud_refs
                if cloud_ref.perturbation not in [config.dmso_6h_perturbation, config.time_24h_perturbation]}

    @cached_property
    def cloud_ref_to_dmso_6h(self) -> Dict[CmapCloudRef, CmapCloudRef]:
        return {cloud_ref: CmapCloudRef(cloud_ref.tissue, config.dmso_6h_perturbation)
                for cloud_ref in self.unique_cloud_refs}

    @cached_property
    def cloud_ref_to_dmso_24h(self) -> Dict[CmapCloudRef, CmapCloudRef]:
        return {cloud_ref: CmapCloudRef(cloud_ref.tissue, config.time_24h_perturbation)
                for cloud_ref in self.unique_cloud_refs}

    @cached_property
    def cloud_ref_to_encoded_label(self) -> Dict[CmapCloudRef, int]:
        return {
            CmapCloudRef(tissue, perturbation): encoded_label
            for (perturbation, tissue), encoded_label in self.perturbation_and_tissue_to_encoded_label.items()
        }

    @cached_property
    def encoded_label_to_original_label(self) -> Dict[int, int]:
        return {encoded_label: self.label_encoder.inverse_transform([encoded_label])[0]
                for encoded_label in self.encoded_labels_unique}

    @cached_property
    def cloud_ref_to_original_label(self) -> Dict[CmapCloudRef, int]:
        return {cloud_ref: self.label_encoder.inverse_transform([encoded_label])[0]
                for cloud_ref, encoded_label in self.cloud_ref_to_encoded_label.items()}

    @cached_property
    def encoded_label_to_cloud_size(self) -> Dict[int, int]:
        return {encoded_label: mask.sum() for encoded_label, mask in self.encoded_label_to_idx_mask.items()}

    @cached_property
    def empty(self) -> RawCmapDataset:
        return RawCmapDataset(
            data_df=self.data_df.loc[[]],
            info_df=self.info_df.loc[[]],
            scaler=self.scaler
        )

    @cached_property
    def summary(self) -> str:
        s = 'CMAP summary:\n'
        s += '*' * 50 + '\n'
        s += 'Display name to cloud size:\n'
        s += '=' * 50 + '\n'
        for encoded_label, cloud_size in self.encoded_label_to_cloud_size.items():
            s += f'{self.encoded_label_to_display_name[encoded_label]} -> {cloud_size}\n'
        s += '=' * 50 + '\n'
        s += f'Clouds count -> {len(self.encoded_labels_unique):,}\n'
        s += f'Total size -> {len(self.data):,}\n'
        s += f'Total untreated samples -> {self.is_dmso_6h_mask.sum():,}\n'
        s += f'Total treated samples -> {(~self.is_dmso_6h_mask).sum():,}\n'
        s += f'Total time 24h samples -> {(self.perturbations == config.time_24h_perturbation).sum():,}\n'
        s += '*' * 50
        return s

    def __init__(self, data_df, info_df, scaler):
        self.data_df: pd.DataFrame = data_df
        self.info_df: pd.DataFrame = info_df
        self.scaler: TransformerMixin = scaler

    def __getitem__(self, index) -> Tuple[NDArray, int]:
        raw_sample = self.data[index]
        info = self.info[index]
        return raw_sample, info['numeric_labels']

    def __len__(self):
        return self.data_df.shape[0]

    def __str__(self):
        return f'CMap dataset, len={len(self)}, number of clouds={self.clouds_count}.'

    def filter_out_inplace(self, indexes):
        self.data_df = self.data_df.loc[~indexes.values]
        self.info_df = self.info_df.loc[~indexes.values]
        self._invalidate_cached_properties()

    def split_by_mask(self, idx_mask: NDArray[bool]) -> Tuple[RawCmapDataset, RawCmapDataset]:
        masked_data = self.data_df.loc[idx_mask]
        masked_info = self.info_df.loc[idx_mask]
        remaining_data = self.data_df.loc[~idx_mask]
        remaining_info = self.info_df.loc[~idx_mask]
        # TODO what to do with the scaler?!
        return RawCmapDataset(masked_data, masked_info, self.scaler), \
               RawCmapDataset(remaining_data, remaining_info, self.scaler)

    def leave_out(self, *cloud_refs: CmapCloudRef) -> Tuple[RawCmapDataset, RawCmapDataset]:
        if len(cloud_refs) == 0:
            return self, self.empty
        leave_out_indexes_list = [self.cloud_ref_to_idx_mask[cloud_ref] for cloud_ref in cloud_refs]
        leave_out_indexes = np.logical_or.reduce(leave_out_indexes_list)
        return self.split_by_mask(~leave_out_indexes)

    def _encoded_label_for(self, p, t):
        perturbation_info_df = self.info_df[self.info_df.perturbation == p]
        perturbation_and_tissue_info_df = perturbation_info_df[perturbation_info_df.tumor == t]
        if len(perturbation_and_tissue_info_df) == 0:
            return None
        return self.label_encoder.transform([perturbation_and_tissue_info_df.numeric_labels[0]])[0]

    def _invalidate_cached_properties(self):
        raise Exception('Fix')
        del self.__dict__['data']
        del self.__dict__['info']
        del self.__dict__['numeric_labels']
        del self.__dict__['clouds_count']
        del self.__dict__['perturbations']
        del self.__dict__['class_labels_to_idx']
        del self.__dict__['perturbation_to_idx_mask']
        del self.__dict__['numeric_label_to_display_name_label']
