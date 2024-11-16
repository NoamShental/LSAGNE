from src.configuration import config
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.model_learning_parameters import InputLearningParameters
from src.tasks.abstract_tasks.experiment_task import ExperimentTask
import numpy as np


class LoadRawCmapTask(ExperimentTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, params: InputLearningParameters) -> RawCmapDataset:
        cmap = RawCmapDataset.load_dataset_from_disk(
            tissues_whitelist=params.tissues_whitelist,
            perturbations_whitelist=params.perturbations_whitelist
        )

        if params.initial_max_dmso_cloud_size is None:
            return cmap

        mask = np.zeros(len(cmap), bool)

        for cloud_encoded_label, cloud_idx in cmap.encoded_label_to_idx.items():
            perturbation, _ = cmap.encoded_label_to_perturbation_and_tissue[cloud_encoded_label]
            if perturbation in [config.time_24h_perturbation, config.dmso_6h_perturbation] and len(cloud_idx) > params.initial_max_dmso_cloud_size:
                cloud_idx = np.random.choice(cloud_idx, params.initial_max_dmso_cloud_size, replace=False)
            mask[cloud_idx] = True

        return RawCmapDataset(data_df=cmap.data_df[mask], info_df=cmap.info_df[mask], scaler=cmap.scaler)
