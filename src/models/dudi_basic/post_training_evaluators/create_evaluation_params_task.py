from typing import List

from prefect import Parameter

from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_dataset_splitter import SplittedCmapDataset
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.drawer import Drawer
from src.models.dudi_basic.model_learning_parameters import InputLearningParameters
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.pipeline_utils import choose_device
from src.random_manager import RandomManager
from src.tasks.abstract_tasks.experiment_task import ExperimentTask
from src.training_summary import TrainingSummary


class CreateEvaluationParamsTask(ExperimentTask):

    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    def run(self,
            original_full_cmap_dataset: RawCmapDataset,
            params: InputLearningParameters,
            training_summary: TrainingSummary) -> TrainingEvaluatorParams:
        random_manager = RandomManager(params.use_seed, params.random_seed)
        device = choose_device(params.use_cuda, self.logger)

        drawer = Drawer(self.logger, self.working_directory)

        left_out_cloud_ref = CmapCloudRef(training_summary.tissue, training_summary.perturbation)
        splitted_cmap_dataset = SplittedCmapDataset.split_cmap_dataset(
            logger=self.logger,
            original_cmap_dataset=original_full_cmap_dataset,
            left_out_cloud_ref=left_out_cloud_ref,
            cross_validation_cloud_refs=params.cross_validation_clouds,
            cloud_ref_to_partial_training_size=params.partial_cloud_training
        )

        original_training_cmap_dataset = splitted_cmap_dataset.training_only
        left_out_cmap_dataset = splitted_cmap_dataset.left_out
        cross_validation_cloud_ref_to_cmap_dataset = splitted_cmap_dataset.cross_validation

        updated_full_cmap_dataset = RawCmapDataset.merge_datasets(training_summary.updated_train_cmap_dataset,
                                                                  *list(cross_validation_cloud_ref_to_cmap_dataset.values()),
                                                                  left_out_cmap_dataset)

        # Necessary when the task result is loaded from pickled file
        training_summary.model.load_state_dict(training_summary.model_state_dict)
        training_summary.model.to(device)

        evaluation_params = TrainingEvaluatorParams(
            working_directory=self.working_directory,
            left_out_cloud_ref=left_out_cloud_ref,
            cross_validation_cloud_refs=params.cross_validation_clouds,
            drawer=drawer,
            original_full_cmap_dataset=original_full_cmap_dataset,
            updated_full_cmap_dataset=updated_full_cmap_dataset,
            original_training_cmap_dataset=original_training_cmap_dataset,
            updated_training_cmap_dataset=training_summary.updated_train_cmap_dataset,
            left_out_cmap_dataset=left_out_cmap_dataset,
            cross_validation_cloud_ref_to_cmap_dataset=cross_validation_cloud_ref_to_cmap_dataset,
            training_summary=training_summary,
            model=training_summary.model,
            device=device,
            random_manager=random_manager,
            embedding_dim=params.embedding_dim,
            enable_triangle=params.enable_triangle,
            predicted_cloud_max_size=params.predicted_cloud_max_size
        )

        return evaluation_params
