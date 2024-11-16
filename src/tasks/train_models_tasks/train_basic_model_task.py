from typing import List

import torch
from prefect import Parameter

from src.configuration import config
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.model_learning_parameters import InputLearningParameters, ModelLearningParameters
from src.pipeline_utils import choose_device
from src.random_manager import RandomManager
from src.simple_model_trainer import SimpleModelTrainer
from src.tasks.abstract_tasks.experiment_task import ExperimentTask
from src.technical_learning_parameters import TechnicalLearningParameters
from src.training_summary import TrainingSummary


class TrainBasicModelTask(ExperimentTask):

    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=True, flow_parameters=flow_parameters, *args, **kwargs)

    def run(self,
            cmap: RawCmapDataset,
            params: InputLearningParameters) -> TrainingSummary:
        torch.autograd.set_detect_anomaly(True)
        model_training_parameters = TechnicalLearningParameters(
            device=choose_device(params.use_cuda, self.logger),
            random_manager=RandomManager(params.use_seed, params.random_seed, self.logger),
            working_directory=self.working_directory
        )
        model_training_parameters = ModelLearningParameters(
            model_training_parameters,
            params,
            cmap
        )
        torch.set_default_dtype(config.torch_numeric_precision_type)
        return SimpleModelTrainer(self.logger, model_training_parameters).train_model()
