from typing import List, Tuple
from prefect import Parameter
from src.data_reduction_utils import UmapReductionTool, TsneReductionTool, PCAReductionTool, DataReductionTool, LDAReductionTool
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.tasks.abstract_tasks.experiment_task import ExperimentTask
from src.training_summary import TrainingSummary


class CreateDataReductionMethodsTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    def run(self,
            training_summary: TrainingSummary
            ) -> Tuple[DataReductionTool, DataReductionTool, DataReductionTool, DataReductionTool]:
        return UmapReductionTool(), \
               TsneReductionTool(seed=training_summary.params.random_manager.random_seed), \
               PCAReductionTool(), LDAReductionTool()
