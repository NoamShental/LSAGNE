from typing import List

import torch
from prefect import Parameter

from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class ReferencePointsEncoderTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            evaluator_params: TrainingEvaluatorParams):
        reference_points = evaluator_params.training_summary.anchor_points
        model = evaluator_params.model
        model.eval()
        device = evaluator_params.device

        return AnchorsAndVectorsEncoder(
            model=model,
            device=device
        ).encode_reference_points(reference_points)
