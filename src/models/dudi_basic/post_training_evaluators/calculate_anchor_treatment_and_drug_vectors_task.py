from typing import List, Dict, Tuple

import torch
from prefect import Parameter

from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.models.multi_device_vector import MultiDeviceVector
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class CalculateAnchorTreatmentAndDrugVectorsTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            evaluator_params: TrainingEvaluatorParams,
            encoded_reference_points) -> Tuple[Dict[str, MultiDeviceVector], Dict[str, MultiDeviceVector]]:
        return calculate_anchor_treatment_and_drug_vectors(
            training_cmap=evaluator_params.updated_training_cmap_dataset,
            encoded_reference_points=encoded_reference_points,
            enable_triangle=evaluator_params.enable_triangle
        )
