from typing import List, Dict, Tuple

import torch
from prefect import Parameter

from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.models.multi_device_vector import MultiDeviceVector
from src.models.training_and_left_out_tensors import LeftOutTensors
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class CalculateReferenceTreatmentAndDrugVectorsTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            evaluator_params: TrainingEvaluatorParams,
            encoded_reference_points,
            left_out_tensors: LeftOutTensors
            ) -> Tuple[Dict[Tuple[str, str], MultiDeviceVector], Dict[Tuple[str, str], MultiDeviceVector]]:
        return calculate_reference_treatment_and_drug_vectors(
            training_cmap=evaluator_params.updated_training_cmap_dataset,
            full_cmap_dataset=evaluator_params.updated_full_cmap_dataset,
            encoded_reference_points=encoded_reference_points,
            enable_triangle=evaluator_params.enable_triangle,
            include_non_trained=True,
            non_trained_cloud_ref_to_true_encoded_z_t={
                evaluator_params.left_out_cloud_ref: left_out_tensors.true_left_out_z_t
            },
            cloud_ref_to_skip=evaluator_params.cross_validation_cloud_refs
        )
