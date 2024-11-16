from typing import List, Dict, Tuple

import torch
from prefect import Parameter

from src.cmap_cloud_ref import CmapCloudRef
from src.models.dudi_basic.post_training_evaluators.calculate_anchor_treatment_and_drug_vectors_task import \
    MultiDeviceVector
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.models.predicted_clouds_calculator import predicted_clouds_calculator, PredictedCloud
from src.models.training_and_left_out_tensors import TrainingTensors
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class CalculatePredictedCloudsTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            training_evaluator_params: TrainingEvaluatorParams,
            perturbation_to_anchor_treatment_vector: Dict[str, MultiDeviceVector],
            perturbation_to_anchor_drug_vector: Dict[str, MultiDeviceVector],
            perturbation_and_tissue_to_reference_treatment_vector: Dict[Tuple[str, str], MultiDeviceVector],
            perturbation_and_tissue_to_reference_drug_vector: Dict[Tuple[str, str], MultiDeviceVector],
            training_tensors: TrainingTensors,
            encoded_reference_points
            ) -> Dict[CmapCloudRef, PredictedCloud]:
        return predicted_clouds_calculator(
            full_cmap_dataset=training_evaluator_params.updated_full_cmap_dataset,
            training_cmap_dataset=training_evaluator_params.updated_training_cmap_dataset,
            model=training_evaluator_params.model,
            perturbation_to_anchor_treatment_vector=perturbation_to_anchor_treatment_vector,
            perturbation_to_anchor_drug_vector=perturbation_to_anchor_drug_vector,
            cloud_ref_to_reference_treatment_vector=perturbation_and_tissue_to_reference_treatment_vector,
            cloud_ref_to_reference_drug_vector=perturbation_and_tissue_to_reference_drug_vector,
            training_z=training_tensors.training_z,
            encoded_reference_points=encoded_reference_points,
            enable_triangle=training_evaluator_params.enable_triangle,
            predicted_cloud_max_size=training_evaluator_params.predicted_cloud_max_size,
            include_non_trained=True,
            non_trained_cloud_refs=[training_evaluator_params.left_out_cloud_ref],
            cloud_ref_to_skip=training_evaluator_params.cross_validation_cloud_refs
        )

