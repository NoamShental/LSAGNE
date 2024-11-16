from typing import List

import torch
from prefect import Parameter

from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.models.training_and_left_out_tensors import TrainingAndLeftOutTensors
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class CreateDataTensorsForEvaluationTask(ExperimentTask):

    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self, evaluator_params: TrainingEvaluatorParams) -> TrainingAndLeftOutTensors:
        training_original_space_t = torch.tensor(
            evaluator_params.updated_training_cmap_dataset.data,
            device=evaluator_params.device
        )
        training_samples_embedding = evaluator_params.model.get_embedding(training_original_space_t)

        true_left_out_original_space_t = torch.tensor(
            evaluator_params.left_out_cmap_dataset.data,
            device=evaluator_params.device
        )
        true_left_out_embedding = evaluator_params.model.get_embedding(
            true_left_out_original_space_t)

        # FIXME use the new class instead
        return TrainingAndLeftOutTensors(
            training_original_space_t=training_original_space_t,
            training_z_t=training_samples_embedding.z_t,
            training_mu_t=training_samples_embedding.mu_t,
            training_log_var_t=training_samples_embedding.log_var_t,
            true_left_out_original_space_t=true_left_out_original_space_t,
            true_left_out_z_t=true_left_out_embedding.z_t,
            true_left_out_mu_t=true_left_out_embedding.mu_t,
            true_left_out_log_var_t=true_left_out_embedding.log_var_t,

            training_original_space=training_original_space_t.cpu().numpy(),
            training_z=training_samples_embedding.z_t.cpu().numpy(),
            training_mu=training_samples_embedding.mu_t.cpu().numpy(),
            training_log_var=training_samples_embedding.log_var_t.cpu().numpy(),
            true_left_out_original_space=true_left_out_original_space_t.cpu().numpy(),
            true_left_out_z=true_left_out_z_t.cpu().numpy(),
            true_left_out_mu=true_left_out_mu_t.cpu().numpy(),
            true_left_out_log_var=true_left_out_log_var_t.cpu().numpy(),
        )
