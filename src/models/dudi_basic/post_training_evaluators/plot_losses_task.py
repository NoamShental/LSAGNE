import os
from typing import List

from prefect import Parameter

from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.os_utilities import create_dir_if_not_exists
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class PlotLossesTask(ExperimentTask):

    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    def run(self,
            training_evaluator_params: TrainingEvaluatorParams):
        training_summary = training_evaluator_params.training_summary
        losses_subfolder = 'losses_curves'
        create_dir_if_not_exists(os.path.join(self.working_directory, losses_subfolder))
        # todo add to training_summary
        n_epochs = training_summary.n_epochs

        training_evaluator_params.drawer.plot_curves(training_summary.epochs_losses_history, n_epochs, f'{losses_subfolder}/all.png')

        for loss_name, values in training_summary.epochs_losses_history.items():
            training_evaluator_params.drawer.plot_curves({loss_name: values}, n_epochs,
                                                         f'{losses_subfolder}/{loss_name}.png')
