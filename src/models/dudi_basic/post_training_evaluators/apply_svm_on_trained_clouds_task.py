import os
from typing import List, Dict

import torch
from prefect import Parameter

from src.cmap_cloud_ref import CmapCloudRef
from src.models.dudi_basic.post_training_evaluators.apply_svm_on_left_out_task import SvmPredictionOnLeftOut
from src.models.dudi_basic.post_training_evaluators.calculate_predicted_clouds_task import PredictedCloud
from src.models.dudi_basic.post_training_evaluators.create_data_tensors_for_evaluation_task import \
    TrainingAndLeftOutTensors
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.models.svm_utils import predict_cmap_clouds_using_svm
from src.cmap_cloud import CmapCloud
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class ApplySvmOnTrainedCloudsTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            training_evaluator_params: TrainingEvaluatorParams,
            training_and_left_out_tensors: TrainingAndLeftOutTensors,
            cloud_ref_to_predicted_cloud: Dict[CmapCloudRef, PredictedCloud],
            svm_prediction_on_left_out: SvmPredictionOnLeftOut
            ):
        self.logger.info('SVM accuracy on predicted clouds:')

        svm = svm_prediction_on_left_out.svm

        cmap_clouds: List[CmapCloud] = []
        for cloud_ref, predicted_cloud in cloud_ref_to_predicted_cloud.items():
            cmap_clouds.append(CmapCloud(
                tissue=cloud_ref.tissue,
                perturbation=cloud_ref.perturbation,
                label=predicted_cloud.true_labels[0] if len(predicted_cloud.true_labels) > 0 else [],
                samples=predicted_cloud.predicted_z
            ))

        df = predict_cmap_clouds_using_svm(svm, cmap_clouds, self.logger)
        df.to_csv(os.path.join(self.working_directory, 'svm_score_on_predicted.csv'))
