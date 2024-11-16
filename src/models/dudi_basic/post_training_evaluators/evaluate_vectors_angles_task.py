import os
from typing import List, Dict, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from prefect import Parameter

from src.cmap_cloud_ref import CmapCloudRef
from src.dataframe_utils import df_to_str
from src.models.dudi_basic.post_training_evaluators.apply_svm_on_left_out_task import SvmPredictionOnLeftOut
from src.models.dudi_basic.post_training_evaluators.calculate_anchor_treatment_and_drug_vectors_task import \
    MultiDeviceVector
from src.models.dudi_basic.post_training_evaluators.calculate_predicted_clouds_task import PredictedCloud
from src.models.dudi_basic.post_training_evaluators.create_data_tensors_for_evaluation_task import \
    TrainingAndLeftOutTensors
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class EvaluateVectorsAnglesTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    def run(self,
            training_evaluator_params: TrainingEvaluatorParams,
            perturbation_to_anchor_treatment_vector: Dict[str, MultiDeviceVector],
            perturbation_to_anchor_drug_vector: Dict[str, MultiDeviceVector],
            training_and_left_out_tensors: TrainingAndLeftOutTensors,
            cloud_ref_to_predicted_cloud: Dict[CmapCloudRef, PredictedCloud],
            svm_prediction_on_left_out: SvmPredictionOnLeftOut,
            encoded_reference_points
            ):
        training_summary = training_evaluator_params.training_summary
        device = training_evaluator_params.device
        model = training_evaluator_params.model
        training_cmap = training_evaluator_params.updated_training_cmap_dataset
        left_out_cmap = training_evaluator_params.left_out_cmap_dataset
        model.eval()

        def _get_treatment_reference_vector_z(predicted_cloud: PredictedCloud):
            return predicted_cloud.true_reference_treatment_vector_z, \
                   predicted_cloud.treatment_vector_used_for_alpha_prediction

        def _get_drug_reference_vector_z(predicted_cloud: PredictedCloud):
            return predicted_cloud.true_reference_drug_vector_z, \
                   predicted_cloud.predicted_reference_drug_vector_z

        @torch.no_grad()
        def _calculate_vectors_angles(
                name: str,
                get_reference_vector_z: Callable[[PredictedCloud], Tuple[NDArray[float], NDArray[float]]],
                perturbation_to_anchor_vector: Dict[str, MultiDeviceVector]
        ) -> pd.DataFrame:
            angles_df_rows = []
            for perturbation in perturbation_to_anchor_vector.keys():
                # there cannot be untreated perturbation here
                anchor_vector = perturbation_to_anchor_vector[perturbation].vector_z
                for tissue in training_cmap.tissues_unique:
                    cloud_ref = CmapCloudRef(tissue, perturbation)
                    is_left_out = training_evaluator_params.left_out_cloud_ref == cloud_ref
                    if cloud_ref not in cloud_ref_to_predicted_cloud:
                        # There is not such class
                        continue
                    predicted_cloud = cloud_ref_to_predicted_cloud[cloud_ref]

                    true_reference_vector, predicted_reference_vector = get_reference_vector_z(predicted_cloud)
                    true_reference_vector_magnitude = np.linalg.norm(true_reference_vector)
                    predicted_reference_vector_magnitude = np.linalg.norm(predicted_reference_vector)
                    anchor_vector_magnitude = np.linalg.norm(anchor_vector)
                    angle_in_degrees_between_anchor_and_predicted_reference_vectors = \
                        np.degrees(np.arccos(np.dot(predicted_reference_vector, anchor_vector) /
                                  (predicted_reference_vector_magnitude * anchor_vector_magnitude)))
                    angles_df_rows.append({
                        'perturbation': perturbation,
                        'tissue': tissue,
                        'angle_in_degrees_between_anchor_and_predicted_reference_vectors': angle_in_degrees_between_anchor_and_predicted_reference_vectors,
                        'true_reference_vector_magnitude': true_reference_vector_magnitude,
                        'predicted_reference_vector_magnitude': predicted_reference_vector_magnitude,
                        'anchor_vector_magnitude': anchor_vector_magnitude,
                        'is_left_out': is_left_out,
                        'true_reference_vector': true_reference_vector,
                        'predicted_reference_vector': predicted_reference_vector,
                        'anchor_vector': anchor_vector
                    })

            df = pd.DataFrame(angles_df_rows)
            df.to_csv(os.path.join(self.working_directory, f'{name}.csv'))

            # in order to print the whole df
            df_str = df_to_str(df, ['perturbation', 'tissue', 'angle_in_degrees_between_anchor_and_predicted_reference_vectors',
                                    'true_reference_vector_magnitude', 'predicted_reference_vector_magnitude',
                                    'anchor_vector_magnitude', 'is_left_out'])
            self.logger.info(f'EvaluateVectorsAnglesTask {name}:\n{df_str}')

            return df

        res = {
            'treatment': _calculate_vectors_angles('treatment_vectors_angles',
                                                   _get_treatment_reference_vector_z,
                                                   perturbation_to_anchor_treatment_vector)
        }

        if training_evaluator_params.enable_triangle:
            res.update({
                'drug': _calculate_vectors_angles('drug_vectors_angles',
                                                  _get_drug_reference_vector_z,
                                                  perturbation_to_anchor_drug_vector)
            })

        return res
