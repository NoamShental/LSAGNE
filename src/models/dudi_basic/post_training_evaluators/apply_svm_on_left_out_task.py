import os
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from prefect import Parameter
from sklearn import svm
from sklearn.metrics import accuracy_score

from src.cmap_cloud_ref import CmapCloudRef
from src.dataframe_utils import df_to_str
from src.models.dudi_basic.post_training_evaluators.calculate_predicted_clouds_task import PredictedCloud
from src.models.dudi_basic.post_training_evaluators.create_data_tensors_for_evaluation_task import \
    TrainingAndLeftOutTensors
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.tasks.abstract_tasks.experiment_task import ExperimentTask
from src.write_to_file_utils import write_str_to_file


@dataclass
class SvmPredictionOnLeftOut:
    accuracy_score: float
    true_left_out_labels: ArrayLike
    predicted_left_out_labels: ArrayLike
    correctly_classified_left_out_z: ArrayLike
    incorrectly_classified_left_out_z: ArrayLike
    svm: svm.LinearSVC


class ApplySvmOnLeftOutTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            training_evaluator_params: TrainingEvaluatorParams,
            training_and_left_out_tensors: TrainingAndLeftOutTensors,
            cloud_ref_to_predicted_cloud: Dict[CmapCloudRef, PredictedCloud]
            ) -> SvmPredictionOnLeftOut:

        model = training_evaluator_params.model
        model.eval()

        left_out_predicted_cloud = cloud_ref_to_predicted_cloud[training_evaluator_params.left_out_cloud_ref]
        linear_svm_classifier, training_with_predicted_z, training_with_predicted_true_labels = train_linear_svm_using_training_and_predicted_non_trained(
            non_trained_predicted_clouds=[left_out_predicted_cloud],
            training_z=training_and_left_out_tensors.training_z,
            training_z_original_numeric_labels=training_evaluator_params.updated_training_cmap_dataset.original_numeric_labels,
            random_seed=training_evaluator_params.random_manager.random_seed
        )

        trained_on_prediction = linear_svm_classifier.predict(training_with_predicted_z)
        self.logger.info(f'SANITY CHECK!!! Checking SVM on trained samples = {accuracy_score(training_with_predicted_true_labels, trained_on_prediction)}')

        sanity_check_cloud_name_accuracy_rows = []
        missed_samples_ids_rows = []
        for cloud_label in np.unique(training_with_predicted_true_labels):
            is_left_out = cloud_label == left_out_predicted_cloud.true_labels[0]
            cloud_mask = training_with_predicted_true_labels == cloud_label
            true_labels = training_with_predicted_true_labels[cloud_mask]
            svm_prediction_labels = trained_on_prediction[cloud_mask]
            display_name = training_evaluator_params.original_full_cmap_dataset.original_label_to_display_name[cloud_label]
            miss_mask = svm_prediction_labels != true_labels
            sanity_check_cloud_name_accuracy_rows.append({
                'display_name': display_name,
                'original_label': cloud_label,
                'is_left_out': is_left_out,
                'cloud_size': len(true_labels),
                'accuracy': accuracy_score(true_labels, svm_prediction_labels),
                'misclassified_size': miss_mask.sum()
            })
            if not is_left_out:
                # here left out is the predicted left out, not the true one
                missed_labels = svm_prediction_labels[miss_mask]
                only_cloud_cmap_idx = training_evaluator_params.updated_training_cmap_dataset.original_label_to_idx[
                    cloud_label]
                missed_sample_ids = \
                training_evaluator_params.updated_training_cmap_dataset.samples_cmap_id[only_cloud_cmap_idx][miss_mask]
                for id, predicted_svm_label in zip(missed_sample_ids, missed_labels):
                    missed_samples_ids_rows.append({
                        'display_name': display_name,
                        'is_left_out': is_left_out,
                        'sample_id': id,
                        'desired_label': cloud_label,
                        'svm_predicted_label': predicted_svm_label
                    })

        sanity_check_cloud_name_to_accuracy_df = pd.DataFrame(sanity_check_cloud_name_accuracy_rows)
        self.logger.info(df_to_str(sanity_check_cloud_name_to_accuracy_df))
        sanity_check_cloud_name_to_accuracy_df.to_csv(os.path.join(self.working_directory, 'svm_sanity_check_accuracy.csv'))

        sanity_check_wrongly_classified_samples_df = pd.DataFrame(missed_samples_ids_rows)
        self.logger.info(df_to_str(sanity_check_wrongly_classified_samples_df))
        sanity_check_wrongly_classified_samples_df.to_csv(
            os.path.join(self.working_directory, 'sanity_check_wrongly_classified_samples.csv'))


        svm_predicted_left_out_labels = linear_svm_classifier.predict(training_and_left_out_tensors.true_left_out_z)

        left_out_true_labels = left_out_predicted_cloud.true_labels[:len(svm_predicted_left_out_labels)]

        svm_acc = accuracy_score(left_out_true_labels, svm_predicted_left_out_labels)
        self.logger.info(f'Left out SVM accuracy = {svm_acc}')

        correctly_classified_true_left_out_mask = left_out_true_labels == svm_predicted_left_out_labels
        left_out_correctly_classified_z = training_and_left_out_tensors.true_left_out_z[correctly_classified_true_left_out_mask]
        left_out_incorrectly_classified_z = training_and_left_out_tensors.true_left_out_z[~correctly_classified_true_left_out_mask]

        with open(os.path.join(self.working_directory, "score-z-closest.txt"), "w") as score_file:
            # Writing data to a file
            score_file.write(f"{training_evaluator_params.left_out_cloud_ref.tissue} {training_evaluator_params.left_out_cloud_ref.perturbation} {svm_acc}")

        # Write to file incorrect details
        incorrectly_classified_left_out_cmap_id = training_evaluator_params.left_out_cmap_dataset.samples_cmap_id[~correctly_classified_true_left_out_mask]
        incorrectly_classified_svm_predicted_labels = svm_predicted_left_out_labels[~correctly_classified_true_left_out_mask]
        incorrectly_classified_desired_labels = left_out_true_labels[~correctly_classified_true_left_out_mask]
        incorrectly_classified_display_names = list(
                     map(training_evaluator_params.updated_training_cmap_dataset.original_label_to_display_name.get,
                         incorrectly_classified_svm_predicted_labels))
        wrongly_classified_count_df = pd.DataFrame({
            'display_name': incorrectly_classified_display_names,
            'desired_label': incorrectly_classified_desired_labels,
            'svm_predicted_label': incorrectly_classified_svm_predicted_labels,
            'cmap_id': incorrectly_classified_left_out_cmap_id
        })
        wrongly_classified_count_df.sort_values(['display_name', 'cmap_id'], inplace=True)
        wrongly_classified_count_df.to_csv(os.path.join(self.working_directory, 'true_left_out_svm_wrongly_classified_samples.csv'))
        display_name_values_counts_df = wrongly_classified_count_df['display_name'].value_counts()

        write_str_to_file(self.working_directory, "wrongly_classified_closest_list.txt",
                          f'Wrongly classified {len(incorrectly_classified_svm_predicted_labels)}/'
                          f'{len(training_evaluator_params.left_out_cmap_dataset)}:\n'
                          f'Full details:\n'
                          f"{'='*50}\n"
                          f'{df_to_str(display_name_values_counts_df)}\n'
                          f'Counts:\n'
                          f"{'='*50}\n"
                          f'{df_to_str(wrongly_classified_count_df)}')

        return SvmPredictionOnLeftOut(
            accuracy_score=svm_acc,
            true_left_out_labels=left_out_true_labels,
            predicted_left_out_labels=svm_predicted_left_out_labels,
            correctly_classified_left_out_z=left_out_correctly_classified_z,
            incorrectly_classified_left_out_z=left_out_incorrectly_classified_z,
            svm=linear_svm_classifier
        )
