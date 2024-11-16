import os
from typing import List, Dict

import numpy as np
import torch
from prefect import Parameter

from src.cmap_cloud_ref import CmapCloudRef
from src.configuration import config
from src.data_reduction_utils import DataReductionTool
from src.drawer import Drawer
from src.models.dudi_basic.post_training_evaluators.apply_svm_on_left_out_task import SvmPredictionOnLeftOut
from src.models.dudi_basic.post_training_evaluators.calculate_predicted_clouds_task import PredictedCloud
from src.models.dudi_basic.post_training_evaluators.create_data_tensors_for_evaluation_task import \
    TrainingAndLeftOutTensors
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.os_utilities import create_dir_if_not_exists
from src.tasks.abstract_tasks.experiment_task import ExperimentTask


class PostTrainingLatentSpaceDrawerTask(ExperimentTask):
    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            training_evaluator_params: TrainingEvaluatorParams,
            training_and_left_out_tensors: TrainingAndLeftOutTensors,
            cloud_ref_to_predicted_cloud: Dict[CmapCloudRef, PredictedCloud],
            svm_prediction_on_left_out: SvmPredictionOnLeftOut,
            data_reduction_tool: DataReductionTool,
            include_training_predicted: bool
            ):
        self.logger.info(f'Drawing latent space using {data_reduction_tool.reduction_algo_name}')
        model = training_evaluator_params.model
        model.eval()
        with_predicted_subfolder_name = 'with_predicted' if include_training_predicted else ''
        drawer_path = os.path.join(self.working_directory, data_reduction_tool.reduction_algo_name, with_predicted_subfolder_name)
        create_dir_if_not_exists(drawer_path)
        drawer = Drawer(self.logger, drawer_path)

        predicted_left_out = cloud_ref_to_predicted_cloud[training_evaluator_params.left_out_cloud_ref]
        left_out_display_name = training_evaluator_params.updated_full_cmap_dataset.encoded_label_to_display_name[
            predicted_left_out.full_cmap_encoded_label]

        predicted_left_out_display_name = f'PREDICTED LEFT OUT {left_out_display_name}'
        correctly_classified_left_out_display_name = f'CORRECTLY CLASSIFIED {left_out_display_name}'
        incorrectly_classified_left_out_display_name = f'INCORRECTLY CLASSIFIED {left_out_display_name}'

        predicted_left_out_z = predicted_left_out.predicted_z
        correctly_left_out_z = svm_prediction_on_left_out.correctly_classified_left_out_z
        incorrectly_left_out_z = svm_prediction_on_left_out.incorrectly_classified_left_out_z

        predicted_trained_clouds_z = np.empty((0, predicted_left_out_z.shape[1]))
        predicted_trained_clouds_display_names = []
        if include_training_predicted:
            predicted_trained_clouds_z = []
            predicted_trained_clouds = [predicted_cloud for
                                        cloud_ref, predicted_cloud in
                                        cloud_ref_to_predicted_cloud.items()
                                        if not (cloud_ref == training_evaluator_params.left_out_cloud_ref)]
            for predicted_cloud in predicted_trained_clouds:
                predicted_trained_clouds_z.append(predicted_cloud.predicted_z)
                predicted_trained_clouds_display_names.append(
                    [f'PREDICTED {predicted_cloud.display_name}'] * len(predicted_cloud.predicted_z))

            predicted_trained_clouds_z = np.concatenate(predicted_trained_clouds_z)
            predicted_trained_clouds_display_names = np.concatenate(predicted_trained_clouds_display_names)

        training_and_predicted_and_left_out_z = np.concatenate((
            training_and_left_out_tensors.training_z,
            predicted_left_out_z,
            correctly_left_out_z,
            incorrectly_left_out_z,
            predicted_trained_clouds_z))

        training_len = len(training_evaluator_params.updated_training_cmap_dataset)
        predicted_left_out_len = len(predicted_left_out_z)
        correct_true_left_out_len = len(correctly_left_out_z)
        incorrect_true_left_out_len = len(incorrectly_left_out_z)

        training_and_predicted_and_left_out_display_names = np.concatenate((
            training_evaluator_params.updated_training_cmap_dataset.display_names,
            [predicted_left_out_display_name] * predicted_left_out_len,
            [correctly_classified_left_out_display_name] * correct_true_left_out_len,
            [incorrectly_classified_left_out_display_name] * incorrect_true_left_out_len,
            predicted_trained_clouds_display_names))

        if training_evaluator_params.embedding_dim > 2:
            #encode text strings to integer for LDA
            univals, encode = np.unique(training_and_predicted_and_left_out_display_names, return_inverse=True)
            training_and_predicted_and_left_out_2d = \
                data_reduction_tool.to_2d_impl(training_and_predicted_and_left_out_z, encode)
        else:
            training_and_predicted_and_left_out_2d = training_and_predicted_and_left_out_z

        specific_display_label_colors = {
            predicted_left_out_display_name: 'Peach',
            correctly_classified_left_out_display_name: 'Electric Lime',
            incorrectly_classified_left_out_display_name: 'Black'
        }

        drawer.plot_2d_scatter(
            training_and_predicted_and_left_out_2d,
            training_and_predicted_and_left_out_display_names,
            specific_display_label_colors=specific_display_label_colors,
            title='Training & Left Out on z_sampled',
            file_name=f'all_clouds.png')

        # Draw final plot on specific tissues
        specific_display_label_colors_for_tissue = dict(specific_display_label_colors)
        other_display_name = 'OTHER'
        specific_display_label_colors_for_tissue[other_display_name] = 'Timberwolf'

        predicted_only_left_out_display_names = np.concatenate((
            [other_display_name] * training_len,
            [predicted_left_out_display_name] * predicted_left_out_len,
            [other_display_name] * correct_true_left_out_len,
            [other_display_name] * incorrect_true_left_out_len,
            predicted_trained_clouds_display_names))
        drawer.plot_2d_scatter(
            training_and_predicted_and_left_out_2d,
            predicted_only_left_out_display_names,
            specific_display_label_colors=specific_display_label_colors_for_tissue,
            title='Predicted Left Out Only on z_sampled',
            file_name=f'predicted_left_out_only.png')

        correct_true_left_out_only_left_out_display_names = np.concatenate((
            [other_display_name] * training_len,
            [other_display_name] * predicted_left_out_len,
            [correctly_classified_left_out_display_name] * correct_true_left_out_len,
            [other_display_name] * incorrect_true_left_out_len,
            [other_display_name] * len(predicted_trained_clouds_display_names)))
        drawer.plot_2d_scatter(
            training_and_predicted_and_left_out_2d,
            correct_true_left_out_only_left_out_display_names,
            specific_display_label_colors=specific_display_label_colors_for_tissue,
            title='Correct True Left Out Only on z_sampled',
            file_name=f'correct_true_left_out_only.png')

        incorrect_true_left_out_only_left_out_display_names = np.concatenate((
            [other_display_name] * training_len,
            [other_display_name] * predicted_left_out_len,
            [other_display_name] * correct_true_left_out_len,
            [incorrectly_classified_left_out_display_name] * incorrect_true_left_out_len,
            [other_display_name] * len(predicted_trained_clouds_display_names)))
        drawer.plot_2d_scatter(
            training_and_predicted_and_left_out_2d,
            incorrect_true_left_out_only_left_out_display_names,
            specific_display_label_colors=specific_display_label_colors_for_tissue,
            title='Incorrect True Left Out Only on z_sampled',
            file_name=f'incorrect_true_left_out_only.png')

        for tissue_code in config.tissue_code_to_name.keys():
            tissue_idx = np.array(np.frompyfunc(lambda display_name: tissue_code in display_name, 1, 1)(
                training_and_predicted_and_left_out_display_names), dtype=np.bool8)
            # tissue_idx = np.char.find(training_and_predicted_and_left_out_display_names, tissue_code) != -1
            display_names = np.copy(training_and_predicted_and_left_out_display_names)
            display_names[~tissue_idx] = 'OTHER'
            drawer.plot_2d_scatter(
                training_and_predicted_and_left_out_2d,
                display_names,
                specific_display_label_colors=specific_display_label_colors_for_tissue,
                title=f'Training & Left Out on z_sampled, only {tissue_code}',
                file_name=f'all_clouds_z_sampled_{tissue_code}.png')
            training_and_predicted_and_left_out_z_sp = training_and_predicted_and_left_out_z[tissue_idx]
            display_names_sp = display_names[tissue_idx]
            #encode text strings to integer for LDA
            univals, encode = np.unique(display_names_sp, return_inverse=True)
            if training_evaluator_params.embedding_dim > 2:
                training_and_predicted_and_left_out_2d_sp = \
                    data_reduction_tool.to_2d_impl(training_and_predicted_and_left_out_z_sp, encode)
            else:
                training_and_predicted_and_left_out_2d_sp = training_and_predicted_and_left_out_z_sp
            drawer.plot_2d_scatter(
                training_and_predicted_and_left_out_2d_sp,
                display_names_sp,
                specific_display_label_colors=specific_display_label_colors_for_tissue,
                title=f'Training & Left Out on z_sampled, focused {tissue_code}',
                file_name=f'all_clouds_z_sampled_{tissue_code}_focused.png')
            if training_and_predicted_and_left_out_2d_sp.shape[1]>2:
                drawer.plot_3d_scatter(
                    training_and_predicted_and_left_out_2d_sp,
                    display_names_sp,
                    specific_display_label_colors=specific_display_label_colors_for_tissue,
                    title=f'Training & Left Out on z_sampled, focused {tissue_code}',
                    file_name=f'all_clouds_z_sampled_{tissue_code}_focused_3D.png')