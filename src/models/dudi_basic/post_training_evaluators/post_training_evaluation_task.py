import os
from typing import List

import torch
from prefect import Parameter

from src.cmap_cloud_ref_and_tag import CmapCloudRefAndTag
from src.cmap_evaluation_data import SplittedCmapEvaluationData
from src.configuration import config
from src.data_reduction_utils import PCAReductionTool, DataReductionTool, TsneReductionTool
from src.models.cmap_cloud_tag import CmapCloudTag
from src.models.dudi_basic.embedded_anchors_and_vectors import EmbeddedAnchorsAndVectors
from src.models.dudi_basic.model_learning_parameters import InputLearningParameters
from src.models.dudi_basic.multi_device_data import AnchorPointsLookup
from src.models.dudi_basic.post_training_evaluators.training_evaluator_params import TrainingEvaluatorParams
from src.models.latent_space_drawer import SvmResultsDrawer
from src.models.svm_utils import perform_svm_accuracy_evaluation, CmapSvmEvaluationResults
from src.pipeline_utils import choose_device
from src.tasks.abstract_tasks.experiment_task import ExperimentTask
from src.training_summary import TrainingSummary


class PostTrainingEvaluationTask(ExperimentTask):

    def __init__(self, flow_parameters: List[Parameter], *args, **kwargs):
        super().__init__(cache_task=False, flow_parameters=flow_parameters, *args, **kwargs)

    @torch.no_grad()
    def run(self,
            params: InputLearningParameters,
            training_summary: TrainingSummary) -> TrainingEvaluatorParams:
        device = choose_device(params.use_cuda, self.logger)

        torch.set_default_dtype(config.torch_numeric_precision_type)

        # Necessary when the task result is loaded from pickled file
        training_summary.model.load_state_dict(training_summary.model_state_dict)
        training_summary.model.to(device)

        epoch_evaluation_data = SplittedCmapEvaluationData.create_instance(
            training_only_cmap=training_summary.cmap_datasets.training_only,
            training_concealed_cmap=training_summary.cmap_datasets.training_concealed,
            model=training_summary.model,
            cross_validation=training_summary.cmap_datasets.cross_validation,
            left_out=training_summary.cmap_datasets.left_out
        )

        on_device_original_space_anchor_points: AnchorPointsLookup = AnchorPointsLookup.create_from_anchor_points(
            training_summary.anchor_points,
            device
        )
        embedded_anchors_and_vectors: EmbeddedAnchorsAndVectors = EmbeddedAnchorsAndVectors.create(
            original_space_anchor_points_lookup=on_device_original_space_anchor_points,
            embedder=training_summary.model
        )

        all_svm_results = perform_svm_accuracy_evaluation(
            splitted_evaluation_data=epoch_evaluation_data,
            embedded_anchors_and_vectors=embedded_anchors_and_vectors,
            predicted_cloud_max_size=training_summary.params.predicted_cloud_max_size,
            random_seed=training_summary.params.random_manager.random_seed,
            perturbations_equivalence_sets=training_summary.params.perturbations_equivalence_sets
        )

        for data_reduction_tool in [
            PCAReductionTool(),
            TsneReductionTool(training_summary.params.random_manager.random_seed)
        ]:
            self.logger.info(f'Drawing using {data_reduction_tool.reduction_algo_name}')
            self.draw_svm_results(
                drawing_root_dir=os.path.join(training_summary.params.working_directory, 'plots'),
                svm_name='svm_1',
                i_epoch=training_summary.best_model_epoch,
                svm_results=all_svm_results.svm_1,
                training_summary=training_summary,
                data_reduction_tool=data_reduction_tool
            )
        self.logger.info(f'CLS-DIR (true): {round(all_svm_results.svm_1.summary["equivalence_sets_svm_acc"].mean() * 100, 2)}')
        self.logger.info(f'CLS-DIR (pred): {round(all_svm_results.svm_2.summary["equivalence_sets_svm_acc"].mean() * 100, 2)}')
        all_svm_results.log_and_save_to_dir(
            logger=self.logger,
            directory_path=os.path.join(training_summary.params.working_directory, 'svm_results', 'best_epoch'),
            i_epoch=training_summary.best_model_epoch
        )

    def draw_svm_results(
            self,
            drawing_root_dir: str,
            svm_name: str,
            i_epoch: int,
            svm_results: CmapSvmEvaluationResults,
            training_summary: TrainingSummary,
            data_reduction_tool: DataReductionTool
    ):
        all_tags = {
            *[cmap_cloud.tag for cmap_cloud in svm_results.training_cmap_clouds],
            *[cloud_ref_and_tag.tag for cloud_ref_and_tag in svm_results.cloud_ref_and_tag_to_prediction_result]
        }
        svm_results_drawer = SvmResultsDrawer.create(
            logger=self.logger,
            working_directory=drawing_root_dir,
            draw_folder_name=f'best_epoch_{i_epoch:05d}',
            svm_name=svm_name,
            svm_results=svm_results,
            data_reduction_tool=data_reduction_tool,
            tag='all tissues'
        )
        self.logger.info(f'Drawing all clouds')
        svm_results_drawer.draw(
            title=f'{svm_name} All',
            file_name='all_clouds.png',
            highlight_trained_and_predicted_svm_prediction=(
                CmapCloudRefAndTag(training_summary.params.left_out_cloud, CmapCloudTag.PREDICTED_LEFT_OUT),
                CmapCloudRefAndTag(training_summary.params.left_out_cloud, CmapCloudTag.LEFT_OUT)
            )
        )
        self.logger.info(f'Drawing left out focused')
        svm_results_drawer.draw(
            title=f'{svm_name} Left Out',
            file_name='left_out_focus.png',
            highlight_trained_and_predicted_svm_prediction=(
                CmapCloudRefAndTag(training_summary.params.left_out_cloud, CmapCloudTag.PREDICTED_LEFT_OUT),
                CmapCloudRefAndTag(training_summary.params.left_out_cloud, CmapCloudTag.LEFT_OUT)
            ),
            tags_to_exclude=all_tags - {CmapCloudTag.PREDICTED_LEFT_OUT, CmapCloudTag.LEFT_OUT}
        )
        all_trained_tissues = {
            cmap_cloud.cloud_ref.tissue
            for cmap_cloud in svm_results.training_cmap_clouds
        }
        for tissue in all_trained_tissues:
            self.logger.info(f'Drawing tissue {tissue} focused')
            tissue_drawer = SvmResultsDrawer.create(
                logger=self.logger,
                working_directory=drawing_root_dir,
                draw_folder_name=f'best_epoch_{i_epoch:05d}',
                svm_name=svm_name,
                svm_results=svm_results.limit_to_tissue(tissue),
                data_reduction_tool=data_reduction_tool,
                tag=f'tissue {tissue.tissue_code}',
                tags_to_exclude=[]
            )
            if training_summary.params.left_out_cloud.tissue == tissue:
                highlight_trained_and_predicted_svm_prediction = (
                    CmapCloudRefAndTag(training_summary.params.left_out_cloud, CmapCloudTag.PREDICTED_LEFT_OUT),
                    CmapCloudRefAndTag(training_summary.params.left_out_cloud, CmapCloudTag.LEFT_OUT)
                )
            else:
                highlight_trained_and_predicted_svm_prediction = None
            for drawer in [svm_results_drawer, tissue_drawer]:
                drawer.draw(
                    title=f'{svm_name} Tissue {tissue}',
                    file_name=f'{tissue.tissue_code} all focus.png',
                    highlight_trained_and_predicted_svm_prediction=highlight_trained_and_predicted_svm_prediction,
                    cloud_ref_and_tag_to_graying_predicate=lambda
                        cloud_ref_and_tag: cloud_ref_and_tag.cloud_ref.tissue != tissue
                )
                drawer.draw(
                    title=f'{svm_name} Tissue {tissue} Real Trained Only Focus',
                    file_name=f'{tissue.tissue_code} real trained only focus.png',
                    highlight_trained_and_predicted_svm_prediction=highlight_trained_and_predicted_svm_prediction,
                    tags_to_exclude=[CmapCloudTag.PREDICTED_TRAINED],
                    cloud_ref_and_tag_to_graying_predicate=lambda
                        cloud_ref_and_tag: cloud_ref_and_tag.cloud_ref.tissue != tissue
                )
                drawer.draw(
                    title=f'{svm_name} Tissue {tissue} Predicted Trained Only Focus',
                    file_name=f'{tissue.tissue_code} predicted trained only focus.png',
                    highlight_trained_and_predicted_svm_prediction=highlight_trained_and_predicted_svm_prediction,
                    tags_to_exclude=[CmapCloudTag.REAL_TRAINED],
                    cloud_ref_and_tag_to_graying_predicate=lambda
                        cloud_ref_and_tag: cloud_ref_and_tag.cloud_ref.tissue != tissue
                )
