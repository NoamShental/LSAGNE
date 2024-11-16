from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from logging import Logger
from os import path
from typing import List, Dict, Tuple, Collection, Optional, Iterable, Callable

import numpy as np
import torch
from numpy.typing import NDArray

from src.assertion_utils import assert_same_len
from src.cmap_cloud_ref_and_tag import CmapCloudRefAndTag
from src.configuration import config
from src.data_reduction_utils import DataReductionTool
from src.drawer import Drawer
from src.models.cmap_cloud_tag import CmapCloudTag
from src.models.svm_utils import CmapSvmEvaluationResults
from src.os_utilities import create_dir_if_not_exists

# class DrawerCmapCloudTag(Enum):
#     REAL_TRAINED = 'Real Trained'
#     PREDICTED_TRAINED = 'Predicted Trained'
#     TRAINING_CONCEALED = 'Training Concealed'
#     LEFT_OUT = 'Left Out'
#     PREDICTED_LEFT_OUT = 'Predicted Left Out'
#     CV = 'CV'
#     PREDICTED_CV = 'Predicted CV'



# @dataclass
# class CloudToDraw:
#     cloud_ref: CmapCloudRef
#     tag: CmapCloudTag
#     samples_after_reduction: NDArray[float]
#     display_name: str
#     color: Optional[str]


TRAINED_COLOR = 'Peach'
CORRECTLY_PREDICTED_COLOR = 'Electric Lime'
INCORRECTLY_PREDICTED_COLOR = 'Black'
OTHER_COLOR = 'Gray'


@dataclass(frozen=True)
class SvmResultsDrawer:
    logger: Logger
    drawer: Drawer
    svm_results: CmapSvmEvaluationResults
    samples_2d: NDArray[float]
    display_labels: NDArray[str]
    cloud_ref_and_tag_to_idx: Dict[CmapCloudRefAndTag, NDArray[int]]
    other_samples_display_name: str = 'OTHER'

    def __post_init__(self):
        assert_same_len(self.samples_2d, self.display_labels)

    @staticmethod
    def _filter_by_tag(
            all_cloud_refs_and_tags: Iterable[CmapCloudRefAndTag],
            tag: CmapCloudTag
    ) -> List[CmapCloudRefAndTag]:
        return [
            cloud_ref_and_tag
            for cloud_ref_and_tag in all_cloud_refs_and_tags
            if cloud_ref_and_tag.tag == tag
        ]

    @cached_property
    def tag_to_cloud_refs_and_tags(self) -> Dict[CmapCloudTag, List[CmapCloudRefAndTag]]:
        tag_to_cloud_refs_and_tags: Dict[CmapCloudTag, List[CmapCloudRefAndTag]] = {}
        for tag in [cloud_ref_and_tag.tag for cloud_ref_and_tag in self.all_cloud_refs_and_tags]:
            tag_to_cloud_refs_and_tags[tag] = self._filter_by_tag(self.all_cloud_refs_and_tags, tag)
        return tag_to_cloud_refs_and_tags

    @cached_property
    def all_cloud_refs_and_tags(self) -> List[CmapCloudRefAndTag]:
        return list(self.cloud_ref_and_tag_to_idx.keys())

    def draw(
            self,
            title: str,
            file_name: str,
            highlight_trained_and_predicted_svm_prediction: Optional[Tuple[CmapCloudRefAndTag, CmapCloudRefAndTag]],
            tags_to_exclude: Iterable[CmapCloudTag] = None,
            cloud_ref_and_tag_to_graying_predicate: Optional[Callable[[CmapCloudRefAndTag], bool]] = None
    ):
        if highlight_trained_and_predicted_svm_prediction:
            trained_highlight, predicted_highlight = highlight_trained_and_predicted_svm_prediction
            assert any([trained_highlight.cloud_ref == cloud.cloud_ref and trained_highlight.tag == cloud.tag
                        for cloud in self.svm_results.training_cmap_clouds])
            assert predicted_highlight in self.svm_results.cloud_ref_and_tag_to_prediction_result
        else:
            trained_highlight, predicted_highlight = None, None
        if not cloud_ref_and_tag_to_graying_predicate:
            cloud_ref_and_tag_to_graying_predicate = lambda _: False
        if not tags_to_exclude:
            tags_to_exclude = []
        specific_display_label_colors: Dict[str, str] = {}
        samples_to_draw_np_list = []
        display_labels_to_draw_np_list = []
        for cloud_ref_and_tag in self.all_cloud_refs_and_tags:
            if cloud_ref_and_tag.tag in tags_to_exclude:
                continue
            idx = self.cloud_ref_and_tag_to_idx[cloud_ref_and_tag]
            if cloud_ref_and_tag == predicted_highlight:
                correct_mask = self.svm_results.cloud_ref_and_tag_to_prediction_result[predicted_highlight].absolute_svm_correct_mask
                correct_idx = idx[correct_mask]
                incorrect_idx = idx[~correct_mask]
                correct_perc = len(correct_idx) / len(correct_mask) * 100
                incorrect_perc = len(incorrect_idx) / len(correct_mask) * 100
                correct_predicted_highlight_name = f'CORRECT {predicted_highlight.name} ({correct_perc:.2f}%)'
                incorrect_predicted_highlight_name = f'INCORRECT {predicted_highlight.name} ({incorrect_perc:.2f}%)'
                specific_display_label_colors.update({
                    correct_predicted_highlight_name: CORRECTLY_PREDICTED_COLOR,
                    incorrect_predicted_highlight_name: INCORRECTLY_PREDICTED_COLOR
                })
                samples_to_draw_np_list.append(self.samples_2d[correct_idx])
                display_labels_to_draw_np_list.append(np.full(len(correct_idx), correct_predicted_highlight_name))
                samples_to_draw_np_list.append(self.samples_2d[incorrect_idx])
                display_labels_to_draw_np_list.append(np.full(len(incorrect_idx), incorrect_predicted_highlight_name))
            else:
                samples_to_draw_np_list.append(self.samples_2d[idx])
                if cloud_ref_and_tag == trained_highlight:
                    true_highlight_name = f'SVM TRAINED {trained_highlight.name}'
                    display_labels = np.full(len(idx), true_highlight_name)
                    specific_display_label_colors[true_highlight_name] = TRAINED_COLOR
                else:
                    display_labels = np.full(len(idx), self.other_samples_display_name) \
                        if cloud_ref_and_tag_to_graying_predicate(cloud_ref_and_tag) else self.display_labels[idx]
                    specific_display_label_colors[self.other_samples_display_name] = OTHER_COLOR
                display_labels_to_draw_np_list.append(display_labels)

        samples_to_draw = np.vstack(samples_to_draw_np_list)
        display_labels_to_draw = np.hstack(display_labels_to_draw_np_list)
        display_labels_importance = list(specific_display_label_colors.keys())
        for display_label in np.unique(display_labels_to_draw):
            if display_label not in display_labels_importance:
                display_labels_importance.append(display_label)
        if self.other_samples_display_name in display_labels_importance:
            display_labels_importance.remove(self.other_samples_display_name)
        self.drawer.plot_2d_scatter(
            samples_to_draw,
            display_labels_to_draw,
            specific_display_label_colors=specific_display_label_colors,
            title=title,
            file_name=file_name,
            display_labels_importance=display_labels_importance
        )

    @classmethod
    def create(
            cls,
            logger: Logger,
            working_directory: str,
            draw_folder_name: str,
            svm_name: str,
            svm_results: CmapSvmEvaluationResults,
            data_reduction_tool: DataReductionTool,
            tag: str = '',
            tags_to_exclude: Optional[Collection[CmapCloudTag]] = None,
    ) -> SvmResultsDrawer:
        if tags_to_exclude is None:
            tags_to_exclude = []
        drawer_path = path.join(working_directory, draw_folder_name, svm_name, data_reduction_tool.reduction_algo_name, tag)
        create_dir_if_not_exists(drawer_path)
        logger.info(f'Creating svm results drawer using {data_reduction_tool.reduction_algo_name} to folder "{drawer_path}".')
        drawer = Drawer(logger, drawer_path)

        cloud_ref_and_tag_to_idx: Dict[CmapCloudRefAndTag, NDArray[int]] = {}
        last_idx = 0
        samples_np_list = []
        display_labels_list: List[NDArray[str]] = []

        for cmap_cloud in [
            *svm_results.training_cmap_clouds,
            *list(svm_results.cloud_ref_and_tag_to_prediction_result.values())
        ]:
            if cmap_cloud.tag in tags_to_exclude:
                continue
            cloud_ref_and_tag = CmapCloudRefAndTag(
                cloud_ref=cmap_cloud.cloud_ref,
                tag=cmap_cloud.tag
            )
            cloud_ref_and_tag_to_idx[cloud_ref_and_tag] = np.arange(last_idx, last_idx + len(cmap_cloud))
            last_idx += len(cmap_cloud)
            samples_np_list.append(cmap_cloud.samples)
            display_labels_list.append(np.array([cloud_ref_and_tag.name] * len(cmap_cloud)))
        samples = np.vstack(samples_np_list)
        display_labels = np.hstack(display_labels_list)
        samples_2d = data_reduction_tool.to_2d(samples, display_labels)

        return cls(
            logger=logger,
            drawer=drawer,
            svm_results=svm_results,
            samples_2d=samples_2d,
            display_labels=display_labels,
            cloud_ref_and_tag_to_idx=cloud_ref_and_tag_to_idx
        )
