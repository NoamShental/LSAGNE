from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from logging import Logger
from typing import List, Optional, Tuple, Collection, Dict, Set, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn import svm

from src.assertion_utils import assert_promise_true, assert_same_len
from src.class_encoder import ClassEncoder
from src.cmap_cloud import CmapCloud
from src.cmap_cloud_ref import CmapCloudRef
from src.cmap_cloud_ref_and_tag import CmapCloudRefAndTag
from src.cmap_evaluation_data import SplittedCmapEvaluationData
from src.dataframe_utils import df_to_str
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.cmap_cloud_tag import CmapCloudTag
from src.models.dudi_basic.embedded_anchors_and_vectors import EmbeddedAnchorsAndVectors
from src.models.predicted_clouds_calculator import predicted_clouds_calculator
from src.perturbation import Perturbation
from src.tissue import Tissue
from src.utils import check_if_in_same_perturbations_equivalence_set
from src.write_to_file_utils import write_str_to_file


def _train_linear_svm(x, labels, random_seed) -> svm.LinearSVC:
    # add C?
    linear_svm_classifier = svm.LinearSVC(random_state=random_seed,
                                          max_iter=10_000,
                                          C=1,
                                          # tol=1e-15,
                                          class_weight='balanced',
                                          dual=False)
    linear_svm_classifier.fit(x, labels)
    return linear_svm_classifier


@dataclass(frozen=True)
class CmapSvm:
    svm: svm.LinearSVC
    _cloud_refs_encoder: ClassEncoder[CmapCloudRef]

    @staticmethod
    def validate_cloud_ref_to_svm_label(cloud_ref_to_svm_label: Dict[CmapCloudRef, int]):
        unique_svm_labels = np.array(list(cloud_ref_to_svm_label.values()))
        assert_promise_true(lambda: unique_svm_labels.min() == 0 and
                                    unique_svm_labels.max() == len(cloud_ref_to_svm_label))

    @classmethod
    def train_svm(
            cls,
            svm_cmap_clouds: Collection[CmapCloud],
            random_seed: int
    ) -> CmapSvm:
        samples_list = []
        cloud_refs_list = []
        for svm_cmap_cloud in svm_cmap_clouds:
            samples_list.append(svm_cmap_cloud.samples)
            cloud_refs_list.append(svm_cmap_cloud.cloud_refs)
        samples = np.vstack(samples_list)
        cloud_refs = np.hstack(cloud_refs_list)
        cloud_refs_encoder = ClassEncoder(set(cloud_refs))
        svm_labels = cloud_refs_encoder.np_class_to_encoded_label_vectorize(cloud_refs)
        return cls(
            svm=_train_linear_svm(samples, svm_labels, random_seed),
            _cloud_refs_encoder=cloud_refs_encoder
        )

    def predict(self, samples: NDArray[float]) -> NDArray[CmapCloudRef]:
        return self._cloud_refs_encoder.np_encoded_label_to_class_vectorize(self.svm.predict(samples))


@dataclass(frozen=True)
class CmapSvmCloudPredictionResults(CmapCloud):
    absolute_svm_correct_mask: NDArray[bool]
    absolute_svm_accuracy: float
    equivalence_sets_svm_correct_mask: NDArray[bool]
    equivalence_sets_svm_accuracy: float
    svm_prediction: NDArray[CmapCloudRef]
    confusion_df: pd.DataFrame | None

    def __post_init__(self):
        assert 0 <= self.absolute_svm_accuracy <= 1
        assert 0 <= self.equivalence_sets_svm_accuracy <= 1

    @classmethod
    def _svm_prediction_confusion(
            cls,
            cmap_cloud: CmapCloud,
            svm_prediction: NDArray[CmapCloudRef],
            perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]]
    ) -> pd.DataFrame:
        df_rows = []
        for predicted_cloud_ref in {*svm_prediction.tolist(), cmap_cloud.cloud_ref}:
            mask = predicted_cloud_ref == svm_prediction
            count = mask.sum()

            if predicted_cloud_ref == cmap_cloud.cloud_ref:
                note = 'True Cloud'
            elif check_if_in_same_perturbations_equivalence_set(
                    cmap_cloud.cloud_ref,
                    predicted_cloud_ref,
                    perturbations_equivalence_sets
                ):
                note = 'Same Equivalence Set'
            else:
                note = ''

            df_rows.append({
                'correct_cloud': cmap_cloud.cloud_ref.name,
                'cloud_size': len(cmap_cloud),
                'type': cmap_cloud.tag.value,
                'predicted_cloud': predicted_cloud_ref.name,
                '#': count,
                '%': count / len(cmap_cloud),
                'note':  note,
                'cmap_ids': cmap_cloud.cmap_ids[mask] if cmap_cloud.cmap_ids is not None else None
            })
        return pd.DataFrame(df_rows)

    # @cached_property
    # def cloud_ref(self) -> CmapCloudRef:
    #     return self.cmap_cloud.cloud_ref
    #
    # @cached_property
    # def tag(self) -> CmapCloudTag:
    #     return self.cmap_cloud.tag
    #
    # @cached_property
    # def samples(self) -> NDArray[float]:
    #     return self.cmap_cloud.samples

    @cached_property
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tissue': self.cloud_ref.tissue,
            'perturbation': self.cloud_ref.perturbation,
            'cloud_ref': self.cloud_ref,
            'predicted_clouds': self.svm_prediction,
            'absolute_svm_acc': self.absolute_svm_accuracy,
            'equivalence_sets_svm_acc': self.equivalence_sets_svm_accuracy,
            'tag': self.tag.value
        }

    @classmethod
    def create(
        cls,
        cmap_cloud: CmapCloud,
        svm_prediction: NDArray[CmapCloudRef],
        perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]],
        calculate_confusion_df: bool = True
    ) -> CmapSvmCloudPredictionResults:
        absolute_svm_correct_mask = svm_prediction == cmap_cloud.cloud_ref
        absolute_svm_accuracy = absolute_svm_correct_mask.sum() / len(cmap_cloud)
        equivalence_sets_svm_correct_mask = np.vectorize(
            lambda current_cloud: check_if_in_same_perturbations_equivalence_set(
                cmap_cloud.cloud_ref,
                current_cloud,
                perturbations_equivalence_sets))(svm_prediction)
        equivalence_sets_svm_accuracy = equivalence_sets_svm_correct_mask.sum() / len(cmap_cloud)
        return cls(
            cloud_ref=cmap_cloud.cloud_ref,
            samples=cmap_cloud.samples,
            tag=cmap_cloud.tag,
            cmap_ids=cmap_cloud.cmap_ids,
            absolute_svm_accuracy=absolute_svm_accuracy,
            absolute_svm_correct_mask=absolute_svm_correct_mask,
            equivalence_sets_svm_accuracy=equivalence_sets_svm_accuracy,
            equivalence_sets_svm_correct_mask=equivalence_sets_svm_correct_mask,
            svm_prediction=svm_prediction,
            confusion_df=cls._svm_prediction_confusion(cmap_cloud, svm_prediction, perturbations_equivalence_sets) if calculate_confusion_df else None
        )


def predict_cmap_clouds_using_svm(
        cmap_svm: CmapSvm,
        cmap_clouds: Collection[CmapCloud],
        perturbations_equivalence_sets: Optional[Collection[Set[Perturbation]]] = None
) -> Tuple[pd.DataFrame, Dict[CmapCloudRefAndTag, CmapSvmCloudPredictionResults]]:
    cloud_level_svm_accuracy_df_rows = []
    cloud_ref_and_tag_to_prediction_result = {}
    for cmap_cloud in cmap_clouds:
        svm_predicted_cloud_refs = cmap_svm.predict(cmap_cloud.samples)
        cloud_prediction_results = CmapSvmCloudPredictionResults.create(cmap_cloud, svm_predicted_cloud_refs, perturbations_equivalence_sets)
        cloud_level_svm_accuracy_df_rows.append(cloud_prediction_results.to_dict)
        cloud_ref_and_tag_to_prediction_result[CmapCloudRefAndTag(cmap_cloud.cloud_ref, cmap_cloud.tag)] = cloud_prediction_results
    df = pd.DataFrame(cloud_level_svm_accuracy_df_rows)
    return df, cloud_ref_and_tag_to_prediction_result


@dataclass(frozen=True)
class CmapSvmEvaluationResults:
    training_cmap_clouds: Collection[CmapCloud]
    summary: pd.DataFrame
    cloud_ref_and_tag_to_prediction_result: Dict[CmapCloudRefAndTag, CmapSvmCloudPredictionResults]
    svm: CmapSvm

    def limit_to_tissue(self, tissue: Tissue) -> CmapSvmEvaluationResults:
        training_cmap_clouds = [
            cmap_cloud for cmap_cloud in self.training_cmap_clouds
            if cmap_cloud.cloud_ref.tissue == tissue
        ]
        cloud_ref_and_tag_to_prediction_result = {
            cloud_ref_and_tag: prediction_result
            for cloud_ref_and_tag, prediction_result in self.cloud_ref_and_tag_to_prediction_result.items()
            if cloud_ref_and_tag.cloud_ref.tissue == tissue
        }
        return CmapSvmEvaluationResults(
            training_cmap_clouds=training_cmap_clouds,
            summary=None,
            cloud_ref_and_tag_to_prediction_result=cloud_ref_and_tag_to_prediction_result,
            svm=self.svm
        )


@dataclass(frozen=True)
class AllCmapSvmEvaluationResults:
    svm_1: CmapSvmEvaluationResults
    svm_2: CmapSvmEvaluationResults

    @staticmethod
    def _log_and_save_to_dir_svm_result(
            logger: Logger,
            directory_path: str,
            svm_summary: pd.DataFrame,
            cloud_ref_and_tag_to_prediction_result: Dict[CmapCloudRefAndTag, CmapSvmCloudPredictionResults],
            svm_name: str,
            svm_notes: str
    ) -> None:
        svm_summary_str_for_logger = df_to_str(svm_summary, ['tissue', 'perturbation', 'absolute_svm_acc', 'equivalence_sets_svm_acc', 'tag'])
        logger.info(f"{svm_name} Summary:\n" + svm_notes + '\n\n' + svm_summary_str_for_logger)
        svm_folder = os.path.join(directory_path, svm_name)
        for cloud_ref_and_tag, prediction_results in cloud_ref_and_tag_to_prediction_result.items():
            confusion_df_str = df_to_str(prediction_results.confusion_df)
            confusion_df_file_name = f'{prediction_results.cloud_ref.name} - {prediction_results.tag.value.lower()} - confusion matrix.txt'
            confusion_df_str += '\n\n'
            confusion_df_str += '=' * 50 + '\n'
            confusion_df_str += 'Confused with CMAP ids:' + '\n'
            confusion_df_str += '=' * 50 + '\n\n'
            cmap_id_with_cloud_name = []
            for _, confusion_df_row in prediction_results.confusion_df.iterrows():
                cloud_name = confusion_df_row['predicted_cloud']
                cmap_ids = confusion_df_row['cmap_ids']
                if cmap_ids is not None:
                    for cmap_id in cmap_ids:
                        cmap_id_with_cloud_name.append((cmap_id, cloud_name))
            # getting sorted by the first element in the tuple
            cmap_id_with_cloud_name.sort()
            for i, (cmap_id, cloud_name) in enumerate(cmap_id_with_cloud_name):
                confusion_df_str += f'{i + 1}: {cmap_id} ({cloud_name})\n'
            write_str_to_file(svm_folder, confusion_df_file_name, confusion_df_str)
        write_str_to_file(svm_folder, 'summary', svm_notes + '\n\n' + df_to_str(svm_summary))

    def log_and_save_to_dir(self, logger: Logger, directory_path: str, i_epoch: int) -> None:
        folder_path = os.path.join(directory_path, f'epoch_{i_epoch:05d}')
        self._log_and_save_to_dir_svm_result(
            logger,
            folder_path,
            self.svm_1.summary,
            self.svm_1.cloud_ref_and_tag_to_prediction_result,
            "svm_1",
            "(Trained on: all true training clouds + predicted CV + predicted left out\n" +
            "Tested on: predicted non-DMSO trained clouds + true training concealed + true CV + true left out)"
        )

        self._log_and_save_to_dir_svm_result(
            logger,
            folder_path,
            self.svm_2.summary,
            self.svm_2.cloud_ref_and_tag_to_prediction_result,
            "svm_2",
            "(Trained on: all true DMSO training clouds + all predicted non-DMSO training clouds + predicted CV + predicted left out\n" +
            "Tested on: true non-DMSO trained clouds + true training concealed + true CV + true left out)\n"
        )


def _split_to_svm_cmap_clouds(
        samples: NDArray[float],
        cmap: RawCmapDataset,
        tag: CmapCloudTag
) -> List[CmapCloud]:
    assert_same_len(samples, cmap)
    svm_cmap_clouds = []
    for cloud_ref, idx in cmap.cloud_ref_to_idx.items():
        svm_cmap_clouds.append(CmapCloud(
            cloud_ref=cloud_ref,
            samples=samples[idx],
            tag=tag,
            cmap_ids=cmap.samples_cmap_id[idx]
        ))
    return svm_cmap_clouds


def perform_svm_accuracy_evaluation(
        splitted_evaluation_data: SplittedCmapEvaluationData,
        embedded_anchors_and_vectors: EmbeddedAnchorsAndVectors,
        predicted_cloud_max_size: int,
        random_seed: int,
        perturbations_equivalence_sets: Collection[Collection[Perturbation]]
) -> AllCmapSvmEvaluationResults:
    updated_full_cmap_dataset = RawCmapDataset.merge_datasets(
        splitted_evaluation_data.training_only.cmap,
        splitted_evaluation_data.training_concealed.cmap,
        splitted_evaluation_data.left_out.cmap,
        splitted_evaluation_data.cross_validation.cmap
    )

    training_only_z = splitted_evaluation_data.training_only.z_t.cpu().numpy()
    training_concealed_z = splitted_evaluation_data.training_concealed.z_t.cpu().numpy()
    cross_validation_z = splitted_evaluation_data.cross_validation.z_t.cpu().numpy()
    left_out_z = splitted_evaluation_data.left_out.z_t.cpu().numpy()

    # calculate predicted cloud for each non DMSO cloud
    non_dmso_cloud_ref_to_predicted_cloud = predicted_clouds_calculator(
        all_cloud_refs=updated_full_cmap_dataset.unique_cloud_refs,
        training_cmap_dataset=splitted_evaluation_data.training_only.cmap,
        embedded_anchors_and_vectors=embedded_anchors_and_vectors,
        training_z=training_only_z,
        predicted_cloud_max_size=predicted_cloud_max_size
    )

    predicted_left_out_svm_cmap_clouds = []
    for cloud_ref in splitted_evaluation_data.left_out.cmap.unique_cloud_refs:
        predicted_left_out_svm_cmap_clouds.append(
            CmapCloud(
                cloud_ref=cloud_ref,
                samples=non_dmso_cloud_ref_to_predicted_cloud[cloud_ref].predicted_z,
                tag=CmapCloudTag.PREDICTED_LEFT_OUT,
                cmap_ids=None
            )
        )

    predicted_cv_svm_cmap_clouds = []
    for cloud_ref in splitted_evaluation_data.cross_validation.cmap.unique_cloud_refs:
        predicted_cv_svm_cmap_clouds.append(
            CmapCloud(
                cloud_ref=cloud_ref,
                samples=non_dmso_cloud_ref_to_predicted_cloud[cloud_ref].predicted_z,
                tag=CmapCloudTag.PREDICTED_CV,
                cmap_ids=None
            )
        )

    # SVM #1 ->
    # Train on: All trained clouds + predicted cv, left out
    # Test on:  Train concealed + true left out + true CV

    # Train SVM #1

    all_training_svm_cmap_clouds = _split_to_svm_cmap_clouds(
        samples=training_only_z,
        cmap=splitted_evaluation_data.training_only.cmap,
        tag=CmapCloudTag.REAL_TRAINED
    )

    svm_1_training_cmap_clouds = [
        *all_training_svm_cmap_clouds,
        *predicted_left_out_svm_cmap_clouds,
        *predicted_cv_svm_cmap_clouds
    ]

    svm_1 = CmapSvm.train_svm(
        svm_cmap_clouds=svm_1_training_cmap_clouds,
        random_seed=random_seed
    )
    # Test SVM #1
    test_non_trained_cmap_clouds = [
        *_split_to_svm_cmap_clouds(
            samples=left_out_z,
            cmap=splitted_evaluation_data.left_out.cmap,
            tag=CmapCloudTag.LEFT_OUT
        ),
        *_split_to_svm_cmap_clouds(
            samples=cross_validation_z,
            cmap=splitted_evaluation_data.cross_validation.cmap,
            tag=CmapCloudTag.CV
        ),
        *_split_to_svm_cmap_clouds(
            samples=training_concealed_z,
            cmap=splitted_evaluation_data.training_concealed.cmap,
            tag=CmapCloudTag.TRAINING_CONCEALED
        )
    ]

    predicted_non_dmso_trained_cmap_clouds: List[CmapCloud] = []
    for cloud_ref in splitted_evaluation_data.training_only.cmap.unique_cloud_refs:
        if not cloud_ref.is_not_dmso_6h_or_24h:
            continue
        predicted_non_dmso_trained_cmap_clouds.append(CmapCloud(
            cloud_ref=cloud_ref,
            samples=non_dmso_cloud_ref_to_predicted_cloud[cloud_ref].predicted_z,
            tag=CmapCloudTag.PREDICTED_TRAINED,
            cmap_ids=None
        ))

    summary_svm_1_df, svm_1_cloud_ref_and_tag_to_prediction_result = predict_cmap_clouds_using_svm(
        svm_1,
        [*test_non_trained_cmap_clouds, *predicted_non_dmso_trained_cmap_clouds],
        perturbations_equivalence_sets
    )

    # SVM #2 ->
    # Train on: True DMSO trained clouds + predicted non DMSO trained clouds + predicted cv, left out
    # Test on:  True non DMSO trained clouds, train concealed + true left out + true CV

    # Train SVM #2
    all_dmso_training_svm_cmap_clouds = [
        svm_cmap_cloud for svm_cmap_cloud in all_training_svm_cmap_clouds
        if not svm_cmap_cloud.cloud_ref.is_not_dmso_6h_or_24h
    ]

    svm_2_training_cmap_clouds = [
        *all_dmso_training_svm_cmap_clouds,
        *predicted_non_dmso_trained_cmap_clouds,
        *predicted_left_out_svm_cmap_clouds,
        *predicted_cv_svm_cmap_clouds
    ]

    svm_2 = CmapSvm.train_svm(
        svm_cmap_clouds=svm_2_training_cmap_clouds,
        random_seed=random_seed
    )

    # Test SVM #2
    test_non_dmso_trained_clouds = [
        svm_cmap_cloud for svm_cmap_cloud in all_training_svm_cmap_clouds
        if svm_cmap_cloud.cloud_ref.is_not_dmso_6h_or_24h
    ]

    summary_svm_2_df, svm_2_cloud_ref_and_tag_to_prediction_result = predict_cmap_clouds_using_svm(
        svm_2,
        [*test_non_trained_cmap_clouds, *test_non_dmso_trained_clouds],
        perturbations_equivalence_sets
    )

    return AllCmapSvmEvaluationResults(
        svm_1=CmapSvmEvaluationResults(
            training_cmap_clouds=svm_1_training_cmap_clouds,
            summary=summary_svm_1_df,
            cloud_ref_and_tag_to_prediction_result=svm_1_cloud_ref_and_tag_to_prediction_result,
            svm=svm_1
        ),
        svm_2=CmapSvmEvaluationResults(
            training_cmap_clouds=svm_2_training_cmap_clouds,
            summary=summary_svm_2_df,
            cloud_ref_and_tag_to_prediction_result=svm_2_cloud_ref_and_tag_to_prediction_result,
            svm=svm_2
        )
    )
