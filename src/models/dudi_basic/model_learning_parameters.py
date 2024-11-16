from dataclasses import dataclass
from typing import List, Dict, Optional, Type, Union

from torch import nn

from src.cmap_cloud_ref import CmapCloudRef
from src.datasets_and_organizers.raw_cmap_dataset import RawCmapDataset
from src.models.dudi_basic.data_loaders.multi_device_fast_data_loader import MultiDeviceFastDataLoader
from src.models.dudi_basic.lsagne_model import LsagneModel
from src.parameters_parser import ParameterParser
from src.technical_learning_parameters import TechnicalLearningParameters


def _extract_fields_names(dataclass_obj)-> List[str]:
    return [k for k,v in dataclass_obj.__class__.__dict__['__dataclass_fields__'].items()]

def _assign_relevant_fields_to_self(obj_self, *dataclass_objs):
    obj_self_fields_names = _extract_fields_names(obj_self)
    for dataclass_obj in dataclass_objs:
        for field_name in _extract_fields_names(dataclass_obj):
            if field_name in obj_self_fields_names:
                setattr(obj_self, field_name, getattr(dataclass_obj, field_name))


@dataclass
class AugmentationParams:
    alias: str
    augmentation_path: str
    augmentation_rate: float
    # what is the prob to choose these params over the others
    prob: float
    clouds_to_augment: list[CmapCloudRef]



@dataclass
class InputLearningParameters:
    left_out_cloud: CmapCloudRef
    n_epochs: int
    embedding_dim: int
    input_dim: int
    batch_size: int
    lr: float
    filter_n_epochs: int
    filter_classifier_inner_dims: List[int]
    vae_encode_inner_dims: List[int]
    vae_decode_inner_dims: List[int]
    mtadam_loss_weights: List[int]
    clouds_classifier_inner_layers: List[int]
    tissues_classifier_inner_layers: List[int]
    enable_vae_skip_connection: bool
    use_mtadam: bool
    use_amsgrad: bool
    use_scheduler: bool
    scheduler_factor: float
    max_radius : float
    scheduler_patience: int
    scheduler_cooldown: int
    mock: bool
    loss_coef: Dict[str, float]
    filter_epoch: int
    drawing_interval: int
    contrastive_margin: float
    use_seed: bool
    use_cuda: bool
    random_seed: Optional[int]
    warmup_reference_points_duration: int
    warmup_reference_points_loss_coef: Dict[str, float]
    reselect_reference_points_epochs: List[int]
    distances_evaluation_interval: int
    evaluate_classifier_accuracy_interval: int
    evaluate_svm_accuracy_interval: int
    perturbations_whitelist: List[str]
    tissues_whitelist: List[str]
    clouds_trimming_epochs: List[int]
    trim_treated_clouds_ratio_to_keep: float
    trim_untreated_clouds_and_time_24h_ratio_to_keep: float
    enable_triangle: bool
    default_linear_layer: Type[nn.Linear]
    initial_max_dmso_cloud_size: Optional[int]
    add_class_weights_to_classifiers: bool
    data_loader_type: Type[MultiDeviceFastDataLoader]
    cross_validation_clouds: List[CmapCloudRef]
    michael_theme: bool
    treatment_and_drug_vectors_distance_loss_cdist_usage: bool
    different_directions_loss_power_factor: float
    use_untrained_clouds_predictions_in_training: bool
    predicted_cloud_max_size: int
    augmentation_params: list[AugmentationParams]
    treatment_vectors_magnitude_regulator_relu_coef: float
    drug_vectors_magnitude_regulator_relu_coef: float
    perturbations_equivalence_sets: List[List[str]]
    perturbations_equivalence_losses_coefs: Dict[str, float]
    augment_during_warmup: bool
    partial_cloud_training: Dict[CmapCloudRef, int]
    number_of_batches: int
    number_of_samples_per_cloud: int

    @staticmethod
    def _create_cloud_refs(
            cmap_dataset: RawCmapDataset,
            tissue_code: str,
            perturbation: str,
            all_literal: str = "all"
    ) -> List[CmapCloudRef]:
        if tissue_code == all_literal and perturbation == all_literal:
            cloud_refs = cmap_dataset.unique_cloud_refs
        elif tissue_code == all_literal:
            cloud_refs = [cloud_ref for cloud_ref in cmap_dataset.unique_cloud_refs
                          if cloud_ref.perturbation == perturbation]
        elif perturbation == all_literal:
            cloud_refs = [cloud_ref for cloud_ref in cmap_dataset.unique_cloud_refs
                          if cloud_ref.tissue_code == tissue_code]
        else:
            return [CmapCloudRef(tissue_code, perturbation)]
        return [cloud_ref for cloud_ref in cloud_refs if cloud_ref.is_not_dmso_6h_or_24h]

    def _parse_cloud_refs(
            self,
            param_name: str,
            clouds_strs: List[str],
            cmap_dataset: RawCmapDataset
    ) -> List[CmapCloudRef]:
        clouds_refs = set()
        if len(clouds_strs) % 2 != 0:
            raise AssertionError(f'{param_name} len is odd.')

        for i in range(0, len(clouds_strs), 2):
            tissue_code = clouds_strs[i]
            perturbation = clouds_strs[i + 1]
            clouds_refs.update(self._create_cloud_refs(cmap_dataset, tissue_code, perturbation))
        return list(clouds_refs)

    def _parse_partial_clouds_requests(
            self,
            param_name: str,
            clouds_strs_with_size: List[Union[int, str]], # [[<tissue code>, <perturbation>, <size>]]
            cmap_dataset: RawCmapDataset
    ) -> Dict[CmapCloudRef, int]:
        cloud_ref_to_training_size: Dict[CmapCloudRef, int] = {}
        if len(clouds_strs_with_size) % 3 != 0:
            raise AssertionError(f'{param_name} size is not divided by 3.')
        clouds_to_exclude = {self.left_out_cloud, *self.cross_validation_clouds}
        for i in range(0, len(clouds_strs_with_size), 3):
            tissue_code = clouds_strs_with_size[i]
            perturbation = clouds_strs_with_size[i + 1]
            cloud_size = clouds_strs_with_size[i + 2]
            cloud_refs = self._create_cloud_refs(cmap_dataset, tissue_code, perturbation)
            for cloud_ref in cloud_refs:
                if cloud_ref in clouds_to_exclude:
                    continue
                if cloud_ref in cloud_ref_to_training_size:
                    existing_size = cloud_ref_to_training_size[cloud_ref]
                    if existing_size != cloud_size:
                        print(f'WARNING: {cloud_ref} size change {existing_size} -> {cloud_size}')
                cloud_ref_to_training_size[cloud_ref] = cloud_size

        for cloud_ref, training_size in cloud_ref_to_training_size.copy().items():
            cmap_cloud_size = len(cmap_dataset.cloud_ref_to_idx[cloud_ref])
            if cmap_cloud_size <= training_size:
                print(f"WARNING: {cloud_ref} desired partial training size is {training_size}, but cmap contains "
                      f"only {cmap_cloud_size} samples, this cloud will be trained as regular training.")
                cloud_ref_to_training_size.pop(cloud_ref)

        return cloud_ref_to_training_size

    def replace_parameters(self, **kwargs):
        kwargs = kwargs.copy()

        cmap_dataset = RawCmapDataset.load_dataset_from_disk(
            perturbations_whitelist=self.perturbations_whitelist,
            tissues_whitelist=self.tissues_whitelist
        )
        parameter_parser = ParameterParser(cmap_dataset)

        for param_name in list(kwargs.keys()):
            if param_name in [
                'data_loader_type',
                'default_linear_layer'
            ]:
                continue
            param_value = kwargs[param_name]
            if param_name in ['warmup_reference_points_loss_coef', 'loss_coef', 'perturbations_equivalence_losses_coefs']:
                loss_coefs = getattr(self, param_name)
                for loss_name, loss_value in param_value.items():
                    if loss_name not in loss_coefs:
                        raise AssertionError('Unrecognized loss...')
                    loss_coefs[loss_name] = loss_value
                param_value = loss_coefs
            if param_name in ['cross_validation_clouds', 'clouds_to_augment']:
                param_value = parameter_parser.parse_cloud_refs(param_name, param_value)
            if param_name == 'left_out_cloud':
                param_value = parameter_parser.parse_cloud_refs(param_name, param_value)[0]
            if param_name == 'partial_cloud_training':
                param_value = parameter_parser.parse_partial_clouds_requests(param_name, param_value, {
                    self.left_out_cloud, *self.cross_validation_clouds
                })
            if param_name == 'augmentation_params':
                param_value = [
                    AugmentationParams(
                        alias=param['alias'],
                        augmentation_path=param['augmentation_path'],
                        augmentation_rate=param['augmentation_rate'],
                        prob=param['prob'],
                        clouds_to_augment=parameter_parser.parse_cloud_refs(param_name, param['clouds_to_augment']),
                    )
                    for param in param_value
                ]
            setattr(self, param_name, param_value)
            del kwargs[param_name]

        if len(kwargs) > 0:
            raise AssertionError('There is a missing parser for one of the keys...')

        # if a cloud is used for cross validation, it can't be used for partial training
        for partial_training_cloud_ref in list(self.partial_cloud_training.keys()):
            if partial_training_cloud_ref in {*self.cross_validation_clouds, self.left_out_cloud}:
                print(f'Warning! {partial_training_cloud_ref} is used both in CV/left out and partial training, '
                      f'this cloud will be removed from partial training.')
                del self.partial_cloud_training[partial_training_cloud_ref]


@dataclass
class ModelLearningParameters(TechnicalLearningParameters, InputLearningParameters):
    model: Optional[LsagneModel]
    raw_cmap_dataset: RawCmapDataset

    def __init__(self,
                 model_training_parameters: TechnicalLearningParameters,
                 basic_model_input_parameters: InputLearningParameters,
                 raw_cmap_dataset: RawCmapDataset,
                 model: Optional[LsagneModel] = None
                 ):
        _assign_relevant_fields_to_self(self, model_training_parameters, basic_model_input_parameters)
        self.model = model
        self.raw_cmap_dataset = raw_cmap_dataset
