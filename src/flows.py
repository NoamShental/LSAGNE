from typing import Optional, Dict, Any, Tuple

from prefect import Flow, Parameter
from torch import nn

from src.cmap_cloud_ref import CmapCloudRef
from src.configuration import config
from src.models.dudi_basic.model_learning_parameters import InputLearningParameters, AugmentationParams
from src.models.dudi_basic.data_loaders.all_clouds_multi_device_fast_data_loader import \
    AllCloudsMultiDeviceFastDataLoader
from src.models.dudi_basic.data_loaders.full_multi_device_fast_data_loader import FullMultiDeviceFastDataLoader
from src.models.dudi_basic.data_loaders.triangle_multi_device_fast_data_loader import TriangleMultiDeviceFastDataLoader
from src.models.dudi_basic.maxnorm_constrained_linear import MaxNormConstrainedLinear
from src.models.dudi_basic.post_training_evaluators.apply_svm_on_left_out_task import ApplySvmOnLeftOutTask
from src.models.dudi_basic.post_training_evaluators.apply_svm_on_trained_clouds_task import ApplySvmOnTrainedCloudsTask
from src.models.dudi_basic.post_training_evaluators.calculate_predicted_clouds_task import CalculatePredictedCloudsTask
from src.models.dudi_basic.post_training_evaluators.calculate_anchor_treatment_and_drug_vectors_task import CalculateAnchorTreatmentAndDrugVectorsTask
from src.models.dudi_basic.post_training_evaluators.calculate_reference_treatment_and_drug_vectors_task import \
    CalculateReferenceTreatmentAndDrugVectorsTask
from src.models.dudi_basic.post_training_evaluators.create_data_reduction_methods_task import \
    CreateDataReductionMethodsTask
from src.models.dudi_basic.post_training_evaluators.create_data_tensors_for_evaluation_task import CreateDataTensorsForEvaluationTask
from src.models.dudi_basic.post_training_evaluators.create_evaluation_params_task import CreateEvaluationParamsTask
from src.models.dudi_basic.post_training_evaluators.evaluate_vectors_angles_task import EvaluateVectorsAnglesTask
from src.models.dudi_basic.post_training_evaluators.plot_losses_task import PlotLossesTask
from src.models.dudi_basic.post_training_evaluators.post_training_evaluation_task import PostTrainingEvaluationTask
from src.models.dudi_basic.post_training_evaluators.post_training_latent_space_drawer_task import PostTrainingLatentSpaceDrawerTask
from src.models.dudi_basic.post_training_evaluators.reference_points_encoder_task import ReferencePointsEncoderTask
from src.tasks.load_raw_cmap_task import LoadRawCmapTask
from src.tasks.train_models_tasks.train_basic_model_task import TrainBasicModelTask
from src.tasks.utilities_tasks.initialize_logger_file_handler_task import InitializeLoggerFileHandlerTask


def create_basic_model_flow(flow_name, use_cuda, add_logger=True, override_parameters: Optional[Dict[str, Any]] = None) -> Tuple[Flow, InputLearningParameters]:
    clouds_classifier = 5000.0
    tissues_classifier = 5000.0
    distance_from_cloud_center = 10000
    max_radius_limiter = 10000
    kld = 0.01
    collinearity = 10000
    different_directions = 10000
    treatment_and_drug_vectors_distance = 10000
    n_epochs = 5000
    warmup_reference_points_duration = 300
    treatment_and_drug_vectors_distance = 10000
    reference_point_reselect_period = 4101
    embedding_dim = 20
    params = InputLearningParameters(
        left_out_cloud=CmapCloudRef('HT29', 'geldanamycin'),
        use_cuda=use_cuda,
        n_epochs=n_epochs,
        random_seed=None, # None
        use_seed=True, # False
        embedding_dim=embedding_dim,
        input_dim=977,
        batch_size=320, # 320
        lr=0.0001,
        filter_n_epochs=300,
        filter_classifier_inner_dims=[30,40],
        vae_encode_inner_dims=[],
        vae_decode_inner_dims=[],
        enable_vae_skip_connection=True,
        mtadam_loss_weights=[0.5, 1.7, 1, 2, 1, 0.8],
        clouds_classifier_inner_layers=[],
        tissues_classifier_inner_layers=[],
        use_mtadam=False,
        use_amsgrad=False,
        use_scheduler=True,
        scheduler_factor=0.3,
        scheduler_patience=500,
        scheduler_cooldown=0,
        mock=False,
        max_radius=0.05,
        loss_coef={
            'vae_mse': 1000.0,
            'vae_kld': kld,
            'vae_l1_regularization': 0.0, # 0.01
            'clouds_classifier': clouds_classifier,
            'tissues_classifier': tissues_classifier, # 3
            'online_contrastive': 0.0,
            'treatment_vectors_collinearity_using_batch_treated': collinearity,
            'treatment_vectors_collinearity_using_batch_control': collinearity,
            'drug_vectors_collinearity_using_batch_treated': collinearity,
            'drug_vectors_collinearity_using_batch_control': collinearity,
            'treatment_vectors_different_directions_using_anchors': 0.0, # 2.5
            'drug_and_treatment_vectors_different_directions_using_anchors': 0.0, # 2.5
            'treatment_vectors_different_directions_using_batch': different_directions,  # 2.5
            'drug_and_treatment_vectors_different_directions_using_batch': different_directions,  # 2.5
            'distance_from_treatment_vector_predicted': 0.0, #10.0
            'distance_from_drug_vector_predicted': 0.0, #10.0
            # 'distance_from_cloud_center': 2.5 * 30,
            'distance_from_cloud_center_6h': distance_from_cloud_center, #2.5
            'distance_from_cloud_center_dmso_24h': distance_from_cloud_center, #2.5
            'distance_from_cloud_center_24h_without_dmso_24h': distance_from_cloud_center, #2.5
            'max_radius_limiter': max_radius_limiter,
            'treatment_and_drug_vectors_distance_p1_p2_loss': treatment_and_drug_vectors_distance,
            'treatment_and_drug_vectors_distance_p1_p2_to_treated_loss': treatment_and_drug_vectors_distance,
            'treatment_vectors_magnitude_regulator': 10.0,
            'drug_vectors_magnitude_regulator': 10.0,
            'std_1': 0.0
        },
        filter_epoch=3000,
        drawing_interval=200,
        contrastive_margin=2.0,
        warmup_reference_points_duration=warmup_reference_points_duration,
        warmup_reference_points_loss_coef={
            'vae_mse': 1.0,
            'vae_kld': kld,
            'vae_l1_regularization': 0.0, # 0.01
            'clouds_classifier': clouds_classifier,
            'tissues_classifier': tissues_classifier, #5
            'online_contrastive': 0.0,
            'treatment_vectors_collinearity_using_batch_treated': 0.0,
            'treatment_vectors_collinearity_using_batch_control': 0.0,
            'drug_vectors_collinearity_using_batch_treated': 0.0,
            'drug_vectors_collinearity_using_batch_control': 0.0,
            'treatment_vectors_different_directions_using_anchors': 0.0,
            'drug_and_treatment_vectors_different_directions_using_anchors': 0.0,
            'treatment_vectors_different_directions_using_batch': 0.0,
            'drug_and_treatment_vectors_different_directions_using_batch': 0.0,
            'distance_from_treatment_vector_predicted': 0.0,
            'distance_from_drug_vector_predicted': 0.0,
            'distance_from_cloud_center_6h': 0.0,
            'distance_from_cloud_center_dmso_24h': 0.0,
            'distance_from_cloud_center_24h_without_dmso_24h': 0.0,
            'max_radius_limiter': 0.0,
            'treatment_and_drug_vectors_distance_p1_p2_loss': 0.0,
            'treatment_and_drug_vectors_distance_p1_p2_to_treated_loss': 0.0,
            'treatment_vectors_magnitude_regulator': 0.0,
            'drug_vectors_magnitude_regulator': 0.0,
            'std_1': 0.0
        },
        # reselect_reference_points_epochs=[i for i in range(warmup_reference_points_duration, n_epochs, 300)],
        reselect_reference_points_epochs = [] if (reference_point_reselect_period < 1 ) else [i for i in range(warmup_reference_points_duration, n_epochs, reference_point_reselect_period)],
        distances_evaluation_interval=100,
        evaluate_classifier_accuracy_interval=10,
        evaluate_svm_accuracy_interval=10,
        perturbations_whitelist=config.perturbations_whitelist,
        tissues_whitelist=config.tissues_whitelist,
        # clouds_trimming_epochs=[150, 300, 450, 600, 750],
        # clouds_trimming_epochs=[501, 1001, 1501, 2001, 2501, 3001],
        # clouds_trimming_epochs=[300, 600, 900, 1200, 1500],
        # clouds_trimming_epochs=[],
        clouds_trimming_epochs=[1000],
        trim_treated_clouds_ratio_to_keep=0.85,
        trim_untreated_clouds_and_time_24h_ratio_to_keep=0.5,
        enable_triangle=True,
        # default_linear_layer=MaxNormConstrainedLinear
        default_linear_layer=nn.Linear,
        initial_max_dmso_cloud_size=None,
        add_class_weights_to_classifiers=False,
        # data_loader_type=FullMultiDeviceFastDataLoader
        # data_loader_type=TriangleMultiDeviceFastDataLoader
        data_loader_type=AllCloudsMultiDeviceFastDataLoader,
        cross_validation_clouds=[
            # FIXME cross validation cannot be empty
            CmapCloudRef('A375', 'raloxifene')
        ],
        michael_theme=True,
        treatment_and_drug_vectors_distance_loss_cdist_usage=False,
        different_directions_loss_power_factor=10,       # needs to be even
        use_untrained_clouds_predictions_in_training=True,
        predicted_cloud_max_size=600,
        augmentation_params=[
            # AugmentationParams(
            #     alias='basic',
            #     augmentation_path='basic',
            #     augmentation_rate=0.7,
            #     prob=0.2,
            #     clouds_to_augment=[
            #         # CmapCloudRef.from_input('A375', 'geldanamycin')
            #     ],
            # )
        ],
        treatment_vectors_magnitude_regulator_relu_coef=1.0,
        drug_vectors_magnitude_regulator_relu_coef=1.0,
        perturbations_equivalence_sets=[
            # ["raloxifene", "isonicotinohydroxamic-acid"]
        ],
        # for all equivalence set coefs:
        # 1.0 --> 100% equivalence for everything in the set
        # 0.0 --> 0%   equivalence for everything in the set (same as no equivalence)
        perturbations_equivalence_losses_coefs={
            'clouds_classifier': 0.5,
            'treatment_vectors_different_directions_using_anchors': 0.5,
            'treatment_vectors_different_directions_using_batch': 0.5,
        },
        augment_during_warmup=False,
        partial_cloud_training={
            # CmapCloudRef.from_input('A375', 'geldanamycin'): 20
        },
        number_of_batches=25,
        number_of_samples_per_cloud=10
    )

    if override_parameters:
        params.replace_parameters(**override_parameters)

    # TODO add parameters assertions

    with Flow(flow_name) as flow:
        # define flow_parameters
        run_name = Parameter('run_name', 'RUN_NAME')
        # The parameter is not connected to any task, so we need to manually add it.
        flow.add_task(run_name)
        flow_parameters = [run_name]
        # create tasks
        original_cmap = LoadRawCmapTask(name='load raw CMAP')(
            params=params
        )
        training_summary = TrainBasicModelTask(
            name='train basic model',
            flow_parameters=flow_parameters)(
                cmap=original_cmap,
                params=params
            )
        PostTrainingEvaluationTask(
            name='create evaluation parameters',
            flow_parameters=flow_parameters)(
            params=params,
            training_summary=training_summary
            )
        # training_evaluator_params = CreateEvaluationParamsTask(
        #     name='create evaluation parameters',
        #     flow_parameters=flow_parameters)(
        #     original_full_cmap_dataset=original_cmap,
        #     params=params,
        #     training_summary=training_summary
        #     )
        # PlotLossesTask(
        #     name='plot losses',
        #     flow_parameters=flow_parameters)(
        #     training_evaluator_params=training_evaluator_params
        # )
        # encoded_reference_points = ReferencePointsEncoderTask(
        #     name='encode reference points',
        #     flow_parameters=flow_parameters)(
        #     evaluator_params=training_evaluator_params
        # )
        # training_and_left_out_tensors = CreateDataTensorsForEvaluationTask(
        #     name='calculate tensors',
        #     flow_parameters=flow_parameters)(
        #     evaluator_params=training_evaluator_params
        # )
        # perturbation_to_anchor_treatment_vector, perturbation_to_anchor_drug_vector = CalculateAnchorTreatmentAndDrugVectorsTask(
        #     name='calculate treatment vectors',
        #     flow_parameters=flow_parameters)(
        #     evaluator_params=training_evaluator_params,
        #     encoded_reference_points=encoded_reference_points
        # )
        # perturbation_and_tissue_to_reference_treatment_vector, perturbation_and_tissue_to_reference_drug_vector = CalculateReferenceTreatmentAndDrugVectorsTask(
        #     name='calculate treatment vectors',
        #     flow_parameters=flow_parameters)(
        #     evaluator_params=training_evaluator_params,
        #     encoded_reference_points=encoded_reference_points,
        #     left_out_tensors=training_and_left_out_tensors
        # )
        # cloud_ref_to_predicted_cloud = CalculatePredictedCloudsTask(
        #     name='calculate predicted left out',
        #     flow_parameters=flow_parameters)(
        #     training_evaluator_params=training_evaluator_params,
        #     perturbation_to_anchor_treatment_vector=perturbation_to_anchor_treatment_vector,
        #     perturbation_to_anchor_drug_vector=perturbation_to_anchor_drug_vector,
        #     perturbation_and_tissue_to_reference_treatment_vector=perturbation_and_tissue_to_reference_treatment_vector,
        #     perturbation_and_tissue_to_reference_drug_vector=perturbation_and_tissue_to_reference_drug_vector,
        #     training_tensors=training_and_left_out_tensors,
        #     encoded_reference_points=encoded_reference_points
        # )
        # svm_prediction_on_left_out = ApplySvmOnLeftOutTask(
        #     name='apply svm left out',
        #     flow_parameters=flow_parameters)(
        #     training_evaluator_params=training_evaluator_params,
        #     training_and_left_out_tensors=training_and_left_out_tensors,
        #     cloud_ref_to_predicted_cloud=cloud_ref_to_predicted_cloud
        # )
        # ApplySvmOnTrainedCloudsTask(
        #     name='apply svm trained clouds',
        #     flow_parameters=flow_parameters)(
        #     training_evaluator_params=training_evaluator_params,
        #     training_and_left_out_tensors=training_and_left_out_tensors,
        #     cloud_ref_to_predicted_cloud=cloud_ref_to_predicted_cloud,
        #     svm_prediction_on_left_out=svm_prediction_on_left_out
        # )
        # umap, tsne, pca, LDA = CreateDataReductionMethodsTask(
        #     name='create data reduction methods',
        #     flow_parameters=flow_parameters)(
        #     training_summary=training_summary
        # )
        # # reduction_tools = [tsne, umap, pca]
        # reduction_tools = [pca, LDA, tsne]
        # for reduction_tool in reduction_tools:
        #     # for include_training_predicted in [True]:
        #     for include_training_predicted in [True, False]:
        #         PostTrainingLatentSpaceDrawerTask(
        #             name='latent space drawer',
        #             flow_parameters=flow_parameters)(
        #             training_evaluator_params=training_evaluator_params,
        #             training_and_left_out_tensors=training_and_left_out_tensors,
        #             cloud_ref_to_predicted_cloud=cloud_ref_to_predicted_cloud,
        #             svm_prediction_on_left_out=svm_prediction_on_left_out,
        #             data_reduction_tool=reduction_tool,
        #             include_training_predicted=include_training_predicted
        #         )
        # EvaluateVectorsAnglesTask(
        #     name='evaluate vectors angles',
        #     flow_parameters=flow_parameters)(
        #     training_evaluator_params=training_evaluator_params,
        #     perturbation_to_anchor_treatment_vector=perturbation_to_anchor_treatment_vector,
        #     perturbation_to_anchor_drug_vector=perturbation_to_anchor_drug_vector,
        #     training_and_left_out_tensors=training_and_left_out_tensors,
        #     cloud_ref_to_predicted_cloud=cloud_ref_to_predicted_cloud,
        #     svm_prediction_on_left_out=svm_prediction_on_left_out,
        #     encoded_reference_points=encoded_reference_points
        # )

        if add_logger:
            InitializeLoggerFileHandlerTask.connect_to_flow(flow, [original_cmap], flow_parameters, params.michael_theme)

    if params.michael_theme:
        print("""
  __  __ _      _                _    __  __           _          ___        _ _            
 |  \\/  (_) ___| |__   __ _  ___| |  |  \\/  | ___   __| | ___    / _ \\ _ __ | (_)_ __   ___ 
 | |\\/| | |/ __| '_ \\ / _` |/ _ \\ |  | |\\/| |/ _ \\ / _` |/ _ \\  | | | | '_ \\| | | '_ \\ / _ \\
 | |  | | | (__| | | | (_| |  __/ |  | |  | | (_) | (_| |  __/  | |_| | | | | | | | | |  __/
 |_|  |_|_|\\___|_| |_|\\__,_|\\___|_|  |_|  |_|\\___/ \\__,_|\\___|   \\___/|_| |_|_|_|_| |_|\\___|                                                                                   
        """)

    return flow, params
