from keras.layers.advanced_activations import PReLU
import os

mock = False
configuration_dict = {
    'version': 'v3.6.7',
    'test_number': '',
    'data_file_name': 'data.h5',
    'information_file_name': 'info.csv',
    'test_set_percent': 0,

    'leave_data_out': True,
    'untreated_labels': ['DMSO', 'GTEX'],
    'untreated_times': [0, 6],
    'perturbation_times': [24],
    'test_pert_times': [48],
    'minimum_samples_to_classify': 30,
    'max_samples_per_cloud': 500,
    'genes_filtering_is_needed': False,
    'healthy_tissues_data_is_needed': True,

    # Randomize the data of leaveout/leavein (treated: only tested treated, labels: all the labels)
    'random_treated': False,
    'random_labels': False,

    # Model parameters
    'maximum_weight': 10000,
    'should_use_class_weights': True,
    'activation': PReLU,
    'initialization_method': 'glorot_uniform',

    # IO parameters (results paths and other IO parameters)
    'root_output_folder': os.path.join('/', 'home', 'bayehez2', 'results', 'remote_debug'),
    'print_model': True,
    'save_calculated_cloud': False,
    'print_2d_tessellation': False,
    'print_clouds_color_by_arithmetics': False,
    'print_clouds_custom_drugs': ["raloxifene", "tamoxifen", "DMSO"],#["raloxifene", "DMSO"],#["geldanamycin", "DMSO"],
    'print_clouds_custom_tumors': ["HEPG2 liver hepatocellular carcinoma"],#["A375 skin malignant melanoma"],#["A549 lung non small cell lung cancer| carcinoma"],
    'print_custom_correctly_classified_drugs': ["tamoxifen"],
    'print_custom_correctly_classified_tumors': ["HEPG2 liver hepatocellular carcinoma"],
    'print_dose': False,

    # Trajectories test configuration
    # If true, calculate trajectories from DMSO24, else from DMSO6
    'trajectories_startpoint_is_24': True,

    # If true, calculate trajectories to predicted, else use real treated
    'trajectories_endpoint_is_predicted': True,

    # Number of points on trajectories vector
    'trajectories_points_on_vector': 2,

    # Number of samples to choose per cloud (in not random trajectories this will be override by 1)
    'trajectories_samples_to_choose': 10,

    # Drug combinations configuration
    'drug_combination_max_drugs': 3,
    'drug_combination_samples_per_cloud': 0,
    'drug_combination_cell_lines': [],# Set to [] for all
    'drug_combinations_calculate_cmap': False,
    'drug_combinations_calculate_gtex': True,
    'drug_combinations_unscale_before_convert_to_12k': False,

    # What data to create
    'drug_combination_run_combined_prediction': True, # If set - Calculate combinations
    'drug_combination_sample_sphere_by_drug_vector': False, # If set - Calculate spheres

    # What to do?
    'drug_combination_save_genes': True, # If set - save the genes
    'drug_combination_predict_death': False, # If set - save predictions
    'drug_combination_num_of_spheres_samples': 10,
    'drug_combination_sample_std_tolerance': 0.8,

    # Encode decode samples file path
    'encode_decode_samples_path': None,

    # Death Predictor
    'death_predictor_svm_file': 'death_predictor.sav',
    'death_predictor_go_cycles_files': 'GoCellCycles.txt',
    'death_predictor_annotation_file': 'DMSO_annotation.txt',
}

if mock:
    # Data handler
    configuration_dict['organized_data_folder'] = os.path.join('/', 'home', 'bayehez2', 'groupData', 'organize_data', 'bashan', 'mock')
    configuration_dict['run_use_perturbations_whitelist'] = False
    configuration_dict['run_use_tissues_whitelist'] = False
    configuration_dict['run_dont_fit_non_whitelisted_perturbations'] = False
    configuration_dict['run_non_whitelisted_tissues_fit_on_control'] = False

    configuration_dict['leave_out_tissue_name'] = '1'
    configuration_dict['leave_out_perturbation_name'] = '1'

    # Model parameters
    configuration_dict['batch_size'] = 50
    configuration_dict['latent_dim'] = 2
    configuration_dict['encoder_dim'] = [20, 20, 20]
    configuration_dict['decoder_dim'] = [20, 20, 20]
    configuration_dict['classifier_intermediate_dim'] = []
    configuration_dict['bias'] = True
    configuration_dict['parallel_vectors_power_factor'] = 10
    configuration_dict['other_perts_power_factor'] = 20

    # Learning parameters
    configuration_dict['epochs'] = 400
    configuration_dict['early_stopping_patience'] = 100
    configuration_dict['epochs_of_filtering'] = []
    configuration_dict['reference_points_change'] = [2, 10, 50, 150]
    configuration_dict['warmup_epochs'] = 50
    configuration_dict['should_update_factors'] = False
    configuration_dict['update_factors_delay'] = 1

    configuration_dict['learning_rate'] = 0.001
    configuration_dict['decay'] = 0
    configuration_dict['vae_sampling_std'] = 0.01

    # Loss parameters
    configuration_dict['vae_loss_factor'] = 3
    configuration_dict['log_xy_loss_factor'] = 2
    configuration_dict['KL_loss_factor'] = 0
    configuration_dict['classifier_loss_factor'] = 5
    configuration_dict['coliniarity_pert_and_time_loss_factor'] = 5
    configuration_dict['coliniarity_pert_loss_factor'] = 5
    configuration_dict['parallel_vectors_loss_factor'] = 1
    configuration_dict['distance_between_vectors_loss_factor'] = 1
    configuration_dict['distance_from_reference_loss_factor'] = 0.3
    configuration_dict['collinearity_other_perts_loss_factor'] = 0

    configuration_dict['updated_vae_loss_factor'] = 0
    configuration_dict['updated_log_xy_loss_factor'] = 0
    configuration_dict['updated_KL_loss_factor'] = 0
    configuration_dict['updated_classifier_loss_factor'] = 0
    configuration_dict['updated_coliniarity_pert_and_time_loss_factor'] = 0
    configuration_dict['updated_coliniarity_pert_loss_factor'] = 0
    configuration_dict['updated_parallel_vectors_loss_factor'] = 0
    configuration_dict['updated_distance_between_vectors_loss_factor'] = 0
    configuration_dict['updated_distance_from_reference_loss_factor'] = 0
    configuration_dict['updated_collinearity_other_perts_loss_factor'] = 0

    configuration_dict['root_output_folder'] = os.path.join('/', 'home', 'bayehez2', 'results', 'remote_debug')
    configuration_dict['print_dose'] = False
else:
    # Cmap
    configuration_dict['raw_data_folder'] = os.path.join('/', 'home', 'bayehez2', 'groupData', 'CmapAllData')
    configuration_dict['organized_data_folder'] = os.path.join('/', 'home', 'bayehez2', 'groupData', 'organized_data', 'bashan', 'remote_debug')
    configuration_dict['unique_clouds_file_name'] = 'unique_clouds.csv'

    # Organizer fields
    configuration_dict['organizer_use_perturbations_whitelist'] = True
    configuration_dict['organizer_perturbations_whitelist'] = ["geldanamycin", "raloxifene", "vorinostat",
                                                               "trichostatin-a", "wortmannin", "sirolimus",
                                                               "isonicotinohydroxamic-acid", "estriol", "tamoxifen"]
    configuration_dict['organizer_use_tissues_whitelist'] = False
    configuration_dict['organizer_tissues_whitelist'] = ["MCF7 breast adenocarcinoma",
                                                         "PC3 prostate adenocarcinoma",
                                                         "A375 skin malignant melanoma",
                                                         "A549 lung non small cell lung cancer| carcinoma",
                                                         "HT29 large intestine colorectal adenocarcinoma",
                                                         "HA1E kidney normal kidney",
                                                         "HCC515 lung carcinoma",
                                                         "HEPG2 liver hepatocellular carcinoma",
                                                         "VCAP prostate carcinoma"]
    configuration_dict['organizer_min_cell_lines_per_perturbation'] = 3
    configuration_dict['organizer_min_samples_per_cell_line'] = 30

    # Data Handler
    # Perturbations
    configuration_dict['run_use_perturbations_whitelist'] = True
    configuration_dict['run_perturbations_whitelist'] = ["geldanamycin", "raloxifene", "vorinostat", "trichostatin-a",
                                                         "wortmannin"]
    configuration_dict['run_dont_fit_non_whitelisted_perturbations'] = True

    # Tumors
    configuration_dict['run_use_tissues_whitelist'] = False
    configuration_dict['run_tissues_whitelist'] = ["MCF7 breast adenocarcinoma",
                                                   "PC3 prostate adenocarcinoma",
                                                   "A375 skin malignant melanoma",
                                                   "A549 lung non small cell lung cancer| carcinoma",
                                                   "HT29 large intestine colorectal adenocarcinoma",
                                                   "HA1E kidney normal kidney",
                                                   "HCC515 lung carcinoma", "HEPG2 liver hepatocellular carcinoma",
                                                   "VCAP prostate carcinoma"]
    configuration_dict['run_non_whitelisted_tissues_fit_on_control'] = True

    # CMAP test set
    configuration_dict['test_data_file_name'] = 'diff_pert_time_data.h5'
    configuration_dict['test_info_file_name'] = 'diff_pert_time_info.csv'

    # CMAP live dead file name
    configuration_dict['live_dead_file_name'] = 'live dead_rate.csv'

    # 12K matrix
    configuration_dict['12K_matrix'] = 'cmap_to_12K.csv'

    # GTex
    configuration_dict['gtex_data_folder'] = os.path.join('/', 'home', 'bayehez2', 'groupData', 'CmapAllData')
    configuration_dict['min_samples_per_gtex_class'] = 100

    # Leave out
    configuration_dict['leave_out_tissue_name'] = 'A375 skin malignant melanoma'
    configuration_dict['leave_out_perturbation_name'] = 'geldanamycin'

    # Model parameters
    configuration_dict['batch_size'] = 50
    configuration_dict['latent_dim'] = 20
    configuration_dict['encoder_dim'] = []
    configuration_dict['decoder_dim'] = []
    configuration_dict['classifier_intermediate_dim'] = []
    configuration_dict['bias'] = True
    configuration_dict['parallel_vectors_power_factor'] = 10
    configuration_dict['other_perts_power_factor'] = 40

    # Learning parameters
    configuration_dict['epochs'] = 3000
    configuration_dict['early_stopping_patience'] = 150
    configuration_dict['epochs_of_filtering'] = [400]
    configuration_dict['reference_points_change'] = [2, 10, 40, 100, 400]
    configuration_dict['warmup_epochs'] = 500
    configuration_dict['should_update_factors'] = False
    configuration_dict['update_factors_delay'] = 1

    configuration_dict['learning_rate'] = 0.001
    configuration_dict['decay'] = 0
    configuration_dict['vae_sampling_std'] = 1

    # Loss parameters
    configuration_dict['vae_loss_factor'] = 10
    configuration_dict['log_xy_loss_factor'] = 3
    configuration_dict['KL_loss_factor'] = 6
    configuration_dict['classifier_loss_factor'] = 1
    configuration_dict['coliniarity_pert_and_time_loss_factor'] = 0.5
    configuration_dict['coliniarity_pert_loss_factor'] = 2
    configuration_dict['parallel_vectors_loss_factor'] = 0.2
    configuration_dict['distance_between_vectors_loss_factor'] = 0.5
    configuration_dict['distance_from_reference_loss_factor'] = 1
    configuration_dict['collinearity_other_perts_loss_factor'] = 0.01

    configuration_dict['updated_vae_loss_factor'] = 0
    configuration_dict['updated_log_xy_loss_factor'] = 0
    configuration_dict['updated_KL_loss_factor'] = 0
    configuration_dict['updated_classifier_loss_factor'] = 0
    configuration_dict['updated_coliniarity_pert_and_time_loss_factor'] = 0
    configuration_dict['updated_coliniarity_pert_loss_factor'] = 0
    configuration_dict['updated_parallel_vectors_loss_factor'] = 0
    configuration_dict['updated_distance_between_vectors_loss_factor'] = 0
    configuration_dict['updated_distance_from_reference_loss_factor'] = 0
    configuration_dict['updated_collinearity_other_perts_loss_factor'] = 0


class ConfigurationMetaClass(type):
    """
    This class is the metaclass for configuration, and assures that each time this file will be imported,
    the same instance of configuration will be received.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Called upon creating new instance of it's successors
        :param args:
        :param kwargs:
        :return: same instance of it's successors
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(ConfigurationMetaClass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConfigurationHandler(metaclass=ConfigurationMetaClass):
    """
    Hyper parameters handling: validation & defaults.
    """

    def __init__(self, configuration):
        """
        Initialize the configuration handler.
        :param configuration: thyper parameters dictionary.
        """
        self.config_map = configuration
        self._hyper_parameters_validation()

    def _hyper_parameters_validation(self):
        """
        hyper parameters validation.
        """
        if self.config_map['leave_data_out']:
            assert self.config_map['leave_out_tissue_name'] is not None, \
                "Missing leave_out_data config param: tissue name..."
            assert self.config_map['leave_out_perturbation_name'] is not None, \
                "Missing leave_out_data config param: perturbation name..."


config = ConfigurationHandler(configuration_dict)
