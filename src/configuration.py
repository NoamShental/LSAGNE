import logging
import os
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from torch.nn.init import xavier_uniform


class RunMode(Enum):
    CLI = 'Run program using the CLI, not the server.'
    PREFECT_SERVER = 'Run on prefect server.'


def _get_logging_level_from_env() -> int:
    logger_level_str = os.getenv('LSAGNE_LOGGER_LEVEL', 'INFO').upper()
    logger_level = logging.getLevelName(logger_level_str)
    if isinstance(logger_level, str):
        raise AssertionError(f'Undefined logger level "{logger_level_str}".')
    return logger_level


def _get_debug_mode_from_env() -> bool:
    debug_mode_str = os.getenv('LSAGNE_DEBUG_MODE', 'false').lower()
    return debug_mode_str in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']


@dataclass
class Configuration:
    root_folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

    root_organized_data_folder = os.path.join(root_folder_path, 'organized_data')

    organized_folder_name = 'cmap'

    organized_data_augmentation_folder_name = 'data_augmentation'

    epsilon: float = 1e-5

    organized_cmap_folder = os.path.join(root_organized_data_folder, organized_folder_name)

    organized_data_augmentation_folder = os.path.join(root_organized_data_folder, organized_data_augmentation_folder_name)

    raw_data_folder = os.path.join(root_folder_path, 'raw_data_folder')

    @property
    def raw_cmap_folder(self):
        return os.path.join(self.raw_data_folder, self.organized_folder_name)

    @property
    def raw_data_augmentation_folder(self):
        return os.path.join(self.raw_data_folder, self.organized_data_augmentation_folder_name)

    output_folder_path = os.path.join(root_folder_path, 'output')

    use_perturbations_whitelist = True

    use_tissues_whitelist = True

    np_numeric_precision_type = np.float64
    torch_numeric_precision_type = torch.float64

    time_24h_perturbation = "time 24h"

    dmso_6h_perturbation = "DMSO"

    reduced_clouds = False

    # Five basic drugs set
    # basic_perturbations = ["geldanamycin"]  # , "tamoxifen"]
    # basic_perturbations = ["geldanamycin", "raloxifene", "trichostatin-a", "wortmannin", "vorinostat"]  # , "tamoxifen"]
    if reduced_clouds:
        basic_perturbations = ["geldanamycin", "trichostatin-a", "wortmannin", time_24h_perturbation]
    else:
        # basic_perturbations = ["geldanamycin", "raloxifene", "trichostatin-a", "wortmannin", "vorinostat", "isonicotinohydroxamic-acid", "sirolimus", "estriol", time_24h_perturbation]  # , "tamoxifen"]
        basic_perturbations = ["geldanamycin", "raloxifene", "trichostatin-a", "wortmannin", "vorinostat",
                               "isonicotinohydroxamic-acid", "sirolimus", time_24h_perturbation]  # , "tamoxifen","tamoxifen"]
    # basic_perturbations = ["geldanamycin", "tamoxifen", "trichostatin-a", "wortmannin", "vorinostat"]

    # Three out-of-samples.
    new_perturbations = ["isonicotinohydroxamic-acid", "sirolimus", "estriol"]

    # perturbations_whitelist = basic_perturbations + new_perturbations
    perturbations_whitelist = basic_perturbations

    if reduced_clouds:
        tissues_whitelist = [
            "PC3 prostate adenocarcinoma",
            "A375 skin malignant melanoma",
            "HT29 large intestine colorectal adenocarcinoma",
            "VCAP prostate carcinoma",
        ]

    else:
        tissues_whitelist = [
            "HCC515 lung carcinoma",
            "HEPG2 liver hepatocellular carcinoma",
            "HA1E kidney normal kidney",
            "VCAP prostate carcinoma",
            "PC3 prostate adenocarcinoma",
            "MCF7 breast adenocarcinoma",
            "A375 skin malignant melanoma",
            "A549 lung non small cell lung cancer| carcinoma",
            "HT29 large intestine colorectal adenocarcinoma"
        ]

    tissue_code_to_name = {tissue.split(' ')[0]: tissue for tissue in tissues_whitelist}

    tissue_name_to_code = {name: code for code, name in tissue_code_to_name.items()}

    perturbation_times = [24]

    min_samples_per_cloud = 30

    min_treat_conc = 2

    cmap_organizer_untreated_labels_times = [6, 24]

    untreated_times = [0, 6]

    use_cuda = True

    flow_name = "basic_runs"

    untreated_labels = [dmso_6h_perturbation]

    dmso_label = untreated_labels[0]

    data_file_name = 'data.h5'

    information_file_name = 'info.csv'

    unique_clouds_file_name = 'unique_clouds.csv'

    slurm_unique_clouds_file_name = 'unique_clouds_1.csv'

    version = '0.10.0'

    description = "DMSO+24h trimming rate 0.5. \n" \
                  "Treated trimming rate 0.1. \n" \
                  "Using large radius loss. \n" \
                  "Using tissue classifier. \n" \
                  f"Working on reduced clouds = {reduced_clouds}. \n" \
                  "Reducing the VAE inner layers #3 \n" \
                  "Triangle enabled \n" \
                  "Added distance from predicted + distance from time 24h \n" \
                  # "Added L2 regularization on VAE layers" \
                  # "Removing skip connection from VAE decoder \n"

    run_mode = RunMode.CLI

    left_tissue_code = 'A375'

    left_perturbation_name = 'geldanamycin'

    log_stdout = True

    # If true the predicted class will be <all DMSO> + <treatment vector>
    # Otherwise it will be <#left_out points closest to the DMSO center> + <treatment vector>
    dudi_basic_include_all_dmso_in_predicted = False

    layer_weight_initialization_method = xavier_uniform

    debug_mode = _get_debug_mode_from_env()

    logger_level = _get_logging_level_from_env()


config = Configuration()
