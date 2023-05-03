import os
import shutil
import logging
from configuration import config


def set_logger(use_file_logger=True, use_stdout_logger=True):
    """
    Set the logger for the test
    :param use_file_logger: if set, log to file
    :param use_stdout_logger: if set, log to stdout
    """
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers = []

    if use_file_logger:
        test_file_handler = logging.FileHandler(os.path.join(config.config_map['output_folder'], "log.txt"))
        test_file_handler.setFormatter(log_formatter)
        test_file_handler.setLevel(logging.INFO)
        root_logger.addHandler(test_file_handler)

    if use_stdout_logger:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)


def set_output_folders(test_name, delete_old_folder=False):
    """
    Set all output folders, and create main output folder
    :param test_name: name of test
    :param delete_old_folder: if set and old folder exists in that path, delete old folder and recreate that folder
    """
    # Set configurations keys per test
    config.config_map['output_folder'] = os.path.join(config.config_map['root_output_folder'], str(test_name))
    config.config_map['data_folder'] = os.path.join(config.config_map['output_folder'], 'Data')
    config.config_map['models_folder'] = os.path.join(config.config_map['output_folder'], 'Models')
    config.config_map['pictures_folder'] = os.path.join(config.config_map['output_folder'], 'Pictures')
    config.config_map['tests_folder'] = os.path.join(config.config_map['output_folder'], 'Tests')
    config.config_map['arithmetic_output_folder'] = os.path.join(config.config_map['output_folder'], 'CalculatedCloud')

    # Delete old output folder
    if delete_old_folder:
        while os.path.isdir(config.config_map['output_folder']):
            shutil.rmtree(config.config_map['output_folder'])

    # Create the new output folder
    if not os.path.isdir(config.config_map['output_folder']):
        os.makedirs(config.config_map['output_folder'])
