from data_handler import DataHandler
from model_handler import ModelHandler
from post_processing.perturbations_tests import PerturbationsTests
from post_processing.tcga_tests import TCGATests
from post_processing.extended_times_tests import ExtendedTimes
from post_processing.semi_supervised_tests import SemiSupervisedTests
from post_processing.confusion_table import ConfusionTable
from post_processing.statistics_tests import StatisticsTests
from post_processing.trajectories_test import TrajectoriesTests
from post_processing.drug_combinations import DrugCombinations
from post_processing.encode_decode import EncodeDecode
from helper_functions import set_logger, set_output_folders
from configuration import config
import pandas as pd

import keras.backend as K
import os
import logging


class PostTests:
    """
    Class to handle all the post processing tests
    """
    def __init__(self, test_name, do_perturbations_tests,
                 do_tcga_tests, do_extended_times_tests, do_semi_supervised_tests, do_confusion_table,
                 do_statistics_tests, do_traj_test, do_random_traj_test, do_drug_combination, encode_samples,
                 decode_samples):
        """
        Initialize the class with results root folder, and set the tests to run
        :param test_name: full name of test
        :param do_perturbations_tests: if set, do perturbations test
        :param do_tcga_tests: if set, do tcga tests
        :param do_extended_times_tests: if set, do extended times tests
        :param do_semi_supervised_tests: if set, do semi supervised tests
        :param do_confusion_table: if set, create confusion table
        :param do_statistics_tests: if not 0, run statistics tests
        :param do_traj_test: if set, run trajectories tests
        :param do_random_traj_test: if set, run random trajectories tests
        :param do_drug_combination: if set, run drug combinations test
        :param encode_samples: if set, create encode samples df
        :param decode_samples: if set, create decode samples df
        """
        self.test_name = test_name
        K.clear_session()
        set_output_folders(test_name, delete_old_folder=False)
        set_logger()

        # Create list of post tests
        self.post_tests = []
        if do_perturbations_tests:
            self.post_tests.append(PerturbationsTests())
        if do_tcga_tests:
            self.post_tests.append(TCGATests())
        if do_extended_times_tests:
            samples_to_test_distance_df = \
                pd.read_hdf(os.path.join(config.config_map['organized_data_folder'],
                                         config.config_map['test_data_file_name']), 'df')
            samples_to_test_distance_info_df = \
                pd.read_csv(os.path.join(config.config_map['organized_data_folder'],
                                         config.config_map['test_info_file_name']))
            samples_to_test_distance_info_df.set_index('inst_id', inplace=True, drop=True)
            self.post_tests.append(ExtendedTimes(samples_to_test_distance_df, samples_to_test_distance_info_df))
        if do_semi_supervised_tests:
            self.post_tests.append(SemiSupervisedTests())
        if do_confusion_table:
            self.post_tests.append(ConfusionTable())
        if do_statistics_tests != 0:
            self.post_tests.append(StatisticsTests(do_statistics_tests))
        if do_traj_test:
            self.post_tests.append(TrajectoriesTests(False))
        if do_random_traj_test:
            self.post_tests.append(TrajectoriesTests(True))
        if do_drug_combination:
            self.post_tests.append(DrugCombinations())
        if encode_samples:
            self.post_tests.append(EncodeDecode(do_decode=False))
        if decode_samples:
            self.post_tests.append(EncodeDecode(do_decode=True))

    def run(self):
        """
        Run all post tests
        """
        data = DataHandler()
        logging.info("Post tests for {}".format(self.test_name))
        weights_path = os.path.join(config.config_map['output_folder'], 'Models', 'model.h5')
        model = ModelHandler(data.number_of_drugs, starting_weights_path=weights_path)

        data.pick_reference_point_function = lambda samples_df: \
            DataHandler.find_reference_point_by_center_of_cloud(model.predict_latent_space, samples_df)
        data.load_reference_points(reference_points_path=os.path.join(config.config_map['output_folder'],
                                                                      'reference_points.p'))
        for post_test in self.post_tests:
            post_test.run(self.test_name, data, model)
