from configuration import config
from latent_space_arithmetic import LatentSpaceArithmetic
from printer import PrintingHandler
from post_processing.distance_calculator import *

import os
import logging


class ExtendedTimes:
    """
    This class can calculate the distance of samples from their perturbation, given by their tumors
    """
    def __init__(self, samples_to_test_df, samples_to_test_info_df):
        """
        Initialize - save the samples to test among the different tests.
        :param samples_to_test_df: DataFrame with data in real space
        :param samples_to_test_info_df: information of the samples to test
        """
        self.participating_tumors = list(set(samples_to_test_info_df.tumor.unique()) &
                                         set(config.config_map['run_tissues_whitelist']))
        self.output_folder_name = 'ExtendedTimes'
        self.output_folder = None
        self.samples_to_test_info_df = samples_to_test_info_df[
            samples_to_test_info_df.tumor.isin(self.participating_tumors)]
        self.samples_to_test_df = samples_to_test_df.loc[self.samples_to_test_info_df.index]

    def run(self, test_name, data, model):
        """
        Run extended times tests - Calculate distance from given samples to perturbation vector.
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        logging.info("Starting perturbation tests: %s", str(test_name))
        output_folder = os.path.join(config.config_map['output_folder'], self.output_folder_name)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # Calculate all the clouds
        latent_space_arithmetic = LatentSpaceArithmetic(data, model)
        data_df, info_df, reference_points = data.get_all_data_and_info()
        encoded_data_df = model.predict_latent_space(data_df)
        encoded_test_data_df = model.predict_latent_space(self.samples_to_test_df)
        output_csv_path = os.path.join(output_folder, 'DMSO_distances.csv')
        calculate_dmso_distances(encoded_data_df, info_df, reference_points, encoded_test_data_df,
                                 self.samples_to_test_info_df, latent_space_arithmetic, self.participating_tumors,
                                 output_csv_path)

        all_calculated_encoded_data_df, all_calculated_decoded_data_df, all_calculated_info_df = \
            calculate_and_test_all_clouds(encoded_data_df, info_df, encoded_test_data_df, self.samples_to_test_info_df,
                                          latent_space_arithmetic, output_folder, True)

        # Print figures
        printer = PrintingHandler(data, model)
        all_encoded_data_df = all_calculated_encoded_data_df.append([encoded_data_df, encoded_test_data_df], sort=False)
        state = "Latent space"
        print_figures(self.samples_to_test_info_df, printer, all_encoded_data_df, info_df, all_calculated_info_df,
                      state, output_folder)
        all_decoded_data_df = all_calculated_decoded_data_df.append([data_df, self.samples_to_test_df], sort=False)
        state = "Real space"
        print_figures(self.samples_to_test_info_df, printer, all_decoded_data_df, info_df, all_calculated_info_df,
                      state, output_folder)
