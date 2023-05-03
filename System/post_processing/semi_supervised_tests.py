from post_processing.distance_calculator import print_figures, compare_perturbation_distances
from latent_space_arithmetic import LatentSpaceArithmetic
from configuration import config
import logging
import os


class SemiSupervisedTests:
    def __init__(self):
        self.output_folder_name = 'SemiSupervised'
        self.output_folder = None

    def run(self, test_name, data, model):
        """
        Run semi supervised tests - Calculate distance from calculated samples to left out treated samples.
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        logging.info("Starting semi supervised tests: %s", str(test_name))
        output_folder = os.path.join(config.config_map['output_folder'], self.output_folder_name)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # Calculate all the clouds
        latent_space_arithmetic = LatentSpaceArithmetic(data, model)
        data_df, info_df, reference_points = data.get_all_data_and_info()
        encoded_data_df = model.predict_latent_space(data_df)
        additional_info_to_test_df = info_df[~info_df.perturbation.isin(config.config_map['untreated_labels'])]
        samples_to_test_info_df = data.info_not_to_fit_df.append(additional_info_to_test_df, sort=False)
        samples_to_test_data_df = data.data_not_to_fit_df.append(data_df.loc[additional_info_to_test_df.index],
                                                                 sort=False)
        encoded_test_data_df = model.predict_latent_space(samples_to_test_data_df)

        # Calculate and print distance for each unlearn cloud from perturbation vector
        all_calculated_encoded_data_df, all_calculated_decoded_data_df, all_calculated_info_df = \
            compare_perturbation_distances(encoded_data_df, info_df, reference_points, encoded_test_data_df,
                                           samples_to_test_info_df, latent_space_arithmetic, output_folder)
