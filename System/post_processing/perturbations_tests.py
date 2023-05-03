from configuration import config
from sklearn.metrics.pairwise import cosine_similarity
from latent_space_arithmetic import LatentSpaceArithmetic
import numpy as np
import pandas as pd
import logging
import os


class PerturbationsTests:
    """
    This class handles all the tests on drugs
    """

    @staticmethod
    def get_angle_between_perturbation(p1, p2, data, model):
        """
        Get the medium angle in perturbation
        :param p1: name of first perturbation
        :param p2: name of second perturbation
        :param data: data handler to read the data from
        :param model: model handler to move data to latent space
        :return: float, biggest angle
        """
        data_df, info_df, reference_points = data.get_all_data_and_info()
        # Get samples of perts
        p1_samples_df = data_df.loc[info_df[info_df.perturbation == p1].index]
        p2_samples_df = data_df.loc[info_df[info_df.perturbation == p2].index]

        # Get control pert time samples
        control_p1_pert_time_df = reference_points[3].loc[p1_samples_df.index]
        control_p2_pert_time_df = reference_points[3].loc[p2_samples_df.index]

        # Encode the data to latent space
        encoded_p1_samples_df = model.predict_latent_space(p1_samples_df)
        encoded_p2_samples_df = model.predict_latent_space(p2_samples_df)
        encoded_control_p1_pert_time_df = model.predict_latent_space(control_p1_pert_time_df)
        encoded_control_p2_pert_time_df = model.predict_latent_space(control_p2_pert_time_df)

        # Calculate vectors
        p1_vectors_df = encoded_p1_samples_df - encoded_control_p1_pert_time_df
        p2_vectors_df = encoded_p2_samples_df - encoded_control_p2_pert_time_df

        # Cosine similarity
        angles_np = cosine_similarity(p1_vectors_df, p2_vectors_df)
        return angles_np.mean()

    @staticmethod
    def perturbations_angle(latent_space_arithmetic):
        """
        Calculate the angle between each pair of perturbations vectors, and print it
        :param latent_space_arithmetic: class to calculate the pert vectors
        :return: DataFrame of angles between perturbations
        """
        # Create list of perturbations
        data = latent_space_arithmetic.data
        perturbation_list = list(data.info_df.perturbation.unique())
        for p in config.config_map['untreated_labels']:
            perturbation_list.remove(p)

        p_angles_matrix_df = pd.DataFrame(index=perturbation_list, columns=perturbation_list)
        for p1 in perturbation_list:
            for p2 in perturbation_list:
                p_angles_matrix_df.loc[p1][p2] =\
                    PerturbationsTests.get_angle_between_perturbation(p1, p2, data, latent_space_arithmetic.model)
        return np.degrees(np.arccos(p_angles_matrix_df.astype(np.float64)))

    @staticmethod
    def run(test_name, data, model):
        """
        Run perturbations test - angles between each pair of perturbations
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        logging.info("Starting perturbation tests: %s", str(test_name))
        latent_space_arithmetic = LatentSpaceArithmetic(data, model)
        output_folder = os.path.join(config.config_map['output_folder'], 'perturbations_tests')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # Calculate and save angles between perturbations
        angles_between_perturbations_df = PerturbationsTests.perturbations_angle(latent_space_arithmetic)
        angles_between_perturbations_df.to_hdf(os.path.join(output_folder, 'angles_between_perturbations.h5'), 'df')
