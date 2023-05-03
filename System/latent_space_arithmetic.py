from configuration import config
import pandas as pd
import numpy as np
import logging


class LatentSpaceArithmetic:
    """
    Test arithmetic for custom cloud, on latent space and real space
    """

    def __init__(self, data, model):
        """
        Initialize, set all parameters
        :param data: Data handler.
        :param model: Model handler.
        """
        self.data = data
        self.model = model

    def get_vectors(self, perturbation, reference_points):
        """
        Get needed vectors to calculate new cloud, based on the cloud's perturbation
        :param perturbation: perturbation of calculated cloud.
        :param reference_points: tuple of all of the reference points
        :return: tuple of 2 numpy array: (perturbation_vector, perturbation_and_time_vector).
        """
        same_pert_index = self.data.info_df[self.data.info_df['perturbation'] == perturbation].iloc[0].name
        treated_ref_points, start_control_ref_points, _, pert_control_ref_points = reference_points[1:5]
        values = [treated_ref_points.loc[same_pert_index].values,
                  pert_control_ref_points.loc[same_pert_index].values,
                  start_control_ref_points.loc[same_pert_index].values]
        indexes = [0, 1, 2]
        columns = treated_ref_points.columns
        cloud_reference_points_df = pd.DataFrame(values, index=indexes, columns=columns)
        latent_space_reference_points_df = self.model.predict_latent_space(cloud_reference_points_df)
        latent_treated = latent_space_reference_points_df.iloc[0]
        latent_pert_time_control = latent_space_reference_points_df.iloc[1]
        latent_start_time_control = latent_space_reference_points_df.iloc[2]
        perturbation_vector = latent_treated - latent_pert_time_control
        perturbation_and_time_vector = latent_treated - latent_start_time_control
        return perturbation_vector.values, perturbation_and_time_vector.values

    @staticmethod
    def calculate_nearest_points_on_2_lines(l1, l2):
        """
        calculate nearest points on 2 lines, based on the article :
        The Minimum Distance Between Two Lines in n-Space - Michael Bard, Denny Himel
        :param l1: first line, containing 2 points, in the format of (p1+t*d)
        :param l2: second line, it is a tuple of 2 points
        :return: tuple of 2 points p1 and p2, while p1 is on l1, and p2 on l2, and the distance between them is the
                 minimum distance between l1 and l2.
        """
        # L1 in the format of : z1 = x0 + xt
        # L2 in the format of : z2 = y0 + ys

        # Unpack the lines
        x0, x = l1
        y0, y = l2

        # Set parameters
        A = np.dot(x, x)
        B = 2 * (np.dot(x0, x) - np.dot(x, y0))
        C = 2 * np.dot(x, y)
        D = 2 * (np.dot(y, y0) - np.dot(y, x0))
        E = np.dot(y, y)

        # Calculate s and t
        s = (2*A*D + B*C) / (C ** 2 - 4 * A * E)
        t = (C * s - B) / (2 * A)

        # Calculate nearest points
        x_nearest = x0 + x * t
        y_nearest = y0 + y * s
        return x_nearest, y_nearest

    def get_drug_vectors_from_base_points(self, drug, reference_points, control_6h_np, control_24h_np):
        """
        Get drug vectors based on base points
        :param drug: drug to get it's vectors
        :param reference_points: reference points
        :param control_6h_np: 1 point of control 6h, from which treatment vector start
        :param control_24h_np: 1 point of control 24h, from which drug vector start
        :return: tuple of drug vector and treatment vector
        """
        perturbation_vector_np, perturbation_and_time_vector_np = self.get_vectors(drug, reference_points)

        # From here we can calculate easily the calculated_vector - the vector we have to add to samples in start
        # time control set, to get the calculated samples:
        l1 = (np.float64(control_6h_np), np.float64(perturbation_and_time_vector_np))
        l2 = (np.float64(control_24h_np), np.float64(perturbation_vector_np))

        # Calculate nearest points on the 2 lines
        p1, p2 = self.calculate_nearest_points_on_2_lines(l1, l2)
        mean_of_calculated_cloud = (p1 + p2) / 2
        treatment_vector = mean_of_calculated_cloud - control_6h_np
        drug_vector = mean_of_calculated_cloud - control_24h_np
        return drug_vector, treatment_vector

    def calculate_perturbations_effect_for_given_data(self, base_samples_df, control_info_df,
                                                      perturbations, reference_points):
        """
        Calculate the effect of perturbation(s) on base samples, given the control data and info, and the reference
        points.
        NOTE: all data samples have to be in latent space
        :param base_samples_df: DataFrame with base samples to use as start point
        :param control_info_df: DataFrame with info of control
        :param perturbations: list of perturbations to apply
        :param reference_points: Reference points to find perturbations vectors
        :return: DataFrame with the base samples after the effect of the perturbations
        """
        start_time_control_info_df = control_info_df[
            control_info_df['pert_time'].isin(config.config_map['untreated_times'])]
        pert_time_control_info_df = control_info_df[
            ~control_info_df['pert_time'].isin(config.config_map['untreated_times'])]

        start_time_sample_ref = reference_points[5].loc[start_time_control_info_df.index].iloc[0]
        pert_time_sample_ref = reference_points[5].loc[pert_time_control_info_df.index].iloc[0]
        ref_points_df = pd.DataFrame([start_time_sample_ref, pert_time_sample_ref])
        encoded_control_ref_df = self.model.predict_latent_space(ref_points_df)
        encoded_start_time_cloud_center_np = encoded_control_ref_df.iloc[0].values
        encoded_pert_time_cloud_center_np = encoded_control_ref_df.iloc[1].values

        if not isinstance(perturbations, list) and not isinstance(perturbations, tuple):
            perturbations = [perturbations]

        # calculate perturbation vector for each perturbation
        calculated_vectors = []
        i = 0
        for p in perturbations:
            drug_vector, treatment_vector = self.get_drug_vectors_from_base_points(p, reference_points,
                                                                                   encoded_start_time_cloud_center_np,
                                                                                   encoded_pert_time_cloud_center_np)
            if i == 0:
                calculated_vectors.append(treatment_vector)
            else:
                calculated_vectors.append(drug_vector)
            i += 1

        # Apply all the calculated vectors we gather
        calculated_samples_df = base_samples_df
        for cv in calculated_vectors:
            calculated_samples_df = calculated_samples_df + cv

        # Create copy of the data in real space
        decoded_calculated_df = self.model.predict_decoder(calculated_samples_df)
        decoded_calculated_df.columns = self.data.data_df.columns
        return calculated_samples_df, decoded_calculated_df

    def calculate_perturbations_effect(self, base_samples_df, tumor, perturbations,
                                       in_latent_space=False):
        """
        Calculate the effect of perturbation(s) on base samples, given the control data and info, and the reference
        points
        :param base_samples_df: DataFrame with samples to apply the perturbations on them
        :param tumor: base tumor that the perturbations will apply on (to calculate the perturbations strength)
        :param perturbations: list of perturbations to apply
        :param in_latent_space: Boolean, Set to True if the given base samples is already in latent space
        :return: DataFrame with the base samples after the effect of the perturbations
        """
        data_df, info_df, reference_points = self.data.get_all_data_and_info()
        control_data_df, control_info_df = \
            self.data.get_data_and_info_by_tumor_and_perturbation(data_df, info_df, tumor,
                                                                  config.config_map['untreated_labels'])
        if not in_latent_space:
            base_samples_df = self.model.predict_latent_space(base_samples_df)
        return self.calculate_perturbations_effect_for_given_data(base_samples_df, control_info_df,
                                                                  perturbations, reference_points)

    def calculate_perturbations_effect_and_info(self, encoded_data_and_info, tumor, perturbations, pert_time):
        """
        Calculate new cloud of treated samples, based on the control and reference points, scaled by treated points.
        :param encoded_data_and_info: DataFrame with treated samples in latent space.
        :param tumor: untreated tumor to calculate the perturbations effect
        :param perturbations: perturbation, or list of perturbations, to apply on the tumor
        :param pert_time: list, perturbation times to calculate, it request DMSO in that time
        :return: tuple of: calculated samples in latent space, in real space, their info DataFrame, and their reference
                           indexes
        """
        # Get start time and perturbation time control samples.
        encoded_data_df, encoded_info_df = encoded_data_and_info

        control_data_df, control_info_df = \
            self.data.get_data_and_info_by_tumor_and_perturbation(encoded_data_df, encoded_info_df, tumor,
                                                                  config.config_map['untreated_labels'])

        start_time_control_info_df = control_info_df[
            control_info_df['pert_time'].isin(config.config_map['untreated_times'])]

        start_time_control_data_df = control_data_df.loc[start_time_control_info_df.index]

        encoded_calculated_df, decoded_calculated_df = \
            self.calculate_perturbations_effect(start_time_control_data_df,
                                                tumor, perturbations, in_latent_space=True)

        start_time_control_info_df = control_info_df[
            control_info_df['pert_time'].isin(config.config_map['untreated_times'])]
        calculated_info_df = start_time_control_info_df.copy()

        # In any case, just use the info of first perturbation
        perturbation = perturbations[0]

        # Create info DataFrame and create different index for the calculated
        calculated_info_df.index = 'calculated_{0}_{1}_'.format(tumor, perturbation) + \
                                   calculated_info_df['original_index']
        encoded_calculated_df.index = calculated_info_df.index
        decoded_calculated_df.index = calculated_info_df.index

        calculated_info_df['arithmetic'] = 'calculated'
        calculated_info_df['perturbation'] = perturbation
        treated_sample = encoded_info_df[(encoded_info_df.perturbation == perturbation) &
                                         (encoded_info_df.tumor == tumor)].iloc[0]

        calculated_info_df['one_hot_labels'] = [treated_sample.one_hot_labels]*calculated_info_df.shape[0]
        calculated_info_df['numeric_labels'] = treated_sample.numeric_labels
        calculated_info_df['pert_time'] = pert_time[0]

        # Create reference points for calculated data
        reference_points = []

        # Get reference points for perturbation
        treated_reference_sample, start_time_control_reference_sample, pert_time_control_reference_sample = \
            self.data.p_dictionary[perturbation]

        # Get reference points for tumor
        start_time_control_sample, pert_time_control_sample = self.data.untreated_dictionary[tumor]
        for i in range(len(self.data.reference_points)):
            reference_points.append(self.data.create_empty_df_from_other(decoded_calculated_df, np.float64))

        samples_count = calculated_info_df.shape[0]
        reference_points[0][:] = np.tile(start_time_control_sample.values, (samples_count, 1))
        reference_points[1][:] = np.tile(treated_reference_sample.values, (samples_count, 1))
        reference_points[2][:] = np.tile(start_time_control_reference_sample.values, (samples_count, 1))
        reference_points[3][:] = np.tile(pert_time_control_sample.values, (samples_count, 1))
        reference_points[4][:] = np.tile(pert_time_control_reference_sample.values, (samples_count, 1))

        return encoded_calculated_df, decoded_calculated_df, calculated_info_df, reference_points
