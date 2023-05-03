from configuration import config

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu
from statistics_calculator import StatisticsCalculator
from tessellation_tester import TessellationTester
import logging
import os


class Tester:
    def __init__(self, model, data, printer):
        """
        Initialize the Tester class
        :param model: the model to check it's accuracy
        :param data: data handler object
        :param printer: printer class to print all the figures
        """
        self.model = model
        self.data = data
        self.printer = printer

        self.test_folder_path = config.config_map['tests_folder']
        if not os.path.isdir(self.test_folder_path):
            os.makedirs(self.test_folder_path)
        self.test_result_path = os.path.join(self.test_folder_path, 'test_results.txt')
        self.update_data()

    def update_data(self):
        """
        Update all internal data according to current state
        """
        # Create DataFrame with all the data, copied.
        if config.config_map['leave_data_out']:
            self.data_df = self.data.data_df.append(self.data.left_data_df)
            self.info_df = self.data.info_df.append(self.data.left_info_df)
            self.reference_points = [self.data.reference_points[i].append(self.data.left_reference_points[i])
                                        for i in range(len(self.data.reference_points))]
        else:
            self.data_df = self.data.data_df
            self.info_df = self.data.info_df
            self.reference_points = [self.data.reference_points[i] for i in range(len(self.data.reference_points))]

    def save_test_result(self, result_line, logging_level=logging.INFO):
        """
        Save result of test to output folder and log
        :param result_line: line of result to save
        :param logging_level: level of result logging, default to INFO
        """
        logging.log(logging_level, result_line)
        with open(self.test_result_path, 'at') as f:
            f.write(result_line + '\n')

    def test_classifier_accuracy(self, data_df, info_df, text_label):
        """
        Predict the classifier accuracy
        :param data_df: DataFrame of samples to test
        :param info_df: DataFrame of the info of that samples
        :param text_label: label of data type to print
        """
        labels_predicted = np.argmax(
            self.model.predict_classifier(data_df).values,
            axis=-1)
        real_labels = info_df['numeric_labels']
        score = accuracy_score(real_labels, labels_predicted)
        score = np.around(score, decimals=4) * 100

        self.save_test_result(text_label + ' classifier accuracy: ' + str(score))
        return score

    def test_coliniarity(self, data_df, text_label):
        """
        Show the average coliniarity after predicting the given data
        :param data_df: data to predict and calculate it's coliniarity
        :param text_label: labels of the data to test
        """
        reference_points = self.data.get_reference_points_to_data_slice(data_df.index, self.reference_points)
        loss = self.model.predict_coliniarity_loss(data_df, reference_points)
        loss = np.around(np.average(loss.values), decimals=5)
        self.save_test_result(text_label + ' coliniarity loss:' + str(loss))
        return loss

    def test_vae(self, data_df, text_label):
        """
        Show the average vae loss on all the data
        :param data_df: Data frame to test
        :param text_label: labels of the data to test
        """
        predicted_data, z_mean_df, z_log_var_df = self.model.predict_variational_loss(data_df)
        original_data = data_df.values
        z_mean_np = z_mean_df.values
        z_log_var_np = z_log_var_df.values

        xent_loss = np.average(mean_squared_error(original_data, predicted_data))
        xent_loss = np.around(np.average(xent_loss), decimals=5)
        kl_loss = np.mean(-5e-4 * np.mean(1. + z_log_var_np - np.square(z_mean_np) - np.exp(z_log_var_np), axis=-1))
        kl_loss = np.around(np.average(kl_loss), decimals=5)

        self.save_test_result(text_label + ' Distance loss:' + str(xent_loss))
        self.save_test_result(text_label + ' KL loss:' + str(kl_loss))
        loss = np.mean(
            config.config_map['log_xy_loss_factor'] * xent_loss + config.config_map['KL_loss_factor'] * kl_loss)
        loss = np.around(np.average(loss), decimals=5)

        self.save_test_result(text_label + ' VAE loss after factor:' + str(loss))
        return xent_loss, kl_loss, loss

    def test_distance_between_points(self, data_df, data_2_df, info_df, text_label, logging_level=logging.DEBUG):
        """
        Test the distance between each cloud for each of the clouds, in the requested data, between data_df and data_df_2
        :param data_df: Data frame with first data points
        :param data_2_df: Data frame with second data points
        :param info_df: DataFrame with the info of the data
        :param text_label: labels of the data to test
        :param logging_level: level of logging line
        """
        labels_df = info_df['numeric_labels']
        distances_df = pd.DataFrame(index=labels_df.unique(), columns=data_df.columns, dtype=np.float64)
        for label in labels_df.unique():
            cloud_indexes = data_2_df[data_2_df.index.isin(labels_df[labels_df.values == label].index)].index
            cloud_predicted_samples_df = data_2_df.loc[cloud_indexes]
            cloud_original_samples_df = data_df.loc[cloud_indexes]
            distances_df.loc[label] = mean_squared_error(cloud_predicted_samples_df, cloud_original_samples_df,
                                               multioutput='raw_values')
            loss = np.average(distances_df.loc[label].values)
            cloud_name = info_df[info_df.numeric_labels == label]['classifier_labels'].iloc[0]
            self.save_test_result('{0} cloud {1} distance: {2}'.format(text_label, cloud_name, str(loss)), logging_level)
        return distances_df

    def test_distance_between_xin_xout(self, data_df, info_df, text_label):
        """
        Test the distance between each cloud for each of the clouds, in the requested data
        :param data_df: DataFrame with all the data
        :param info_df: DataFrame with the info of the data
        :param text_label: labels of the data to test
        """
        predicted_data_df, _, _ = self.model.predict_variational_loss(data_df)
        return self.test_distance_between_points(data_df, predicted_data_df, info_df, text_label)

    def test_distance_as_percentage_of_std(self, data_df, info_df, text_label):
        """
        Calculate the distance divided by standard deviation of the input data
        :param data_df: input data to test
        :param info_df: DataFrame with the info of the data
        :param text_label: output label
        :return: DataFrame with all the calculated distances
        """
        labels_df = info_df['numeric_labels']

        # Calculate std for each label
        std_df = pd.DataFrame(index=labels_df.unique(), columns=data_df.columns, dtype=np.float64)
        for label in labels_df.unique():
            cloud_indexes = data_df[data_df.index.isin(labels_df[labels_df.values == label].index)].index
            cloud_samples_df = data_df.loc[cloud_indexes]
            std_df.loc[label] = np.std(cloud_samples_df, axis=0)

        # Calculate the distances as percentage of std
        distances_df = self.test_distance_between_xin_xout(data_df, info_df, text_label)
        distances_as_percentage_df = 100*(distances_df / std_df)

        # Save the distances table, and some metrics on them
        output_file_name = os.path.join(self.test_folder_path, text_label + ' percentage of std.csv')
        distances_as_percentage_df.to_csv(output_file_name)

        maximum_df = pd.DataFrame(index=data_df.columns, columns=['cloud', 'percentage'])
        maximum_df['cloud'] = distances_as_percentage_df.idxmax()
        maximum_df['percentage'] = distances_as_percentage_df.max()
        output_file_name = os.path.join(self.test_folder_path, text_label + ' maximum percentage of std.csv')
        maximum_df.to_csv(output_file_name)

        mean_df = pd.DataFrame(index=data_df.columns, columns=['percentage'])
        mean_df['percentage'] = distances_as_percentage_df.mean()
        output_file_name = os.path.join(self.test_folder_path, text_label + ' mean percentage of std.csv')
        mean_df.to_csv(output_file_name)

        median_df = pd.DataFrame(index=data_df.columns, columns=['percentage'])
        median_df['percentage'] = distances_as_percentage_df.median()
        output_file_name = os.path.join(self.test_folder_path, text_label + ' median percentage of std.csv')
        median_df.to_csv(output_file_name)

        self.save_test_result(
            '{0} mean distance in scale of std: {1:.3f}%'.format(text_label, mean_df['percentage'].mean()))
        self.save_test_result(
            '{0} median distance in scale of std: {1:.3f}%'.format(text_label, median_df['percentage'].median()))
        self.save_test_result(
            '{0} max distance in scale of std: {1:.3f}%'.format(text_label, maximum_df['percentage'].max()))

        return distances_as_percentage_df

    @staticmethod
    def filter_by_selector(dataframes, selector_df):
        """
        return rows of results_df, after masked by selector.
        :param dataframes: list of DataFrame
        :param selector_df: binary DataFrame, with same indexes as results_df
        :return: list of filtered dataframes
        """
        selector_df = selector_df.loc[dataframes[0].index]
        selector_df = selector_df[selector_df.calculate == 1]
        return [x.loc[selector_df.index] for x in dataframes]

    def test_dimensions_distribution(self, data_df, text_label):
        """
        Check the distribution and mean of each dimension in latent space
        :param data_df: data to test
        :param text_label: output label
        :return: mean of data, and mean of std in each dimension, in latent space
        """
        encoded_df = self.model.predict_latent_space(data_df)
        mean_data = encoded_df.values.mean()
        mean_data = np.around(mean_data, decimals=4)
        self.save_test_result(text_label + ' latent space mean:' + str(mean_data))
        std_by_dimensions = encoded_df.std(axis=0)
        std_overall = np.around(std_by_dimensions.mean(), decimals=4)
        self.save_test_result(text_label + ' latent space mean of std:' + str(std_overall))
        return mean_data, std_overall

    def do_sanity_tests(self, current_run_state):
        """
        Do all the tests
        :param current_run_state: current state of program, to add to the output result
        """
        self.update_data()
        is_test_set = config.config_map['test_set_percent'] > 0
        results = {}
        class_train_data_df, class_train_info_df = self.filter_by_selector([self.data.train_data_df,
                                                                            self.data.train_info_df],
                                                                           self.data.selectors[1])
        results['classifier train'] = self.test_classifier_accuracy(class_train_data_df, class_train_info_df, current_run_state + ' Train')
        if is_test_set:
            class_test_data_df, class_test_info_df = self.filter_by_selector([self.data.test_data_df,
                                                                             self.data.test_info_df],
                                                                             self.data.selectors[1])
            results['classifier test'] = self.test_classifier_accuracy(class_test_data_df, class_test_info_df, current_run_state + ' Test')

        vae_train_data_df = self.filter_by_selector([self.data.train_data_df], self.data.selectors[0])[0]
        results['vae distance train'], results['vae kl train'], results['vae global train'] = \
            self.test_vae(vae_train_data_df, current_run_state + ' Train')
        if is_test_set:
            vae_test_data_df = self.filter_by_selector([self.data.test_data_df], self.data.selectors[0])[0]
            results['vae distance test'], results['vae kl test'], results['vae global test'] = \
                self.test_vae(vae_test_data_df, current_run_state + ' Test')

        colin_train_data_df = self.filter_by_selector([self.data.train_data_df], self.data.selectors[2])[0]
        results['coliniarity train'] = self.test_coliniarity(colin_train_data_df, current_run_state + ' Train')
        if is_test_set:
            colin_test_data_df = self.filter_by_selector([self.data.test_data_df], self.data.selectors[2])[0]
            results['coliniarity test'] = self.test_coliniarity(colin_test_data_df, current_run_state + ' Test')
            colin_all_data_df = self.filter_by_selector([self.data.data_df], self.data.selectors[2])[0]
            self.test_coliniarity(colin_all_data_df, current_run_state + ' All')

        self.test_distance_as_percentage_of_std(self.data.data_df,
                                                self.data.info_df,
                                                current_run_state)
        vae_data_df = self.filter_by_selector([self.data.data_df], self.data.selectors[0])[0]
        results['latent space mean'], results['latent space std'] = \
            self.test_dimensions_distribution(vae_data_df, current_run_state)

        return results

    @staticmethod
    def _get_data_and_info_by_tumor_and_perturbation(data_df, info_df, tumor, perturbation):
        """
        Get slice of the data and the information, based on perturbation and tumor
        :param data_df: data df to slice from.
        :param info_df: info df of the data.
        :param tumor: tumor to slice by.
        :param perturbation: perturbation, or list of perturbations, to slice by
        :return: sliced data and info
        """
        if type(perturbation) is list:
            sliced_indexes = info_df[(info_df['tumor'] == tumor) &
                                     (info_df['perturbation'].isin(perturbation))].index
        else:
            sliced_indexes = info_df[(info_df['tumor'] == tumor) &
                                     (info_df['perturbation'] == perturbation)].index
        sliced_info_df = info_df.loc[sliced_indexes]
        sliced_data_df = data_df.loc[sliced_indexes]
        return sliced_data_df, sliced_info_df

    @staticmethod
    def test_mann_whitney_u(x_np, y_np):
        """
        calculate the Mann Whitney U and p-value, for given 2 1D arrays.
        :param x_np: first array sample.
        :param y_np: second array sample.
        :return: tuple of U and p-value.
        """
        _, mann_p = mannwhitneyu(x_np, y_np, alternative='two-sided')
        return mann_p

    @staticmethod
    def calculate_cosine_similiarity_matrix(first_vertexes_np, second_vertexes_np, reference_vertex):
        """
        Calculate the
        :param first_vertexes_np: numpy array of first cloud vertexes.
        :param second_vertexes_np: numpy array of first cloud vertexes.
        :param reference_vertex: vertex of angle to calculate, 1d array of features.
        :return: an array, with all the possible options angle - between the points in first_vertexes_np, to points in
                second_vertexes_np, based on points in vertexes_of_angle_np
        """
        first_vectors_matrix = first_vertexes_np - reference_vertex
        second_vectors_matrix = second_vertexes_np - reference_vertex
        return 1 - cosine_similarity(first_vectors_matrix, second_vectors_matrix)

    def calculate_mann_whitney_u_for_angles(self, x_first_np, x_second_np, x_vertex_of_angle, x_name,
                                            y_first_np, y_second_np, y_vertex_of_angle, y_name, text_label):
        """
        Calculate the Mann Whitney U and p-value, from calculating the angles between 2 sets of parameters.
        :param x_first_np: for x angles array, the first group of points
        :param x_second_np: for x angles array, the second group of points
        :param x_vertex_of_angle: for x angles array, the vertex of angle to calculate, 1d array of features.
        :param x_name: name of x angles array.
        :param y_first_np: for x angles array, the first group of points
        :param y_second_np: for y angles array, the second group of points
        :param y_vertex_of_angle: for y angles array, the vertex of angle to calculate, 1d array of features.
        :param y_name: name of y angles array
        :param text_label: label for figure.
        :return: tuple of U and p-value.
        """
        x_angles_matrix = Tester.calculate_cosine_similiarity_matrix(x_first_np, x_second_np, x_vertex_of_angle)
        x_angles_array = x_angles_matrix.ravel()
        y_angles_matrix = Tester.calculate_cosine_similiarity_matrix(y_first_np, y_second_np, y_vertex_of_angle)
        y_angles_array = y_angles_matrix.ravel()

        max_size = max(x_angles_array.shape[0], y_angles_array.shape[0], 10000)
        if x_angles_array.shape[0] > max_size:
            x_angles_array = np.random.choice(x_angles_array, size=max_size)
        if y_angles_array.shape[0] > max_size:
            y_angles_array = np.random.choice(y_angles_array, size=max_size)

        observations_dictionary = {x_name: x_angles_array, y_name: y_angles_array}
        self.printer.print_distribution(observations_dictionary, text_label,
                                        output_directory=os.path.join(config.config_map['pictures_folder'],
                                                                      'Angles'))

        return Tester.test_mann_whitney_u(x_angles_array, y_angles_array)

    def _angle_test_for_all_perturbations(self, reference_samples_df, encoded_data_and_info, start_time_control_center,
                                          pert_time_control_center, tumor,
                                          reference_perturbation, text_prefix):
        """
        For each perturbation that the tested tissue tests for, calculate the cosine similarity between that tissue
        and the reference point.
        :param reference_samples_df: Reference samples to calculate the cosing similarity.
        :param encoded_data_and_info: encoded data and it's info.
        :param start_time_control_center: Control samples in start time center.
        :param pert_time_control_center: Control samples in pert time center.
        :param tumor: tumor of reference samples.
        :param reference_perturbation: perturbation of reference samples.
        :param text_prefix: prefix of logging line.
        """
        encoded_data_df, encoded_info_df = encoded_data_and_info

        tumor_info_df = encoded_info_df[encoded_info_df['tumor'] == tumor]

        for perturbation in tumor_info_df['perturbation'].unique():

            # Skip Untreated perturbation, or cloud with same perturbation
            if perturbation in config.config_map['untreated_labels'] or perturbation == reference_perturbation:
                continue

            curr_pert_data_df, curr_pert_info_df = self._get_data_and_info_by_tumor_and_perturbation(
                encoded_data_df, tumor_info_df, tumor, perturbation)

            curr_pert_perturbation_and_time_vector = curr_pert_data_df - start_time_control_center
            curr_pert_perturbation_vector = curr_pert_data_df - pert_time_control_center
            reference_perturbation_and_time_vector = reference_samples_df - start_time_control_center
            reference_perturbation_vector = reference_samples_df - pert_time_control_center

            pert_and_time_cos_sim = cosine_similarity(curr_pert_perturbation_and_time_vector,
                                                      reference_perturbation_and_time_vector)
            pert_cos_sim = cosine_similarity(curr_pert_perturbation_vector,
                                                      reference_perturbation_vector)
            pert_and_time_loss = np.mean(1 - pert_and_time_cos_sim)
            pert_loss = np.mean(1 - pert_cos_sim)
            self.save_test_result('{0} {1}, start time control: {2}'.format(text_prefix, perturbation, str(pert_and_time_loss)))
            self.save_test_result('{0} {1}, pert time control: {2}'.format(text_prefix, perturbation, str(pert_loss)))

    def _angle_test(self, calculated_latent_space_df, encoded_data_and_info, tumor,
                    perturbation, state):
        """
        Check the latent-space arithmetic: calculated samples vs. other perturbation treated samples.
        :param calculated_latent_space_df: Calculated samples to test their angle, in latent space.
        :param encoded_data_and_info: all the data in latent space and it's info.
        :param tumor: tumor name of calculated data.
        :param perturbation: perturbation name of calculated data.
        :param state: state of running, for saving the figure.
        """
        # First, calculate the angle between calculated and real
        encoded_data_df, encoded_info_df = encoded_data_and_info

        # Extract control and treated data
        control_data_df, control_info_df = self._get_data_and_info_by_tumor_and_perturbation(
            encoded_data_df, encoded_info_df, tumor, config.config_map['untreated_labels'])
        start_time_control_info_df = \
            control_info_df[control_info_df['pert_time'].isin(config.config_map['untreated_times'])]
        pert_time_control_info_df = \
            control_info_df[~control_info_df['pert_time'].isin(config.config_map['untreated_times'])]
        start_time_control_data_df = control_data_df.loc[start_time_control_info_df.index]
        pert_time_control_data_df = control_data_df.loc[pert_time_control_info_df.index]

        encoded_treated_data_df, encoded_treated_info_df = self._get_data_and_info_by_tumor_and_perturbation(
            encoded_data_df, encoded_info_df, tumor, perturbation)

        start_time_control_center = np.mean(start_time_control_data_df)
        pert_time_control_center = np.mean(pert_time_control_data_df)
        calculated_pert_and_time_vector = calculated_latent_space_df - start_time_control_center
        calculated_pert_vector = calculated_latent_space_df - pert_time_control_center
        real_pert_and_time_vector = encoded_treated_data_df - start_time_control_center
        real_pert_vector = encoded_treated_data_df - pert_time_control_center

        pert_and_time_cos_sim = cosine_similarity(calculated_pert_and_time_vector, real_pert_and_time_vector)
        pert_cos_sim = cosine_similarity(calculated_pert_vector, real_pert_vector)
        pert_and_time_loss = np.mean(1 - pert_and_time_cos_sim)
        pert_loss = np.mean(1 - pert_cos_sim)

        # Angles for control in start time
        self.save_test_result('{0} cosine similarity, between encoded and calculated, pert and time vector: {1:.5f}'
                              .format(state, pert_and_time_loss))
        # Mann Whitney test
        p_value = self.calculate_mann_whitney_u_for_angles(calculated_latent_space_df, encoded_treated_data_df,
                                                           start_time_control_center, 'calculated to encoded',
                                                           encoded_treated_data_df, encoded_treated_data_df,
                                                           start_time_control_center, 'encoded to encoded',
                                                           state + ' start time control angles histogram')
        self.save_test_result(
            '{0}, Statistical tests, for angles in latent space to start time control, Mann whitney P: {1:.5f}'.format(
                state, p_value))

        # Angles for control in pert time
        self.save_test_result('{0} cosine similarity, between encoded and calculated, pert only vector: {1:.5f}'
                              .format(state, pert_loss))

        p_value = self.calculate_mann_whitney_u_for_angles(calculated_latent_space_df, encoded_treated_data_df,
                                                           pert_time_control_center, 'calculated to encoded',
                                                           encoded_treated_data_df, encoded_treated_data_df,
                                                           pert_time_control_center, 'encoded to encoded',
                                                           state + ' pert time control angles histogram')
        self.save_test_result(
            '{0}, Statistical tests, for angles in latent space to pert time control, Mann whitney P: {1:.5f}'.format(
                state, p_value))

        # Test cosine similarity between real cloud, and other clouds

        text_prefix = '{0} cosine similarity, between encoded and perturbation'.format(state)
        self._angle_test_for_all_perturbations(encoded_treated_data_df,
                                               encoded_data_and_info,
                                               start_time_control_center,
                                               pert_time_control_center,
                                               tumor,
                                               perturbation,
                                               text_prefix)

        # Test cosine similarity between calculated cloud, and other clouds
        text_prefix = '{0} cosine similarity, between calculated and perturbation'.format(state)
        self._angle_test_for_all_perturbations(calculated_latent_space_df,
                                               encoded_data_and_info,
                                               start_time_control_center,
                                               pert_time_control_center,
                                               tumor,
                                               perturbation,
                                               text_prefix)

    @staticmethod
    def test_by_tessellation(calculated_data_df, calculated_info_df, all_data_and_info, tested_pert_times, state,
                             out_file, save_function, printer=None):

        """
        Test if a svm model can say the difference between calculated and real data.
        :param calculated_data_df: DataFrame of calculated cloud
        :param calculated_info_df: DataFrame with info of calculated data.
        :param all_data_and_info: tuple of data and info, of the rest of the world
        :param tested_pert_times: list of perturbations times to test
        :param state: current state of test
        :param out_file: output file to save results
        :param save_function: function to save textual results to
        :param printer: printer handler, optional
        :return: tuple of 3 accuracies: whole train set, calculated set, and treated set
        """
        data_df, info_df = all_data_and_info
        tumor = calculated_info_df.iloc[0].tumor
        perturbation = calculated_info_df.iloc[0].perturbation

        # Drop treated, and add calculated
        if config.config_map['leave_data_out']:
            treated_info_df = info_df[(info_df.tumor == tumor) & (info_df.perturbation == perturbation)]
            treated_data_df = data_df.loc[treated_info_df.index]
            info_to_fit_df = info_df.drop(treated_info_df.index)
            data_to_fit_df = data_df.drop(treated_data_df.index)
            info_to_fit_df = info_to_fit_df.append(calculated_info_df[info_to_fit_df.columns])
            data_to_fit_df = data_to_fit_df.append(calculated_data_df)
            tessellation_tester = TessellationTester()
            tessellation_tester.fit(data_to_fit_df, info_to_fit_df)
            train_accuracy, _ = tessellation_tester.get_accuracy(data_to_fit_df, info_to_fit_df)
            calculated_accuracy, _ = tessellation_tester.get_accuracy(calculated_data_df, calculated_info_df)
            treated_accuracy, treated_df = tessellation_tester.get_accuracy(treated_data_df, treated_info_df)
        else:
            data_to_fit_df = data_df
            info_to_fit_df = info_df
            tessellation_tester = TessellationTester()
            tessellation_tester.fit(data_to_fit_df, info_to_fit_df)
            train_accuracy, _ = tessellation_tester.get_accuracy(data_to_fit_df, info_to_fit_df)
            calculated_accuracy, _ = tessellation_tester.get_accuracy(data_to_fit_df, info_to_fit_df)
            treated_accuracy, treated_df = tessellation_tester.get_accuracy(data_to_fit_df, info_to_fit_df)

        # Create third accuracy, without treated on control pert time
        control_pert_time = info_df[
            (info_df.tumor == tumor) &
            (info_df.perturbation.isin(config.config_map['untreated_labels'])) &
            (info_df.pert_time.isin(tested_pert_times))].iloc[0].numeric_labels
        filtered_treated_df = treated_df[treated_df.predicted != control_pert_time]
        try:
            treated_accuracy_without_pert_time = 100 * accuracy_score(filtered_treated_df.predicted,
                                                                      filtered_treated_df.real)
        except RuntimeError:
            # Runtime Error occurred in case of empty list, in case all the treated predicted as control pert time.
            treated_accuracy_without_pert_time = -1

        save_function(state + ' svm train accuracy: %.1f %%' % train_accuracy)
        save_function(state + ' svm calculated accuracy: %.1f %%' % calculated_accuracy)
        save_function(state + ' svm treated accuracy: %.1f %%' % treated_accuracy)
        save_function(
            state + ' svm treated accuracy without pert time: %.1f %%' % treated_accuracy_without_pert_time)

        # Plot decision boundaries if the dimension is 2
        if config.config_map['print_2d_tessellation'] and data_to_fit_df.shape[1] == 2 and printer is not None:
            printer.plot_decision_boundary(tessellation_tester.predict, data_to_fit_df, info_to_fit_df.numeric_labels,
                                           state + ' SVM boundaries')

        results_to_save = treated_df.copy()
        results_to_save.loc[~results_to_save.index.isin(filtered_treated_df.index), 'predicted'] = -1
        # Save treated results
        results_to_save.to_hdf(out_file, 'df')
        results = {'train': "{0:.2f}".format(train_accuracy),
                   'calculated': "{0:.2f}".format(calculated_accuracy),
                   'treated': "{0:.2f}".format(treated_accuracy),
                   'treated without treated on control pert time': "{0:.2f}".format(treated_accuracy_without_pert_time)}
        return results, treated_df

    def calculated_cloud_tests(self, calculated_tuple, tumor, perturbation, state):
        """
        Calculate cloud based on given tumor and perturbation, and do all the tests on it.
        :param calculated_tuple: tuple of calculated data in latent space, in real space, and it's info.
        :param tumor: name of tumor to test
        :param perturbation: name of perturbation to test
        :param state: state of running, for log text.
        """
        self.update_data()
        calculated_latent_space_df, calculated_real_space_df, calculated_info_df, reference_indexes = calculated_tuple

        # Test angles in latent space.
        latent_space_df = self.model.predict_latent_space(self.data_df)

        self._angle_test(calculated_latent_space_df, (latent_space_df, self.info_df), tumor, perturbation, state)

        # Statistical calculator
        logging.info(state + ' statistical tests between treated and encoded')
        # First create treated tuple
        treated_info_df = self.info_df[(self.info_df.tumor == tumor) &
                                       (self.info_df.perturbation == perturbation)]
        encoded_treated_data_df = latent_space_df.loc[treated_info_df.index]
        original_treated_data_df = self.data_df.loc[treated_info_df.index]

        # Now we can create the correlation calculator, and show the correlation distribution
        correlation_calculator = StatisticsCalculator(
            (encoded_treated_data_df, original_treated_data_df, treated_info_df))
        statistics_results =\
            correlation_calculator.do_statistical_tests(calculated_tuple, state, tumor, perturbation, self.data)

        # Test treated predictions
        encoded_data_and_info_to_classifier = self.filter_by_selector([latent_space_df, self.info_df],
                                                                      self.data.selectors[1])
        logging.info(state + ' test treated predictions')
        output_files = os.path.join(self.test_folder_path, 'svm_latent_space_results.h5')
        latent_space_svm_results = self.test_by_tessellation(calculated_latent_space_df,
                                                             calculated_info_df,
                                                             encoded_data_and_info_to_classifier,
                                                             config.config_map['perturbation_times'],
                                                            state + ' latent space',
                                                            output_files,
                                                            self.save_test_result,
                                                            self.printer)

        return latent_space_svm_results, statistics_results
