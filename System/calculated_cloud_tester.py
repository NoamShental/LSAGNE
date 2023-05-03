from latent_space_arithmetic import LatentSpaceArithmetic
from configuration import config
from data_handler import DataHandler

import os
import logging
import pandas as pd


class CalculatedCloudTester:
    def __init__(self, data, model, tester, printer, training_manager):
        """
        Initialize, set all parameters
        :param data: Data handler.
        :param model: Model handler.
        :param tester: Tester class.
        :param printer: Printer class.
        :param training_manager: Training Manager class
        """
        self.data = data
        self.model = model
        self.tester = tester
        self.printer = printer
        self.training_manager = training_manager
        self.latent_space_arithmetic = LatentSpaceArithmetic(data, model)

        self.tumor_column = 'tumor'
        self.perturbation_column = 'perturbation'
        self.time_column = 'pert_time'
        self.default_output_folder = config.config_map['arithmetic_output_folder']

    def _print_angles_vectors(self, calculated_data_and_info, encoded_data_and_info, calculated_reference_points,
                              perturbation, reference_points, state):
        """
        Print angles vectors between calculated and real points
        :param calculated_data_and_info: DataFrame with calculated data, in latent space, and their info.
        :param encoded_data_and_info: DataFrame with encoded samples, in latent space, and their info.
        :param calculated_reference_points: tuple of 5 samples in latent space, the reference points for calculated cloud
        :param perturbation: perturbation name of calculated data.
        :param reference_points: tuple of all reference points
        :param state: state of running, for saving the figure.
        """
        encoded_data_df, encoded_info_df = encoded_data_and_info
        calculated_latent_space_df, calculated_info_df = calculated_data_and_info

        # Get data and info DataFrames and reference points list, for both calculated and encoded samples
        # (for calculated's perturbation)
        sliced_indexes = encoded_info_df[encoded_info_df['perturbation'] == perturbation].index
        perturbation_data_df = encoded_data_df.loc[sliced_indexes]
        perturbation_info_df = encoded_info_df.loc[sliced_indexes]

        # Get reference points for perturbation data
        real_space_reference_points =\
            self.data.get_reference_points_to_data_slice(perturbation_info_df['original_index'], reference_points)

        # Change reference points indexes to be like here - "encoded_xyz"
        for i in range(len(real_space_reference_points)):
            real_space_reference_points[i].index = sliced_indexes
        latent_space_reference_points = [self.model.predict_latent_space(x) for x in real_space_reference_points]
        latent_space_calculated_reference_points =\
            [self.model.predict_latent_space(x) for x in calculated_reference_points]

        data_df = perturbation_data_df.append(calculated_latent_space_df, sort=False)
        info_df = perturbation_info_df.append(calculated_info_df, sort=False)
        reference_points = self.data.append_reference_points(latent_space_reference_points,
                                                             latent_space_calculated_reference_points)

        # Calculate perturbations and plot them.
        pert_and_time_vectors_df, pert_vectors_df = \
            self.printer.calculate_perturbation_vectors_latent_space(perturbation_data_df, latent_space_reference_points)
        self.printer.vectors_plot(pert_and_time_vectors_df, perturbation_info_df,
                                  '{} encoded pert and time angles'.format(state),
                                  os.path.join(self.printer.default_output_folder, 'Angles'))
        self.printer.vectors_plot(pert_vectors_df, perturbation_info_df,
                                  '{} encoded pert only angles'.format(state),
                                  os.path.join(self.printer.default_output_folder, 'Angles'))

        # Calculate perturbations and plot them.
        pert_and_time_vectors_df, pert_vectors_df =\
            self.printer.calculate_perturbation_vectors_latent_space(data_df, reference_points)
        self.printer.vectors_plot(pert_and_time_vectors_df, info_df,
                                  '{} calculated vs encoded pert and time angles'.format(state),
                                  os.path.join(self.printer.default_output_folder, 'Angles'))
        self.printer.vectors_plot(pert_vectors_df, info_df,
                                  '{} calculated vs encoded pert only angles'.format(state),
                                  os.path.join(self.printer.default_output_folder, 'Angles'))

    def calculate_classifier_label(self, info_to_calculate, override_series, data_df, info_df):
        """
        Create new column in info_to_calculate, 'correctly_classified', that will be 'Correct' if the classifier was
        right on it's original sample.
        :param info_to_calculate: all the info in the real space.
        :param override_series: Series with override parameters, to override the results
        :param data_df: DataFrame with all the data
        :param info_df: DataFrame with all the info
        """
        # First - get only the original samples:
        original_info_df = info_df.copy()
        original_data_df = data_df
        self.model.get_correctly_classified(original_data_df, original_info_df)
        original_info_df = original_info_df[['correctly_classified']]
        info_to_calculate = info_to_calculate.merge(original_info_df, left_on='original_index', right_index=True, how='left',
                                        suffixes=('_drop', ''))
        info_to_calculate = info_to_calculate.drop(columns='correctly_classified_drop')
        info_to_calculate.loc[override_series.index, 'correctly_classified'] = override_series
        return info_to_calculate

    @staticmethod
    def _get_correct_samples(data_df, info_df):
        correct_info_df = info_df[info_df['correctly_classified'] == 'Correct']
        correct_data_df = data_df.loc[correct_info_df.index]
        return correct_data_df, correct_info_df

    def _print_latent_space_figures(self, tsne_latent_space_df, all_latent_space_info_df, calculated_tumor,
                                    calculated_perturbation, tumor_list, reference_points, state):
        """
        Print all the figures of latent space:
        1. Only encoded, labeled by clouds and classifier right/wrong.
        2. For each special cloud - encoded + calculated, labeled by arithmetic and classifier.
        :param tsne_latent_space_df: all the data after tsne.
        :param all_latent_space_info_df: all the info of the data
        :param calculated_tumor: tumor of calculated cloud.
        :param calculated_perturbation: perturbation of calculated cloud.
        :param tumor_list: list of all tumors in the data set
        :param reference_points: tuple of all the reference points
        :param state: label for the figures.
        """
        # Divide to encoded and calculated data, and copy them, so the changes here won't be saved.
        encoded_info_df = all_latent_space_info_df[all_latent_space_info_df['arithmetic'] == 'encoded'].copy()
        calculated_info_df = all_latent_space_info_df[all_latent_space_info_df['arithmetic'] == 'calculated'].copy()

        # Set same scale to all figures
        self.printer.set_printing_plot_limits(tsne_latent_space_df)

        # Set output folder
        output_folder = os.path.join(self.printer.default_output_folder, 'Latent space')

        # Get encoded data:
        encoded_data_df = tsne_latent_space_df.loc[encoded_info_df.index]
        calculated_data_df = tsne_latent_space_df.loc[calculated_info_df.index]

        if config.config_map['latent_dim'] == 2:
            perturbation_vector, perturbation_and_time_vector = \
                self.latent_space_arithmetic.get_vectors(calculated_perturbation, reference_points)
            vectors_to_print = []
            for tumor in tumor_list:
                start_time_reference_point, pert_time_reference_point = self.data.untreated_dictionary[tumor]
                real_space_ref_points_df = pd.DataFrame([start_time_reference_point, pert_time_reference_point],
                                                         index=[0, 1], columns=start_time_reference_point.index)
                latent_space_ref_points_df = self.model.predict_latent_space(real_space_ref_points_df)
                vectors_to_print.extend([(latent_space_ref_points_df.loc[0].values, perturbation_and_time_vector),
                                         (latent_space_ref_points_df.loc[1].values, perturbation_vector)])
        else:
            vectors_to_print = None

        if config.config_map['print_clouds_color_by_arithmetics']:
            color_dict = {}
            for t in encoded_info_df.tumor.unique():
                t_info_df = encoded_info_df[encoded_info_df.tumor == t]
                for p in t_info_df.perturbation.unique():
                    p_info_df = t_info_df[t_info_df.perturbation == p]
                    if p not in config.config_map['untreated_labels']:
                        color_dict[p_info_df.iloc[0].classifier_labels] = 'red'
                    else:
                        for pert_time in p_info_df.pert_time.unique():
                            info_cloud_df = p_info_df[p_info_df.pert_time == pert_time]
                            if pert_time in config.config_map['untreated_times']:
                                color_dict[info_cloud_df.iloc[0].classifier_labels] = 'blue'
                            else:
                                color_dict[info_cloud_df.iloc[0].classifier_labels] = 'orange'

        elif config.config_map['print_clouds_custom_drugs'] != [] and \
                config.config_map['print_clouds_custom_tumors'] != []:
            color_dict = {}
            for label in encoded_info_df.classifier_labels.unique():
                color_dict[label] = "grey"
            for t in config.config_map['print_clouds_custom_tumors']:
                t_info_df = encoded_info_df[encoded_info_df.tumor == t]
                colors = ['red', 'purple']
                i = 0
                for p in config.config_map['print_clouds_custom_drugs']:
                    p_info_df = t_info_df[t_info_df.perturbation == p]
                    if p_info_df.shape[0] != 0:
                        if p not in config.config_map['untreated_labels']:
                            color_dict[p_info_df.iloc[0].classifier_labels] = colors[i]
                            i += 1
                            i %= len(colors)
                        else:
                            for pert_time in p_info_df.pert_time.unique():
                                info_cloud_df = p_info_df[p_info_df.pert_time == pert_time]
                                if pert_time in config.config_map['untreated_times']:
                                    color_dict[info_cloud_df.iloc[0].classifier_labels] = 'blue'
                                else:
                                    color_dict[info_cloud_df.iloc[0].classifier_labels] = 'orange'
        else:
            color_dict = None
        if color_dict is not None:
            # Print encoded by clouds and classifier.
            self.printer.sns_plot(encoded_data_df,
                                  encoded_info_df,
                                  ['classifier_labels'],
                                  state + ' latent space special colors',
                                  output_directory=output_folder,
                                  color_dict=color_dict,
                                  vectors_to_print=vectors_to_print,
                                  show_legend=[False])

        # Print encoded by clouds and classifier.
        self.printer.sns_plot(encoded_data_df,
                              encoded_info_df,
                              ['classifier_labels'],
                              state + ' latent space',
                              output_directory=output_folder,
                              color_dict=None,
                              vectors_to_print=vectors_to_print,
                              show_legend=[False])

        if config.config_map['print_custom_correctly_classified_drugs'] != [] and config.config_map['print_custom_correctly_classified_tumors'] != []:
            encoded_info_to_print_df = encoded_info_df[
                (encoded_info_df.tumor.isin(config.config_map['print_custom_correctly_classified_tumors'])) &
                (encoded_info_df.perturbation.isin(config.config_map['print_custom_correctly_classified_drugs']))]
            encoded_data_to_print_df = encoded_data_df.loc[encoded_info_df.index]
            # Print encoded by clouds and classifier.
            self.printer.sns_plot(encoded_data_to_print_df,
                                  encoded_info_to_print_df,
                                  ['correctly_classified'],
                                  state + ' latent space custom clouds',
                                  color_dict={'Correct': 'green', "Failed": 'black'},
                                  output_directory=output_folder,
                                  vectors_to_print=vectors_to_print,
                                  show_legend=[True])

        # Print encoded by clouds and classifier.
        self.printer.sns_plot(encoded_data_df,
                              encoded_info_df,
                              ['correctly_classified'],
                              state + ' latent space',
                              color_dict={'Correct': 'green', "Failed": 'black'},
                              output_directory=output_folder,
                              vectors_to_print=vectors_to_print,
                              show_legend=[True])

        # For each calculated cloud, print that cloud by arithmetic
        encoded_control_data, encoded_control_info = DataHandler.get_data_and_info_by_tumor_and_perturbation(
            encoded_data_df, encoded_info_df, calculated_tumor, config.config_map['untreated_labels'])
        sliced_indexes = encoded_control_info[
            encoded_control_info['pert_time'].isin(config.config_map['untreated_times'])].index
        encoded_start_control_info_df = encoded_control_info.loc[sliced_indexes]
        encoded_start_control_data_df = encoded_control_data.loc[sliced_indexes]

        sliced_indexes = encoded_control_info[
            ~encoded_control_info['pert_time'].isin(config.config_map['untreated_times'])].index
        encoded_pert_control_info_df = encoded_control_info.loc[sliced_indexes]
        encoded_pert_control_data_df = encoded_control_data.loc[sliced_indexes]

        encoded_start_control_info_df['arithmetic'] = 'Start time control'
        encoded_pert_control_info_df['arithmetic'] = 'Pert time control'

        encoded_treated_data, encoded_treated_info = DataHandler.get_data_and_info_by_tumor_and_perturbation(
            encoded_data_df, encoded_info_df, calculated_tumor, calculated_perturbation)
        encoded_treated_info['arithmetic'] = 'Encoded treated'

        calculated_cloud_data_df, calculated_cloud_info_df = DataHandler.get_data_and_info_by_tumor_and_perturbation(
            calculated_data_df, calculated_info_df, calculated_tumor, calculated_perturbation)

        # Show different colors to correct and classifier
        encoded_start_control_info_df['arithmetic'] =\
            encoded_start_control_info_df['arithmetic'] + ' ' + encoded_start_control_info_df['correctly_classified']
        encoded_pert_control_info_df['arithmetic'] = \
            encoded_pert_control_info_df['arithmetic'] + ' ' + encoded_pert_control_info_df['correctly_classified']
        calculated_cloud_info_df['arithmetic'] =\
            calculated_cloud_info_df['arithmetic'] + ' ' + calculated_cloud_info_df['correctly_classified']
        encoded_treated_info['arithmetic'] = \
            encoded_treated_info['arithmetic'] + ' svm ' + encoded_treated_info['correctly_classified']

        # Create big DataFrames for the data and info
        current_data_df = encoded_start_control_data_df.append(
            [encoded_pert_control_data_df, calculated_cloud_data_df, encoded_treated_data])
        current_info_df = encoded_start_control_info_df.append(
            [encoded_pert_control_info_df, calculated_cloud_info_df, encoded_treated_info])

        # Printing properties
        color_dict = {'Start time control Correct': 'blue', 'Start time control Failed': 'm',
                      'Pert time control Correct': 'orange', 'Pert time control Failed': 'lightsalmon',
                      'calculated Correct': 'green', 'calculated Failed': 'mediumseagreen',
                      'Encoded treated svm Correct': 'red',
                      'Encoded treated svm Failed': 'black'}
        #markers = [".", ".", "*", "h"]
        #markers = markers[:len(current_info_df['arithmetic'].unique())]
        #markers = None

        if config.config_map['latent_dim'] == 2:
            perturbation_vector, perturbation_and_time_vector = \
                self.latent_space_arithmetic.get_vectors(calculated_perturbation, reference_points)
            start_time_reference_point, pert_time_reference_point = self.data.untreated_dictionary[calculated_tumor]
            real_space_ref_points_df = pd.DataFrame([start_time_reference_point, pert_time_reference_point],
                                                     index=[0, 1], columns=start_time_reference_point.index)
            latent_space_ref_points_df = self.model.predict_latent_space(real_space_ref_points_df)
            vectors_to_print = [(latent_space_ref_points_df.loc[0].values, perturbation_and_time_vector),
                                (latent_space_ref_points_df.loc[1].values, perturbation_vector)]
        else:
            vectors_to_print = None

        # Add statistics info
        legend = {}
        for curr_class in current_info_df['arithmetic'].unique():
            if curr_class == 'calculated Correct' or curr_class == 'calculated Failed':
                continue
            num_of_samples = current_info_df[current_info_df['arithmetic'] == curr_class].shape[0]
            legend[curr_class] = num_of_samples

        # If requested - append dose to figure
        if config.config_map['print_dose']:
            treated_indexes = encoded_treated_info.index
            current_info_df.loc[treated_indexes, 'arithmetic'] =\
                current_info_df.loc[treated_indexes]['arithmetic'] + ': ' + \
                current_info_df.loc[treated_indexes]['pert_dose'].map(str)

        # Print encoded by clouds and classifier.
        self.printer.sns_plot(current_data_df,
                              current_info_df,
                              ['arithmetic'],
                              '{0} {1} calculated {2} {3}'.format(state, config.config_map['test_number'],
                                                                  calculated_tumor, calculated_perturbation),
                              output_directory=output_folder,
                              color_dict=color_dict,
                              vectors_to_print=vectors_to_print)

    def _print_real_space_figures(self, tsne_real_space_df, all_real_space_info_df, calculated_tumor,
                                  calculated_perturbation, state):
        """
        Print 6 figures in latent space:
            1. original labeled by clouds and classifier wrong/right.
            2. decoded labeled by clouds and classifier wrong/right.
            3. for each special cloud:
                a. original + decoded + calculated + control, labeled by arithmetic and classifier.
            4. Before and after encode-decode, for each cloud.
        :param tsne_real_space_df: all real space data, after TSNE.
        :param all_real_space_info_df: all real space info, after TSNE.
        :param calculated_tumor: tumor of calculated cloud.
        :param calculated_perturbation: perturbation of calculated cloud.
        :param state: running state, for output figures labels.
        """
        # Divide to encoded and calculated data, and copy them, so the changes here won't be saved.
        original_info_df = all_real_space_info_df[all_real_space_info_df['arithmetic'] == 'original'].copy()
        decoded_info_df = all_real_space_info_df[all_real_space_info_df['arithmetic'] == 'decoded'].copy()
        calculated_info_df = all_real_space_info_df[all_real_space_info_df['arithmetic'] == 'calculated'].copy()

        # Set same scale to all figures
        self.printer.set_printing_plot_limits(tsne_real_space_df)

        # Set output folder
        output_folder = os.path.join(self.printer.default_output_folder, 'Real space')

        # Get data:
        original_data_df = tsne_real_space_df.loc[original_info_df.index]
        decoded_data_df = tsne_real_space_df.loc[decoded_info_df.index]
        calculated_data_df = tsne_real_space_df.loc[calculated_info_df.index]

        if config.config_map['print_clouds_color_by_arithmetics']:
            color_dict = {}
            for t in original_info_df.tumor.unique():
                t_info_df = original_info_df[original_info_df.tumor == t]
                for p in t_info_df.perturbation.unique():
                    p_info_df = t_info_df[t_info_df.perturbation == p]
                    if p not in config.config_map['untreated_labels']:
                        color_dict[p_info_df.iloc[0].classifier_labels] = 'red'
                    else:
                        for pert_time in p_info_df.pert_time.unique():
                            info_cloud_df = p_info_df[p_info_df.pert_time == pert_time]
                            if pert_time in config.config_map['untreated_times']:
                                color_dict[info_cloud_df.iloc[0].classifier_labels] = 'blue'
                            else:
                                color_dict[info_cloud_df.iloc[0].classifier_labels] = 'orange'

        else:
            color_dict = None

        # Print original by clouds and classifier.
        self.printer.sns_plot(original_data_df,
                              original_info_df,
                              ['classifier_labels'],
                              state + ' All original ',
                              show_legend=[False, True],
                              output_directory=output_folder,
                              color_dict=color_dict)

        # Print original by clouds and classifier.
        self.printer.sns_plot(original_data_df,
                              original_info_df,
                              ['correctly_classified'],
                              state + ' All original ',
                              show_legend=[False, True],
                              output_directory=output_folder)

        # Print decoded by clouds and classifier.
        self.printer.sns_plot(decoded_data_df,
                              decoded_info_df,
                              ['classifier_labels'],
                              state + ' All decoded ',
                              show_legend=[False, True],
                              output_directory=output_folder,
                              color_dict=color_dict)

        # Print decoded by clouds and classifier.
        self.printer.sns_plot(decoded_data_df,
                              decoded_info_df,
                              ['correctly_classified'],
                              state + ' All decoded ',
                              show_legend=[False, True],
                              output_directory=output_folder)

        # For each calculated cloud, print that cloud by arithmetic
        decoded_cloud_data_df, decoded_cloud_info_df = \
            DataHandler.get_data_and_info_by_tumor_and_perturbation(decoded_data_df, decoded_info_df, calculated_tumor,
                                                                    calculated_perturbation)
        decoded_cloud_info_df['arithmetic'] = 'Decoded treated'

        original_control_data_df, original_control_info_df = \
            DataHandler.get_data_and_info_by_tumor_and_perturbation(original_data_df, original_info_df,
                                                                    calculated_tumor,
                                                                    config.config_map['untreated_labels'])

        sliced_indexes = original_control_info_df[
            original_control_info_df['pert_time'].isin(config.config_map['untreated_times'])].index
        original_start_control_info_df = original_control_info_df.loc[sliced_indexes]
        original_start_control_data_df = original_control_data_df.loc[sliced_indexes]
        sliced_indexes = original_control_info_df[
            ~original_control_info_df['pert_time'].isin(config.config_map['untreated_times'])].index
        original_pert_control_info_df = original_control_info_df.loc[sliced_indexes]
        original_pert_control_data_df = original_control_data_df.loc[sliced_indexes]

        original_start_control_info_df['arithmetic'] = 'Start time control'
        original_pert_control_info_df['arithmetic'] = 'Pert time control'

        original_treated_data_df, original_treated_info_df = \
            DataHandler.get_data_and_info_by_tumor_and_perturbation(original_data_df, original_info_df,
                                                                    calculated_tumor, calculated_perturbation)
        original_treated_info_df['arithmetic'] = 'Original treated'

        calculated_cloud_data_df, calculated_cloud_info_df = \
            DataHandler.get_data_and_info_by_tumor_and_perturbation(calculated_data_df, calculated_info_df,
                                                                    calculated_tumor, calculated_perturbation)

        # Show different colors to correct and classifier
        original_start_control_info_df['arithmetic'] =\
            original_start_control_info_df['arithmetic'] + ' ' + original_start_control_info_df['correctly_classified']
        original_pert_control_info_df['arithmetic'] = \
            original_pert_control_info_df['arithmetic'] + ' ' + original_pert_control_info_df['correctly_classified']
        calculated_cloud_info_df['arithmetic'] =\
            calculated_cloud_info_df['arithmetic'] + ' ' + calculated_cloud_info_df['correctly_classified']
        decoded_cloud_info_df['arithmetic'] = \
            decoded_cloud_info_df['arithmetic'] + ' svm ' + decoded_cloud_info_df['correctly_classified']
        original_treated_info_df['arithmetic'] = \
            original_treated_info_df['arithmetic'] + ' svm ' + original_treated_info_df['correctly_classified']

        # Create big DataFrames for the data and info
        current_data_df = original_start_control_data_df.append(
            [original_pert_control_data_df, calculated_cloud_data_df, decoded_cloud_data_df, original_treated_data_df])
        current_info_df = original_start_control_info_df.append(
            [original_pert_control_info_df, calculated_cloud_info_df, decoded_cloud_info_df, original_treated_info_df])

        # Printing properties
        color_dict = {'Start time control Correct': 'blue', 'Start time control Failed': 'm',
                      'Pert time control Correct': 'orange', 'Pert time control Failed': 'lightsalmon',
                      'calculated Correct': 'green', 'calculated Failed': 'mediumseagreen',
                      'Decoded treated svm Correct': 'red',
                      'Decoded treated svm Failed': 'lightslategray',
                      'Original treated svm Correct': 'c',
                      'Original treated svm Failed': 'darkcyan'}
        #markers = [".", ".", "*", "h", "P"]
        #markers = markers[:len(current_info_df['arithmetic'].unique())]
        #markers = None

        # Add statistics info
        classifier_results = {}
        for curr_class in current_info_df['arithmetic'].unique():
            if curr_class in ['calculated Correct', 'calculated Failed', "Decoded treated"]:
                continue
            num_of_samples = current_info_df[current_info_df['arithmetic'] == curr_class].shape[0]
            classifier_results[curr_class] = num_of_samples

        # Print original + calculated + decoded, by arithmetic.
        self.printer.sns_plot(current_data_df,
                              current_info_df,
                              ['arithmetic'],
                              '{0} {1} calculated {2} {3}'.format(state, config.config_map['test_number'],
                                                                  calculated_tumor, calculated_perturbation),
                              output_directory=output_folder,
                              color_dict=color_dict)

        # Print original and decoded clouds, for each cloud in data
        original_and_decoded_data_df = original_data_df.append(decoded_data_df)
        original_and_decoded_info_df = original_info_df.append(decoded_info_df)
        self.printer.print_original_and_decoded(original_and_decoded_data_df,
                                                original_and_decoded_info_df,
                                                'classifier_labels',
                                                'is_decoded',
                                                state)
        return classifier_results

    def test_latent_space_arithmetic(self, state):
        """
        Perform and check the latent-space arithmetic.
        Main stages:
        1. Move all the data to latent space and real space after encode-decode.
        2. Calculate the cloud, and move to real space.
        3. Test the cloud.
        3. Create 2 DataFrame - in latent space and real space, that will contain the encoded/decoded data
                                and calculated cloud.
        4. Move latent space DataFrame through TSNE, and print all figures.
        6. Move real space DataFrame through TSNE, and print all figures.
        :param state: string that describe current state of the test (start/end etc.)
        """
        results = {}
        data_df, info_df, reference_points = self.data.get_all_data_and_info()

        # Set additional needed columns
        info_df = info_df.copy()
        info_df['arithmetic'] = 'original'
        info_df['correctly_classified'] = 'unknown'
        info_df['is_decoded'] = 0

        # Create encoded and decoded data sets and their info
        encoded_data_df = self.model.predict_latent_space(data_df)
        encoded_info_df = info_df.copy()
        encoded_info_df['arithmetic'] = 'encoded'
        encoded_data_df.index = 'encoded_' + encoded_data_df.index
        encoded_info_df.index = 'encoded_' + encoded_info_df.index
        encoded_data_and_info = (encoded_data_df, encoded_info_df)

        tumor = config.config_map['leave_out_tissue_name']
        perturbation = config.config_map['leave_out_perturbation_name']
        logging.info('%s latent space arithmetic for tumor %s perturbation %s', state, tumor, perturbation)

        # Calculate new cloud
        calculated_tuple = \
            self.latent_space_arithmetic.calculate_perturbations_effect_and_info(
                encoded_data_and_info, tumor, [perturbation], config.config_map['perturbation_times'])

        # If requested, save calculated cloud
        if config.config_map['save_calculated_cloud']:
            if not os.path.isdir(self.default_output_folder):
                os.makedirs(self.default_output_folder)
            calculated_latent_space_df, calculated_real_space_df, calculated_info_df, calculated_reference_points = \
                calculated_tuple
            calculated_latent_space_df.to_hdf(os.path.join(self.default_output_folder, 'latent_space_calculated.h5'),
                                              'df')
            calculated_real_space_df.to_hdf(os.path.join(self.default_output_folder, 'real_space_calculated.h5'), 'df')
            calculated_info_df.to_csv(os.path.join(self.default_output_folder, 'info_calculated.csv'))

        # Print the angles vectors
        calculated_latent_space_df, calculated_real_space_df, calculated_info_df, calculated_reference_points = \
            calculated_tuple
        self._print_angles_vectors((calculated_latent_space_df, calculated_info_df), encoded_data_and_info,
                                   calculated_reference_points, perturbation, reference_points, state)

        # Do all numeric tests to the new cloud
        svm_results, statistics_results = self.tester.calculated_cloud_tests(calculated_tuple, tumor, perturbation,
                                                                             state)
        results['svm_latent_space'] = svm_results[0]
        results['statistics_results'] = statistics_results



        # Predict decoded data to show it
        decoded_data_df, _, _ = self.model.predict_variational_loss(data_df)
        decoded_info_df = info_df.copy()
        decoded_info_df['arithmetic'] = 'decoded'
        decoded_info_df['is_decoded'] = 1
        decoded_data_df.index = 'decoded_' + decoded_data_df.index
        decoded_info_df.index = 'decoded_' + decoded_info_df.index

        # Create 4 DataFrames:
        # 1. All the data in latent space
        # 2. All the data in real space
        # 3. All the info in latent space
        # 4. All the info in real space
        encoded_data_df, encoded_info_df = encoded_data_and_info
        all_latent_space_data_df = encoded_data_df.append(calculated_latent_space_df)
        all_real_space_data_df = data_df.append([decoded_data_df, calculated_real_space_df])
        all_latent_space_info_df = encoded_info_df.append(calculated_info_df)
        all_real_space_info_df = info_df.append([decoded_info_df, calculated_info_df])

        # Add correctly classified label
        svm_results_df = svm_results[1]
        svm_results_df['success'] = \
            svm_results_df['real'] == svm_results_df['predicted']
        svm_results_df['success'] = svm_results_df['success'].apply(
            lambda x: 'Correct' if x else 'Failed')

        # Create a copy and change it's index from xyz to encoded_xyz
        encoded_svm_results_df = svm_results_df.copy()
        encoded_svm_results_df.index = 'encoded_' + encoded_svm_results_df.index
        all_latent_space_info_df = self.calculate_classifier_label(all_latent_space_info_df,
                                                                   encoded_svm_results_df['success'],
                                                                   data_df, info_df)

        # Add another copy with "decoded_xyz" to decoded data
        decoded_svm_results_df = svm_results_df.copy()
        decoded_svm_results_df.index = 'decoded_' + decoded_svm_results_df.index
        svm_results_df = svm_results_df.append(decoded_svm_results_df)
        all_real_space_info_df = self.calculate_classifier_label(all_real_space_info_df,
                                                                 svm_results_df['success'],
                                                                 data_df, info_df)

        # Do TSNE to latent space, and show figures
        logging.info(state + ' latent space TSNE')
        tsne_latent_space_df = self.printer.do_tsne(all_latent_space_data_df)
        self._print_latent_space_figures(tsne_latent_space_df, all_latent_space_info_df, tumor, perturbation,
                                         info_df.tumor.unique(), reference_points, state)
        logging.info(state + ' finished print latent space figures')

        # Do TSNE to real space, and show figures
        logging.info(state + ' real space TSNE')
        tsne_real_space_df = self.printer.do_tsne(all_real_space_data_df)
        classifier_results = self._print_real_space_figures(tsne_real_space_df, all_real_space_info_df, tumor,
                                                            perturbation, state)
        logging.info(state + ' finished print real space figures')
        results.update(classifier_results)
        return results
