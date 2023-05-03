from configuration import config, mock

import os
import pickle
import pandas as pd
import numpy as np
import joblib
import logging
from keras.utils import np_utils


class DataHandler:
    """
    This class will handle all the data, and keep it updated
    """
    def __init__(self, find_reference_points=True):
        """
        Load all the data, and split to train and test sets
        :param find_reference_points: if set, find reference points
        """
        self.perturbation_column = 'perturbation'
        self.tumor_column = 'tumor'
        self.time_column = 'pert_time'
        self.pick_reference_point_function = self.dummy_reference_points_choose
        self._load_data(find_reference_points)

        # Create classifier labels
        self.info_df['classifier_labels'] = self.info_df['tumor'] + ' ' + self.info_df["perturbation"] + ' ' + self.info_df["pert_time"].map(str)

        # Create numeric labels, this must be placed after all the filtering
        labels, _ = pd.factorize(self.info_df["classifier_labels"])
        self.info_df["numeric_labels"] = labels

        # Create one-hot encoding
        self.info_df['one_hot_labels'] = np_utils.to_categorical(self.info_df['numeric_labels']).tolist()

        # Add configuration keys by the data
        config.config_map['input_size'] = self.data_df.shape[1]
        config.config_map['num_classes'] = max(self.info_df["numeric_labels"]) + 1

        if config.config_map['random_treated']:
            self._random_treated_data()

        self.info_df['original_index'] = self.info_df.index

        # If configured - leave part of the data out.
        if config.config_map['leave_data_out']:
            self._leave_data_out()

        # Split the data to train and test
        self._split_to_train_and_test()

        # If configured - genes filtering.
        if config.config_map['genes_filtering_is_needed']:
            self.drugs_dominates_genes_dict = {}
            for drug in config.config_map['basic_perturbations']:
                # Load the file contains the most dominant genes of the given drug.
                curr_dominant_genes_filename = os.path.join(config.config_map['organized_data_folder'],
                                                            'common_0_24_{}.txt'.format(drug))
                with open(curr_dominant_genes_filename) as f:
                    self.drugs_dominates_genes_dict[drug] = f.read().splitlines()

        logging.info("Total number of loaded data samples={}".format(self.data_df.shape[0]))
        logging.info("Total number of loaded perturbations={}: {}".format(len((self.info_df['perturbation'].unique())),
                                                                          list(self.info_df['perturbation'].unique())))
        logging.info("Total Number of loaded tissues={}: {}".format(len((self.info_df['tumor'].unique())),
                                                                 list(self.info_df['tumor'].unique())))

    def _random_treated_data(self):
        """
        Take samples from other clouds to be the leave out clouds
        """
        treated_cloud_info_df = self.info_df[(self.info_df.perturbation == config.config_map['leave_out_perturbation_name']) &
                                             (self.info_df.tumor == config.config_map['leave_out_tissue_name'])]
        rest_data_info_df = self.info_df.loc[~self.info_df.index.isin(treated_cloud_info_df.index)]
        info_to_drop = rest_data_info_df.sample(treated_cloud_info_df.shape[0])
        self.info_df.drop(info_to_drop.index, inplace=True)
        self.data_df.loc[treated_cloud_info_df.index] = self.data_df.loc[info_to_drop.index].values
        self.data_df.drop(info_to_drop.index, inplace=True)


    @staticmethod
    def dummy_reference_points_choose(data_df):
        """
        Function to choose reference point for given data.
        All the given data was taken from the same cloud.
        This is a dummy function that just return the first one.
        :param data_df: DataFrame with samples
        :return: chosen reference point index.
        """
        if data_df.shape[0] == 0:
            zeros_arr = np.zeros(shape=data_df.shape[1], dtype=data_df.dtypes[0])
            return pd.Series(zeros_arr, index=data_df.columns)
        return data_df.iloc[0]

    @staticmethod
    def find_reference_point_by_center_of_cloud(processing_function, data_df):
        """
        Find reference point for cloud, by the point that most close to center of cloud
        :param processing_function: function to call to the samples, before calculating the center
        :param data_df: DataFrame with samples, to calculate their reference points
        :return: chosen reference point
        """
        if data_df.shape[0] == 0:
            zeros_arr = np.zeros(shape=data_df.shape[1], dtype=data_df.dtypes[0])
            return pd.Series(zeros_arr, index=data_df.columns)
        processed_df = processing_function(data_df)
        cloud_center = processed_df.values.mean(axis=0)
        index = np.linalg.norm(processed_df.values - cloud_center, axis=1).argmin()
        return data_df.iloc[index]


    @staticmethod
    def get_data_and_info_by_tumor_and_perturbation(data_df, info_df, tumor, perturbation):
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

    def get_all_data_and_info(self, copy=False):
        """
        Create all the data, info and reference points (both data and leaveout.
        :return: tuple of data, info and reference points
        """
        # Create DataFrame with all the data, copied.
        # Check if leave data out, and if we already left some data out
        if config.config_map['leave_data_out'] and hasattr(self, 'left_data_df'):
            if copy:
                data_df = self.data_df.copy().append(self.left_data_df.copy())
                info_df = self.info_df.copy().append(self.left_info_df.copy())
                reference_points = [self.reference_points[i].append(self.left_reference_points[i]) for i in
                                    range(len(self.reference_points))]
            else:
                data_df = self.data_df.append(self.left_data_df)
                info_df = self.info_df.append(self.left_info_df)
                reference_points = [self.reference_points[i].append(self.left_reference_points[i]) for i in
                                    range(len(self.reference_points))]
        else:
            if copy:
                data_df = self.data_df.copy()
                info_df = self.info_df.copy()
                reference_points = [self.reference_points[i] for i in range(len(self.reference_points))]
            else:
                data_df = self.data_df
                info_df = self.info_df
                reference_points = [self.reference_points[i] for i in range(len(self.reference_points))]
        return data_df, info_df, reference_points

    def get_unscaled_data(self, data_df):
        """
        Unscaled given data to original data range.
        :param data_df: scaled data to get the original values
        :return: DataFrame with unscaled data.
        """
        if not mock:
            data_np = self.scaler.inverse_transform(data_df)
        return pd.DataFrame(data_np, data_df.index, data_df.columns)

    def get_12k_data(self, data_df, need_unscale=True):
        """
        Calculate 12K genes matrix from given data df
        :param data_df: input data df with CMap genes
        :param need_unscale: if true, unscale the data before transform to 12K
        :return: DataFrame with same rows and index, but with all 12K genes
        """
        if need_unscale:
            data_df = self.get_unscaled_data(data_df)
        transformation_matrix_path = os.path.join(config.config_map['organized_data_folder'],
                                                  config.config_map['12K_matrix'])
        transformation_matrix = pd.read_csv(transformation_matrix_path)
        transformation_matrix.set_index('rid', inplace=True)
        offset_series = transformation_matrix['OFFSET']
        mult_matrix = transformation_matrix.drop('OFFSET', axis=1)

        # Drop missing columns from transformation matrix
        mult_columns = set(mult_matrix.columns)
        data_columns = set(data_df.columns)
        missing_columns = mult_columns.difference(data_columns)
        mult_matrix.drop(list(missing_columns), inplace=True, axis=1)

        # Dot product the data with the transformation matrix, and add the offset
        result_df = data_df.dot(mult_matrix.T)
        result_df += offset_series

        return result_df.join(data_df)

    def _leave_data_out(self):
        """
        Leave data out according to the given configuration.
        The left data should be calculated after model learning, using the latent space.
        """
        req_tumor = config.config_map['leave_out_tissue_name']
        req_perturbation = config.config_map['leave_out_perturbation_name']

        self.left_info_df = self.info_df.loc[(self.info_df['tumor'] == req_tumor) &
                                             (self.info_df['perturbation'] == req_perturbation)]
        self.info_df = self.info_df.loc[~((self.info_df['tumor'] == req_tumor) &
                                          (self.info_df['perturbation'] == req_perturbation))]

        self.left_data_df = self.data_df.loc[self.left_info_df.index]
        self.data_df = self.data_df.loc[self.info_df.index]

        # Split reference points.
        self.left_reference_points = self.get_reference_points_to_data_slice(self.left_info_df.index)
        self.reference_points = self.get_reference_points_to_data_slice(self.info_df.index)

        logging.info("Total number of left out data samples={}".format(self.left_data_df.shape[0]))
        logging.info("Total number of left out perturbations={}:{}"
                     .format(len(self.left_info_df['perturbation'].unique()),
                             list(self.left_info_df['perturbation'].unique())))
        logging.info("Total Number of left out tissues={}:{}"
                     .format(len(self.left_info_df['tumor'].unique()),
                             list(self.left_info_df['tumor'].unique())))

    def _create_perturbation_reference_dictionary(self, untreated_labels, update_missing_only=False):
        """
        Create reference dictionary - dictionary of reference points for each perturbation:
            perturbation -> (treated_reference_sample, pert_time_untreated_reference_sample, control_time_untreated_reference_point )
        :param untreated_labels: list of control names in the perturbation column
        :param update_missing_only: if set, find only missing perts
        :return: the reference dictionary
        """

        # Create reference points for each perturbation.
        perturbations_list = self.info_df[self.perturbation_column].unique()
        if not update_missing_only:
            self.p_dictionary = {}
        for p in perturbations_list:
            if p in self.p_dictionary:
                continue

            # Slice DF of untreated samples with same tumor,
            # and take one sample of pert time, and one sample of start time
            tumor = self.reference_tumors_dict[p]
            tumor_info_df = self.info_df[self.info_df[self.tumor_column] == tumor]
            untreated_info_df = tumor_info_df[
                tumor_info_df[self.perturbation_column].isin(untreated_labels)]

            start_time_control_info_index = untreated_info_df[
                untreated_info_df[self.time_column].isin(config.config_map['untreated_times'])].index
            start_time_control_reference_sample =\
                self.pick_reference_point_function(self.data_df.loc[start_time_control_info_index])

            pert_time_control_info_index = untreated_info_df[
                ~untreated_info_df[self.time_column].isin(config.config_map['untreated_times'])].index
            pert_time_control_reference_sample =\
                self.pick_reference_point_function(self.data_df.loc[pert_time_control_info_index])

            treated_info_index = tumor_info_df[tumor_info_df[self.perturbation_column] == p].index
            treated_reference_sample = self.pick_reference_point_function(self.data_df.loc[treated_info_index])
            self.p_dictionary[p] = (treated_reference_sample,
                                    start_time_control_reference_sample,
                                    pert_time_control_reference_sample)

    def _create_untreated_reference_point(self, untreated_labels, update_missing_only=False):
        """
        Create dictionary with untreated samples for each tissue, one in perturbation time, and one in start time
        tissue -> pert_time_control_sample, control_untreated_sample
        :param untreated_labels: list of control names in the perturbation column
        :param update_missing_only: if set, find only missing perts
        """
        untreated_samples_df = self.info_df[
            self.info_df[self.perturbation_column].isin(untreated_labels)]
        untreated_tissues_list = untreated_samples_df[self.tumor_column].unique()
        if not update_missing_only:
            self.untreated_dictionary = {}
        for t in untreated_tissues_list:
            if t in self.untreated_dictionary:
                continue
            untreated_tumor_info_df = untreated_samples_df[untreated_samples_df[self.tumor_column] == t]
            start_time_control_index = untreated_tumor_info_df[
                untreated_tumor_info_df[self.time_column].isin(config.config_map['untreated_times'])].index
            pert_time_control_index = untreated_tumor_info_df[
                ~untreated_tumor_info_df[self.time_column].isin(config.config_map['untreated_times'])].index
            start_time_reference_point = self.pick_reference_point_function(self.data_df.loc[start_time_control_index])
            pert_time_reference_point = self.pick_reference_point_function(self.data_df.loc[pert_time_control_index])
            self.untreated_dictionary[t] = (start_time_reference_point, pert_time_reference_point)

    def _create_clouds_reference_points_dictionary(self, untreated_labels, update_missing_only=False):
        """
        Create dictionary with reference point for each cloud.
        :param untreated_labels: list of control names in the perturbation column
        :param update_missing_only: if set, find only missing perts
        """
        data_df, info_df, _ = self.get_all_data_and_info()
        if not update_missing_only:
            self.clouds_reference_dictionary = {}
        for p in info_df[self.perturbation_column].unique():
            # check for untreated labels - reference points already in  untreated_dictionary
            if p in untreated_labels:
                continue

            perturbation_info_df = info_df[info_df[self.perturbation_column] == p]
            for t in perturbation_info_df[self.tumor_column].unique():
                if (p, t) in self.clouds_reference_dictionary:
                    continue

                # If we already decided for reference point for that cloud, in the perturbation dictionary, just choose
                # that point
                if self.reference_tumors_dict[p] == t:
                    self.clouds_reference_dictionary[(p, t)] = self.p_dictionary[p][0]
                else:
                    cloud_index = perturbation_info_df[perturbation_info_df[self.tumor_column] == t].index
                    self.clouds_reference_dictionary[(p, t)] =\
                        self.pick_reference_point_function(data_df.loc[cloud_index])

    def _fill_reference_points_for_untreated_perturbation(self, perturbation):
        """
        Fill in untreated reference points.
        :param perturbation: current perturbation name.
        """
        # Unpack reference points
        start_time_control_points, treated_reference_points, start_time_control_reference_points, \
            pert_time_control_points, pert_time_control_reference_points, same_cloud_reference_points =\
            self.reference_points[0:6]
        perts_ref_points = self.reference_points[6:]

        # Get info_df for perturbation
        pert_info_df = self.info_df[self.info_df[self.perturbation_column] == perturbation]

        # Untreated perturbation, we don't use pert_time_control_points and pert_time_control_reference_points
        pert_time_control_points.loc[pert_info_df.index] = self.data_df.loc[pert_info_df.index]
        pert_time_control_reference_points.loc[pert_info_df.index] = self.data_df.loc[pert_info_df.index]

        # We also don't use the data_df
        for ref_df in perts_ref_points:
            ref_df.loc[pert_info_df.index] = self.data_df.loc[pert_info_df.index]

        # Untreated perturbation - check if it in start time, or at pert time
        for time in pert_info_df[self.time_column].unique():
            cloud_with_time_info_df = pert_info_df[pert_info_df[self.time_column] == time]

            if time in config.config_map['untreated_times']:
                # Control on start time, don't use any reference points
                start_time_control_points.loc[cloud_with_time_info_df.index] = \
                    self.data_df.loc[cloud_with_time_info_df.index]
                treated_reference_points.loc[cloud_with_time_info_df.index] = \
                    self.data_df.loc[cloud_with_time_info_df.index]
                start_time_control_reference_points.loc[cloud_with_time_info_df.index] = \
                    self.data_df.loc[cloud_with_time_info_df.index]
                for tumor in cloud_with_time_info_df[self.tumor_column].unique():
                    tumor_info_df = cloud_with_time_info_df[cloud_with_time_info_df[self.tumor_column] == tumor]
                    same_cloud_reference_points.loc[tumor_info_df.index, :] =\
                        self.untreated_dictionary[tumor][0].values
            else:

                # Control on pert time - set time vector as first vector,
                _, start_time_control_reference_sample, pert_time_control_reference_sample = \
                    self.p_dictionary[perturbation]

                # Set reference points that not depend on tumor
                treated_reference_points.loc[cloud_with_time_info_df.index, :] = \
                    pert_time_control_reference_sample.values
                start_time_control_reference_points.loc[cloud_with_time_info_df.index, :] = \
                    start_time_control_reference_sample.values

                # Differs for each tumor
                for tumor in cloud_with_time_info_df[self.tumor_column].unique():
                    start_time_control_sample, _ = self.untreated_dictionary[tumor]
                    tumor_info_df = cloud_with_time_info_df[cloud_with_time_info_df[self.tumor_column] == tumor]
                    start_time_control_points.loc[tumor_info_df.index, :] = start_time_control_sample.values
                    same_cloud_reference_points.loc[tumor_info_df.index, :] =\
                        self.untreated_dictionary[tumor][1].values

    def _fill_reference_points_for_treated_perturbation(self, perturbation):
        """
        Fill in treated reference points.
        :param perturbation: current perturbation name.
        """
        # Unpack reference points
        start_time_control_points, treated_reference_points, start_time_control_reference_points,\
            pert_time_control_points, pert_time_control_reference_points, same_cloud_reference_points = \
            self.reference_points[:6]
        perts_ref_points = self.reference_points[6:]

        # Get info_df for perturbation
        pert_info_df = self.info_df[self.info_df[self.perturbation_column] == perturbation]

        # Get reference points for perturbation
        treated_reference_sample, start_time_control_reference_sample, pert_time_control_reference_sample = \
            self.p_dictionary[perturbation]

        # Set reference points that not depends on tumor
        treated_reference_points.loc[pert_info_df.index, :] = treated_reference_sample.values
        start_time_control_reference_points.loc[pert_info_df.index, :] = start_time_control_reference_sample.values
        pert_time_control_reference_points.loc[pert_info_df.index, :] = pert_time_control_reference_sample.values

        # For each other perturbation, fill in the other ref points
        curr_pert_index = 0
        for p in self.info_df.perturbation.unique():
            if p == perturbation or p in config.config_map['untreated_labels']:
                continue
            curr_pert_treated = curr_pert_index
            curr_pert_control = curr_pert_index + 1
            perts_ref_points[curr_pert_treated].loc[pert_info_df.index, :] = self.p_dictionary[p][0].values
            perts_ref_points[curr_pert_control].loc[pert_info_df.index, :] = self.p_dictionary[p][2].values
            curr_pert_index += 2

        # treated perturbation (only in pert time)
        for t in pert_info_df[self.tumor_column].unique():
            # Get cloud data
            cloud_info_df = pert_info_df[pert_info_df[self.tumor_column] == t]
            start_time_control_sample, pert_time_control_sample = self.untreated_dictionary[t]

            # Set reference points that depends on tumor
            start_time_control_points.loc[cloud_info_df.index, :] = start_time_control_sample.values
            pert_time_control_points.loc[cloud_info_df.index, :] = pert_time_control_sample.values
            same_cloud_reference_points.loc[cloud_info_df.index, :] =\
                self.clouds_reference_dictionary[(perturbation, t)].values

    def update_reference_points(self):
        """
        Find the reference points to data.
        :return: tuple of 3 numpy arrays: Untreated reference samples of same tissue,
                                          Treated reference samples of same perturbation,
                                          Untreated reference samples of same perturbation.
        """
        untreated_labels = config.config_map['untreated_labels']

        # Create perturbation reference dictionary
        self._create_perturbation_reference_dictionary(untreated_labels)

        # Create untreated dictionary
        self._create_untreated_reference_point(untreated_labels)

        # Create cloud reference points dictionary
        self._create_clouds_reference_points_dictionary(untreated_labels)

        # For each cloud, set their reference points
        for perturbation in self.info_df[self.perturbation_column].unique():
            if perturbation in untreated_labels:
                self._fill_reference_points_for_untreated_perturbation(perturbation)
            else:
                self._fill_reference_points_for_treated_perturbation(perturbation)

    def _filter_data(self):
        """
        Filter out data that not in our
        """
        tumors_with_perturbation = self.info_df[self.info_df.perturbation == config.config_map['leave_out_perturbation_name']].tumor.unique()
        self.info_df = self.info_df[
            self.info_df.tumor.isin(tumors_with_perturbation)]

        self.data_df = self.data_df.loc[self.info_df.index]

    @staticmethod
    def create_empty_df_from_other(example_df, dtype):
        """
        Create new, empty DF, with same columns and indexes like given one.
        :param example_df: another df to make a copy.
        :param dtype: type of data to force, may be None
        :return: the new, empty df
        """
        return pd.DataFrame(np.zeros(example_df.shape, dtype=dtype), index=example_df.index, columns=example_df.columns)

    def _create_reference_points(self):
        """
        Create 5 empty DF for reference points.
        """
        start_time_control_points = self.create_empty_df_from_other(self.data_df, np.float64)
        treated_reference_points = self.create_empty_df_from_other(self.data_df, np.float64)
        start_time_control_reference_points = self.create_empty_df_from_other(self.data_df, np.float64)
        pert_time_control_points = self.create_empty_df_from_other(self.data_df, np.float64)
        pert_time_control_reference_points = self.create_empty_df_from_other(self.data_df, np.float64)
        same_cloud_reference_points = self.create_empty_df_from_other(self.data_df, np.float64)
        self.reference_points = [
            start_time_control_points, treated_reference_points, start_time_control_reference_points,
            pert_time_control_points, pert_time_control_reference_points, same_cloud_reference_points]
        for _ in range(self.number_of_drugs - 1):
            treated_ref = self.create_empty_df_from_other(self.data_df, np.float64)
            control_ref = self.create_empty_df_from_other(self.data_df, np.float64)
            self.reference_points.extend([treated_ref, control_ref])

    def _find_reference_tumor_for_perturbation(self):
        """
        Choose reference tumor for each perturbation.
        This will be the tumor that the reference point for that perturbation will be picked from
        """
        self.reference_tumors_dict = {}

        is_leaving_data_out = config.config_map['leave_data_out']
        leave_out_tissue_name = None
        leave_out_perturbation_name = None
        if is_leaving_data_out:
            leave_out_tissue_name = config.config_map['leave_out_tissue_name']
            leave_out_perturbation_name = config.config_map['leave_out_perturbation_name']

        for p in self.info_df[self.perturbation_column].unique():
            perturbation_info_df = self.info_df[self.info_df[self.perturbation_column] == p]
            perturbation_tumors = perturbation_info_df[self.tumor_column].unique()
            biggest_tumor_size = 0
            biggest_tumor = None
            for t in perturbation_tumors:

                # Make sure we don't choose the cloud we leave out as reference point
                if is_leaving_data_out and p == leave_out_perturbation_name and t == leave_out_tissue_name:
                    continue

                # Otherwise - just pick the biggest tumor among the perturbation.
                tumor_size = perturbation_info_df[perturbation_info_df[self.tumor_column] == t].shape[0]
                if tumor_size > biggest_tumor_size:
                    biggest_tumor = t
                    biggest_tumor_size = tumor_size
            self.reference_tumors_dict[p] = biggest_tumor
            logging.info("Chosen reference points for perturbation {0}: tissue {1}".format(p, biggest_tumor))

    def _create_losses_selectors(self):
        """
        Create 3 pandas DataFrames - one for each main loss (VAE, classifier, coliniarity).
        The values it gets is 0 or 1, this will be multiple by the ending loss
        """
        vae_selectors_df = pd.DataFrame(np.ones(self.data_df.shape[0]), index=self.data_df.index,
                                        columns=['calculate'])
        classifier_selectors_df = pd.DataFrame(np.ones(self.data_df.shape[0]), index=self.data_df.index,
                                               columns=['calculate'])
        collinearity_selectors_df = pd.DataFrame(np.ones(self.data_df.shape[0]), index=self.data_df.index,
                                                 columns=['calculate'])
        distance_selectors_df = pd.DataFrame(np.ones(self.data_df.shape[0]), index=self.data_df.index,
                                                columns=['calculate'])

        # Set the selectors
        for t in self.info_df[self.tumor_column].unique():
            tumor_info_df = self.info_df[self.info_df.tumor == t]
            for p in tumor_info_df[self.perturbation_column].unique():
                tumor_and_perturbation_info_df = tumor_info_df[tumor_info_df[self.perturbation_column] == p]
                if p in config.config_map['untreated_labels']:
                    # Set classifier selectors to ignore small clouds
                    # DMSO - Divide to start and pert time clouds
                    start_time_cloud_df = tumor_and_perturbation_info_df[
                        tumor_and_perturbation_info_df.pert_time.isin(config.config_map['untreated_times'])]
                    if start_time_cloud_df.shape[0] < config.config_map['minimum_samples_to_classify']:
                        classifier_selectors_df.loc[start_time_cloud_df.index] = 0

                    pert_time_cloud_df = tumor_and_perturbation_info_df[
                        tumor_and_perturbation_info_df.pert_time.isin(config.config_map['perturbation_times'])]
                    if pert_time_cloud_df.shape[0] < config.config_map['minimum_samples_to_classify']:
                        classifier_selectors_df.loc[pert_time_cloud_df.index] = 0

                    distance_selectors_df.loc[tumor_and_perturbation_info_df.index] = 0
                # else - perturbation - don't mind the times
                elif tumor_and_perturbation_info_df.shape[0] < config.config_map['minimum_samples_to_classify']:
                    classifier_selectors_df.loc[tumor_and_perturbation_info_df.index] = 0

            # Set collinearity and distance selectors to ignore tumors without perturbations
            if set(tumor_info_df.perturbation.unique()) <= set(config.config_map['untreated_labels']):
                collinearity_selectors_df.loc[tumor_info_df.index] = 0

        # Save the selectors
        self.selectors = [vae_selectors_df, classifier_selectors_df, collinearity_selectors_df, distance_selectors_df]

    def _load_data(self, find_reference_points):
        """
        Read the data from files
        :param find_reference_points: if set, find reference points
        """
        organized_data_folder = config.config_map['organized_data_folder']
        data_path = os.path.join(organized_data_folder, config.config_map['data_file_name'])
        info_path = os.path.join(organized_data_folder, config.config_map['information_file_name'])
        scaler_path = os.path.join(organized_data_folder, 'cmap_scaler')

        self.data_df = pd.read_hdf(data_path, 'df')

        info_types = {'inst_id': str,
                      'perturbation': str,
                      'tumor': str,
                      'pert_time': np.int32}
        self.info_df = pd.read_csv(info_path, dtype=info_types)

        if config.config_map['random_labels']:
            info_index = self.info_df.inst_id
            self.info_df = self.info_df.sample(frac=1)
            self.info_df.inst_id = info_index.values
        self.info_df.set_index('inst_id', inplace=True, drop=True)

        for t in self.info_df.tumor.unique():
            p='DMSO'
            for pert_time in self.info_df.pert_time.unique():
                cloud_df = self.info_df[(self.info_df.tumor == t) & (self.info_df.perturbation == p) &
                                        (self.info_df.pert_time == pert_time)]
                if cloud_df.shape[0] > config.config_map['max_samples_per_cloud']:
                    cloud_data_df = self.data_df.loc[cloud_df.index]
                    mean = cloud_data_df.mean(axis=0)
                    cloud_data_df['distances'] = np.linalg.norm(cloud_data_df - mean, axis=1)
                    index_to_keep = cloud_data_df['distances'].nsmallest(config.config_map['max_samples_per_cloud']).index
                    index_to_drop = cloud_df.loc[~cloud_df.index.isin(index_to_keep)].index
                    self.info_df.drop(index_to_drop, inplace=True)

        self.info_not_to_fit_df = pd.DataFrame(columns=self.info_df.columns)

        # If need, keep only perturbations that in whitelist
        if config.config_map['run_use_perturbations_whitelist']:
            perturbations_and_control = config.config_map['run_perturbations_whitelist']
            perturbations_and_control.extend(config.config_map['untreated_labels'])
            self.info_df = self.info_df[self.info_df.perturbation.isin(perturbations_and_control)]
        elif config.config_map['run_dont_fit_non_whitelisted_perturbations']:
            # If requested, split non whitelisted perturbations, which not fitting on it
            non_whitelisted_perts_df = self.info_df[
                ~self.info_df.perturbations.isin(config.config_map['run_perturbations_whitelist'])]
            self.info_not_to_fit_df = self.info_not_to_fit_df.append(non_whitelisted_perts_df)

        # If requested, keep only tumors that in whitelist
        if config.config_map['run_use_tissues_whitelist']:
            self.info_df = self.info_df[(self.info_df.tumor.isin(config.config_map['run_tissues_whitelist']))]
        elif config.config_map['run_non_whitelisted_tissues_fit_on_control']:
            # If requested, split non whitelisted tissues to control, which fitting on it, and treated
            non_whitelisted_tissues_df = self.info_df[
                (~self.info_df.tumor.isin(config.config_map['run_tissues_whitelist'])) &
                (~self.info_df.perturbation.isin(config.config_map['untreated_labels']))]
            self.info_not_to_fit_df = self.info_not_to_fit_df.append(non_whitelisted_tissues_df)

        # Drop from data_df and info_df not whitelisted perturbations
        self.info_df.drop(self.info_not_to_fit_df.index, inplace=True)
        self.data_not_to_fit_df = self.data_df.loc[self.info_not_to_fit_df.index]
        self.data_df = self.data_df.loc[self.info_df.index]

        drugs = self.info_df.perturbation.unique()
        self.number_of_drugs = len([x for x in drugs if x not in config.config_map['untreated_labels']])

        self._create_reference_points()
        self._find_reference_tumor_for_perturbation()
        self._create_losses_selectors()
        if find_reference_points:
            self.update_reference_points()

        if not mock:
            self.scaler = joblib.load(scaler_path)

    def get_reference_points_to_data_slice(self, slice_indexes, original_reference_points = None):
        """
        Create tuple of reference points to part of the data, based on their index
        :param slice_indexes: index of the slice
        :param original_reference_points: reference points to slice from, if None then slice from self.reference_points
        :return: tuple of reference points
        """
        if original_reference_points is None:
            original_reference_points = self.reference_points
        reference_points = []
        for i in range(len(original_reference_points)):
            reference_points.append(original_reference_points[i].loc[slice_indexes])
        return reference_points

    @staticmethod
    def append_reference_points(ref_points_a, ref_points_b):
        """
        Append 2 sets of reference points to one set
        :param ref_points_a: first set
        :param ref_points_b: second set
        :return: reference points set with all the points
        """
        return [ref_points_a[i].append(ref_points_b[i]) for i in range(len(ref_points_a))]

    @staticmethod
    def _round_down_to_batch_size(values):
        """
        Round down values to next batch size.
        :param values: DataFrame or numpy array to round down.
        :return: rounded down object.
        """
        length = values.shape[0]
        batch_size = config.config_map['batch_size']
        rounded_shape = np.int(length / batch_size) * batch_size
        return values[:rounded_shape]

    def _split_to_train_and_test(self):
        """
        Split the data to test and train sets.
        :param configuration: configuration of software.
        :return: the test and train indexes
        """
        # Create test set from all the samples set.
        test_indexes = self.data_df.sample(frac=config.config_map['test_set_percent']).index
        test_indexes = self._round_down_to_batch_size(test_indexes)
        self.test_data_df = self.data_df.loc[test_indexes]
        self.test_info_df = self.info_df.loc[test_indexes]

        # Create reference points to test part
        self.test_reference_points = self.get_reference_points_to_data_slice(self.test_data_df.index)

        # Create selectors to test part
        self.test_selectors = self.get_reference_points_to_data_slice(self.test_data_df.index, self.selectors)

        # We use the rest of the samples as train.
        train_indexes = self._round_down_to_batch_size(self.data_df.drop(test_indexes).index)
        self.train_data_df = self.data_df.loc[train_indexes]
        self.train_info_df = self.info_df.loc[train_indexes]

        # Create reference points to train part
        self.train_reference_points = self.get_reference_points_to_data_slice(self.train_data_df.index)

        # Create selectors to train part
        self.train_selectors = self.get_reference_points_to_data_slice(self.train_data_df.index, self.selectors)

    def update_reference_points_and_set_to_train_and_test(self):
        """
        Update reference points, and set train and test set accordingly
        """
        self.update_reference_points()

        # Now update train and test reference points, without changing the objects themselves
        new_train_reference_points = self.get_reference_points_to_data_slice(self.train_data_df.index)
        for i in range(len(new_train_reference_points)):
            self.train_reference_points[i][:] = new_train_reference_points[i].values

        new_test_reference_points = self.get_reference_points_to_data_slice(self.test_data_df.index)
        for i in range(len(new_test_reference_points)):
            self.test_reference_points[i][:] = new_test_reference_points[i].values

    def drop_from_train_set(self, indexes):
        """
        Drop samples from train set
        :param indexes: indexes of samples to drop
        """
        train_indexes = self.train_data_df.drop(indexes).index
        train_indexes = self._round_down_to_batch_size(train_indexes)
        self.train_data_df = self.data_df.loc[train_indexes]
        self.train_info_df = self.info_df.loc[train_indexes]

        # Create reference points to train part
        self.train_reference_points = self.get_reference_points_to_data_slice(self.train_data_df.index)

        # Create selectors to train part
        self.train_selectors = self.get_reference_points_to_data_slice(self.train_data_df.index, self.selectors)

    def save_reference_points(self):
        """
        Save reference points dictionaries, for further use
        """
        reference_points = [self.p_dictionary, self.untreated_dictionary, self.clouds_reference_dictionary]
        output_path = os.path.join(config.config_map['output_folder'], 'reference_points.p')
        with open(output_path, 'wb') as fp:
            pickle.dump(reference_points, fp)

    def load_reference_points(self, reference_points_path=None):
        """
        Load reference points from disk
        :param reference_points_path: path to saved reference points, by default using default saving path
        """
        if reference_points_path is None:
            saved_path = os.path.join(config.config_map['output_folder'], 'reference_points.p')
        else:
            saved_path = reference_points_path
        with open(saved_path, 'rb') as fp:
            self.p_dictionary, self.untreated_dictionary, self.clouds_reference_dictionary = pickle.load(fp)

        # Fill in dictionaries missing perturbations and tumors
        untreated_labels = config.config_map['untreated_labels']
        update_missing_only = True

        # Create perturbation reference dictionary
        self._create_perturbation_reference_dictionary(untreated_labels, update_missing_only)
        self._create_untreated_reference_point(untreated_labels, update_missing_only)
        self._create_clouds_reference_points_dictionary(untreated_labels, update_missing_only)

        # For each cloud, set their reference points
        for perturbation in self.info_df[self.perturbation_column].unique():
            if perturbation in untreated_labels:
                self._fill_reference_points_for_untreated_perturbation(perturbation)
            else:
                self._fill_reference_points_for_treated_perturbation(perturbation)
