from configuration import config

import pandas as pd
from sklearn import preprocessing
import numpy as np
import os
import shutil
import logging


class DataOrganizer:
    """
    Create and organize the data into 2 files - data.h5 and info.csv
    """

    def __init__(self):
        """
        Initializer - set all files constants
        """
        # Create output root folder if not exists
        if not os.path.isdir(config.config_map['root_output_folder']):
            os.makedirs(config.config_map['root_output_folder'])

        # Set the logger configuration
        self._set_logger()

        self.mock_configuration = {
            'samples_per_cloud': 250,
            'samples_per_small_cloud': 20,
            'column_numbers': 2,
            'std': 0.005,
            'normalize_data': True}

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name)
        self.clouds_centers = dict()
        self.clouds_centers[(0, untreated_label, 0)] = (0.1, -0.5)
        self.clouds_centers[(0, untreated_label, 24)] = (0.1, -0.3)
        self.clouds_centers[(0, 1, 24)] = (0.2, -0.3)

        self.clouds_centers[(1, untreated_label, 0)] = (0.3, -0.5)
        self.clouds_centers[(1, untreated_label, 24)] = (0.3, -0.3)
        self.clouds_centers[(1, 1, 24)] = (0.4, -0.3)

        self.clouds_centers[(2, untreated_label, 0)] = (0.7, -0.5)
        self.clouds_centers[(2, untreated_label, 24)] = (0.7, -0.3)
        self.clouds_centers[(2, 1, 24)] = (0.8, -0.3)

        self.clouds_centers[(3, untreated_label, 0)] = (0.1, 0)
        self.clouds_centers[(3, untreated_label, 24)] = (0.1, 0.2)
        self.clouds_centers[(3, 1, 24)] = (0.2, 0.2)

        self.clouds_centers[(4, untreated_label, 0)] = (0.3, 0)
        self.clouds_centers[(4, untreated_label, 24)] = (0.3, 0.2)
        self.clouds_centers[(4, 1, 24)] = (0.4, 0.2)

        self.small_clouds_centers = dict()

        self.small_clouds_centers[(5, untreated_label, 0)] = (0.5, -0.5)
        self.small_clouds_centers[(5, untreated_label, 24)] = (0.5, -0.3)
        self.small_clouds_centers[(5, 1, 24)] = (0.6, -0.3)

        self.small_clouds_centers[(6, untreated_label, 0)] = (0.7, 0)
        self.small_clouds_centers[(6, untreated_label, 24)] = (0.7, 0.2)
        self.small_clouds_centers[(6, 1, 24)] = (0.8, 0.2)

        """
        self.clouds_centers[(5, untreated_label, 0)] = (-0.1, -0.5)
        self.clouds_centers[(5, untreated_label, 24)] = (-0.1, -0.4)
        self.clouds_centers[(5, 1, 24)] = (-0.2, -0.4)

        self.clouds_centers[(6, untreated_label, 0)] = (-0.3, -0.5)
        self.clouds_centers[(6, untreated_label, 24)] = (-0.3, -0.2)
        self.clouds_centers[(6, 1, 24)] = (-0.5, -0.2)

        self.clouds_centers[(7, untreated_label, 0)] = (-0.7, -0.5)
        self.clouds_centers[(7, untreated_label, 24)] = (-0.7, -0.3)
        self.clouds_centers[(7, 1, 24)] = (-0.8, -0.3)

        self.clouds_centers[(8, untreated_label, 0)] = (-0.1, 0.0)
        self.clouds_centers[(8, untreated_label, 24)] = (-0.1, 0.1)
        self.clouds_centers[(8, 1, 24)] = (-0.2, 0.1)

        self.clouds_centers[(9, untreated_label, 0)] = (-0.3, 0.0)
        self.clouds_centers[(9, untreated_label, 24)] = (-0.3, 0.3)
        self.clouds_centers[(9, 1, 24)] = (-0.5, 0.3)"""

        self.number_of_clouds = len(self.clouds_centers)
        self.number_of_small_clouds = len(self.small_clouds_centers)
        self.total_samples_number = self.mock_configuration['samples_per_cloud'] * self.number_of_clouds + \
                                    self.mock_configuration['samples_per_small_cloud'] * self.number_of_small_clouds

    @staticmethod
    def _set_logger():
        """
        Set the logger for DataOrganizer.
        """
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
        root_logger = logging.getLogger()
        root_logger.handlers = []

        file_handler = logging.FileHandler(os.path.join(config.config_map['root_output_folder'],
                                                        'data_organizer_log.txt'))
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.DEBUG)

    def _create_random_sample_of_half_moon(self, samples_number, inner_circle_radius, center_positions, rotate_angel,
                                           std):
        """
        Create a half moon shape
        :param samples_number: number of samples in the shape
        :param inner_circle_radius: radios without samples from the center of the shape
        :param center_positions: center position of the shape
        :param rotate_angel: rotate angel (0 = D)
        :param std: standard deviation of the distribution
        :return:
        """
        samples = np.random.normal(0, std, (samples_number, self.mock_configuration['column_numbers']))

        samples[:, 1] = np.abs(samples[:, 1])
        norm = np.linalg.norm(samples, axis=-1)
        with_ir = norm + inner_circle_radius
        angle = np.arccos(samples[:, 0] / norm)
        samples[:, 0] = with_ir * np.cos(angle)
        samples[:, 1] = with_ir * np.sin(angle)

        # Create the rotation matrix
        rotation_matrix = [[np.cos(rotate_angel), -(np.sin(rotate_angel))],
                           [np.sin(rotate_angel), np.cos(rotate_angel)]]
        samples = np.dot(samples, rotation_matrix)

        # Center it around the requested location
        samples += center_positions
        return samples

    def _create_random_samples_for_tissue(self, samples_number, center_positions, std):
        """
        Create samples in circle shape
        :param samples_number: number of samples to create
        :param center_positions: the center position of the circle
        :param std: standard deviation for the normal distribuation
        :return: samples
        """
        x = np.random.normal(center_positions[0], std, (samples_number, 1))
        y = np.random.normal(center_positions[1], 2*std, (samples_number, 1))
        return np.concatenate([x, y], axis=1)

    def _create_data(self):
        """
        Create all the data
        :return: nd array of samples, tissue map and perturbations map
        """
        # Initialize samples array and the maps
        samples_np = np.zeros((self.total_samples_number, self.mock_configuration['column_numbers']),
                              dtype=np.float64)
        tissues_map = np.zeros(self.total_samples_number, dtype=int)
        perturbations_map = np.zeros(self.total_samples_number, dtype=int)
        time_map = np.zeros(self.total_samples_number, dtype=int)

        # Initialize variables to the for loop
        std = self.mock_configuration['std']
        current_sample = 0
        for key, center in self.clouds_centers.items():
            samples_np[current_sample:current_sample + self.mock_configuration['samples_per_cloud'], :] = \
                self._create_random_samples_for_tissue(self.mock_configuration['samples_per_cloud'], center, std)
            tissues_map[current_sample:current_sample + self.mock_configuration['samples_per_cloud']] = key[0]
            perturbations_map[current_sample:current_sample + self.mock_configuration['samples_per_cloud']] = key[1]
            time_map[current_sample:current_sample + self.mock_configuration['samples_per_cloud']] = key[2]

            # Update for variables
            current_sample += self.mock_configuration['samples_per_cloud']

        # Create small clouds
        for key, center in self.small_clouds_centers.items():
            samples_np[current_sample:current_sample + self.mock_configuration['samples_per_small_cloud'], :] = \
                self._create_random_samples_for_tissue(self.mock_configuration['samples_per_small_cloud'], center, std)
            tissues_map[current_sample:current_sample + self.mock_configuration['samples_per_small_cloud']] = key[0]
            perturbations_map[current_sample:current_sample + self.mock_configuration['samples_per_small_cloud']] = key[1]
            time_map[current_sample:current_sample + self.mock_configuration['samples_per_small_cloud']] = key[2]

            # Update for variables
            current_sample += self.mock_configuration['samples_per_small_cloud']

        # Create the DataFrames
        self.data_df = pd.DataFrame(samples_np,
                                    index=[str(x) for x in range(self.total_samples_number)],
                                    columns=range(self.mock_configuration['column_numbers']))

        self.info_df = pd.DataFrame({'tumor': tissues_map,
                                     'perturbation': perturbations_map,
                                     'pert_time': time_map},
                                    index=self.data_df.index)

    def _add_columns(self):
        """
        Add needed columns to data
        """
        self.info_df['perturbation'] = self.info_df['perturbation'].apply(str)
        self.info_df['tumor'] = self.info_df['tumor'].apply(str)
        self.info_df.loc[self.info_df['perturbation'] == '0', 'perturbation'] = 'DMSO'

        self.info_df['classifier_labels'] = self.info_df['tumor'] + ' ' + self.info_df["perturbation"] + ' ' + \
                                            self.info_df["pert_time"].map(str)

        labels, _ = pd.factorize(self.info_df["classifier_labels"])
        self.info_df["numeric_labels"] = labels

        self.info_df['inst_id'] = self.info_df.index

        # Remove all the data that not in the selected samples, after all the filtering
        self.data_df = self.data_df.loc[self.info_df["inst_id"]]

    def organize_data(self):
        """
        Read and process all the data, and then save it to output files
        """
        # Create the data
        self._create_data()

        # Add columns
        self._add_columns()

        # Data normalization.
        if self.mock_configuration['normalize_data']:
            data_np = preprocessing.MinMaxScaler().fit_transform(self.data_df)
            self.data_df = pd.DataFrame(data_np, index=self.data_df.index, columns=self.data_df.columns)
            #self.data_df = (self.data_df - self.data_df.mean()) / (self.data_df.max() - self.data_df.min())

        # Save the data and the information in 2 organized files
        organized_data_folder = config.config_map['organized_data_folder']

        # Delete old folder
        if os.path.isdir(organized_data_folder):
            shutil.rmtree(organized_data_folder)

        # Create the folder
        os.makedirs(organized_data_folder)

        # Save the data
        data_path = os.path.join(organized_data_folder, config.config_map['data_file_name'])
        info_path = os.path.join(organized_data_folder, config.config_map['information_file_name'])
        self.data_df.to_hdf(data_path, key='df')
        self.info_df.to_csv(info_path, sep=',', index=False, columns=['inst_id',
                                                                      'perturbation',
                                                                      'tumor',
                                                                      'classifier_labels',
                                                                      'numeric_labels',
                                                                      'pert_time'])


if __name__ == '__main__':
    organizer = DataOrganizer()
    organizer.organize_data()

