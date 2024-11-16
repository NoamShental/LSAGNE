""""
DATE: 9.12.19
FILE DESCRIPTION:
    This module is used to create mock data. Mock data is required while system developing,
    to create 2D mock data that is easy to debug. In addition to the mock section in the configuration file,
    it has its own configuration variables (setting data properties).
    Its output files are in the same format as the CMap organized data,
    thus the running stage is transparent to type of the data under the hood. When mock organizer is initialized,
    it sets the clouds centers for each class (class means tumor over perturbation over time).
    Later on, by calling organize_data() it creates clouds of samples (using normal distribution around the center), and their info.
	We also use this module to create 3D data, to produce mock figures for our paper explanations.
"""
import os
import logging
import random
import pandas as pd
import numpy as np

from src.configuration import config
from src.os_utilities import delete_dir_if_exists, create_dir_if_not_exists


class DataOrganizer:
    """
    Create and organize the data into two files: data.h5 and info.csv.
    """
    def __init__(self):
        """
        Initializer - set all files constants.
        """
        delete_dir_if_exists(config.organized_cmap_folder)
        create_dir_if_not_exists(config.organized_cmap_folder)

        # Set the logger configuration.
        self._set_logger()

        self.mock_configuration = {     # Example of values for 2D:
            'samples_per_cloud': 250,  # 250
            # The column nmbers is the number of dimensions in the real-space.
            'column_numbers': 3,        # 2
            'std': 0.35,                # 0.1 ; 0.25
            'normalize_data': True      #
        }

        # self.mock_configuration = {     # Example of values for 2D:
        #     'samples_per_cloud': 250,  # 250
        #     # The column nmbers is the number of dimensions in the real-space.
        #     'column_numbers': 2,        # 2
        #     'std': 0.1,                # 0.1 ; 0.25
        #     'normalize_data': True      #
        # }

        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers = dict()
        if self.mock_configuration['column_numbers'] == 2:
            #self._create_1PerturbationX25Tissues()
            # self._create_4PerturbationX6Tissues()
            self._create_2PerturbationX3Tissues()
            #self._create_7PerturbationX6Tissues_with_random_centr()
        elif self.mock_configuration['column_numbers'] == 3:
            #self._create_4PerturbationX6Tissues_of_3_dim()
            self._create_1PerturbationX4Tissues_of_3_dim()
            # self._create_1PerturbationX3Tissues_of_3_dim()

        self.number_of_clouds = len(self.clouds_centers)
        self.total_samples_number = self.mock_configuration['samples_per_cloud'] * self.number_of_clouds


    def _create_1PerturbationX25Tissues(self):

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers[(0, untreated_label, 24)] = (-5, -5)
        self.clouds_centers[(0, 1, 24)] = (-3, -3)

        self.clouds_centers[(1, untreated_label, 24)] = (0, -5)
        self.clouds_centers[(1, 1, 24)] = (2, -3)

        self.clouds_centers[(2, untreated_label, 24)] = (5, -5)
        self.clouds_centers[(2, 1, 24)] = (7, -3)

        self.clouds_centers[(3, untreated_label, 24)] = (-5, 0)
        self.clouds_centers[(3, 1, 24)] = (-3, 2)

        self.clouds_centers[(4, untreated_label, 24)] = (0, 0)
        self.clouds_centers[(4, 1, 24)] = (2, 2)

        self.clouds_centers[(5, untreated_label, 24)] = (-10, 4)
        self.clouds_centers[(5, 1, 24)] = (-8, 6)

        self.clouds_centers[(6, untreated_label, 24)] = (-6, 4)
        self.clouds_centers[(6, 1, 24)] = (-4, 6)

        self.clouds_centers[(7, untreated_label, 24)] = (0, 4)
        self.clouds_centers[(7, 1, 24)] = (2, 6)

        self.clouds_centers[(8, untreated_label, 24)] = (5, 4)
        self.clouds_centers[(8, 1, 24)] = (7, 6)

        """
        self.clouds_centers[(9, untreated_label, 24)] = (9, 4)
        self.clouds_centers[(9, 1, 24)] = (11, 6)

        self.clouds_centers[(10, untreated_label, 24)] = (9, 0)
        self.clouds_centers[(10, 1, 24)] = (11, 2)

        self.clouds_centers[(11, untreated_label, 24)] = (9, -3)
        self.clouds_centers[(11, 1, 24)] = (11, -1)

        self.clouds_centers[(12, untreated_label, 24)] = (9, -6)
        self.clouds_centers[(12, 1, 24)] = (11, -4)

        self.clouds_centers[(13, untreated_label, 24)] = (9, -9)
        self.clouds_centers[(13, 1, 24)] = (11, -7)

        self.clouds_centers[(14, untreated_label, 24)] = (5, -9)
        self.clouds_centers[(14, 1, 24)] = (7, -7)

        self.clouds_centers[(15, untreated_label, 24)] = (0, -9)
        self.clouds_centers[(15, 1, 24)] = (2, -7)

        self.clouds_centers[(16, untreated_label, 24)] = (-5, -9)
        self.clouds_centers[(16, 1, 24)] = (-3, -7)

        self.clouds_centers[(17, untreated_label, 24)] = (-10, -9)
        self.clouds_centers[(17, 1, 24)] = (-8, -7)

        self.clouds_centers[(18, untreated_label, 24)] = (-10, -7)
        self.clouds_centers[(18, 1, 24)] = (-8, -5)

        self.clouds_centers[(19, untreated_label, 24)] = (-10, -4)
        self.clouds_centers[(19, 1, 24)] = (-8, -2)

        self.clouds_centers[(20, untreated_label, 24)] = (-10, 0)
        self.clouds_centers[(20, 1, 24)] = (-8, 2)

        self.clouds_centers[(21, untreated_label, 24)] = (-10, -15)
        self.clouds_centers[(21, 1, 24)] = (-8, -13)

        self.clouds_centers[(22, untreated_label, 24)] = (-5, -15)
        self.clouds_centers[(22, 1, 24)] = (-3, -13)

        self.clouds_centers[(23, untreated_label, 24)] = (0, -15)
        self.clouds_centers[(23, 1, 24)] = (2, -13)

        self.clouds_centers[(24, untreated_label, 24)] = (5, -15)
        self.clouds_centers[(24, 1, 24)] = (7, -13)
        """


    def _create_1PerturbationX25Tissues_of_10_dim(self):

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers[(0, untreated_label, 24)] = (-5, -5, 1, 4 , 6, 7, -8, 10, -12, 0)
        self.clouds_centers[(0, 1, 24)]              = (-3, -3, 1, 4 , 6, 7, -8, 10, -12, 0)

        self.clouds_centers[(1, untreated_label, 24)] = (0, -5, 1, -4 , 6, 7, -8, 10, -12, 7)
        self.clouds_centers[(1, 1, 24)]              = (2, -3, 1, -4 , 6, 7, -8, 10, -12, 7)

        self.clouds_centers[(2, untreated_label, 24)] = (5, -5, 10, -4 , 6, -7, -8, 10, 2, 7)
        self.clouds_centers[(2, 1, 24)]              = (7, -3, 10, -4 , 6, -7, -8, 10, 2, 7)

        self.clouds_centers[(3, untreated_label, 24)] = (-5, 0, 1, -14 , 6, -7, -8, -3, 2, 7)
        self.clouds_centers[(3, 1, 24)]              = (-3, 2, 1, -14 , 6, -7, -8, -3, 2, 7)

        self.clouds_centers[(4, untreated_label, 24)] = (0, 0, 7, -14 , 6, -7, 0, -3, 4, 7)
        self.clouds_centers[(4, 1, 24)]              = (2, 2, 7, -14 , 6, -7, 0, -3, 4, 7)

        self.clouds_centers[(5, untreated_label, 24)] = (-10, 4, 7, 3 , -6, -7, 0, 2, 14, 1)
        self.clouds_centers[(5, 1, 24)]              = (-8, 6, 7, 3 , -6, -7, 0, 2, 14, 1)

        self.clouds_centers[(6, untreated_label, 24)] = (-6, 4, 2, 3 , 3, 0, 0, 2, -14, 1)
        self.clouds_centers[(6, 1, 24)]              = (-4, 6, 2, 3 , 3, 0, 0, 2, -14, 1)

        self.clouds_centers[(7, untreated_label, 24)] = (0, 4, 2, 1 , 3, 0, -8, 2, 3, 1)
        self.clouds_centers[(7, 1, 24)]              = (2, 6, 2, 1 , 3, 0, -8, 2, 3, 1)

        self.clouds_centers[(8, untreated_label, 24)] = (5, 4, 2, 2 , -6, 0, -8, -4, 3, 9)
        self.clouds_centers[(8, 1, 24)]              = (7, 6, 2, 2 , -6, 0, -8, -4, 3, 9)

        self.clouds_centers[(9, untreated_label, 24)] = (9, 4, 2, 0 , -6, 1, -8, -4, 0, 9)
        self.clouds_centers[(9, 1, 24)]              = (11, 6, 2, 0 , -6, 1, -8, -4, 0, 9)

        self.clouds_centers[(10, untreated_label, 24)] = (9, 0, 2, 0 , 3, 1, -5, 7, 0, 9)
        self.clouds_centers[(10, 1, 24)]              = (11, 2, 2, 0 , 3, 1, -5, 7, 0, 9)

        self.clouds_centers[(11, untreated_label, 24)] = (9, -3, 2, 0 , -3, 1, 2, 7, 0, 4)
        self.clouds_centers[(11, 1, 24)]              = (11, -1, 2, 0 , -3, 1, 2, 7, 0, 4)

        self.clouds_centers[(12, untreated_label, 24)] = (9, -6, 2, 8 , -3, 1, 12, 7, 0, -1)
        self.clouds_centers[(12, 1, 24)]              = (11, -4, 2, 8 , -3, 1, 12, 7, 0, -1)

        self.clouds_centers[(13, untreated_label, 24)] = (9, -9, 12, 8 , -3, 4, 2, -7, 0, -1)
        self.clouds_centers[(13, 1, 24)]              = (11, -7, 12, 8 , -3, 4, 2, -7, 0, -1)

        self.clouds_centers[(14, untreated_label, 24)] = (5, -9, 12, 2 , -3, 7, -11, -7, 0, -1)
        self.clouds_centers[(14, 1, 24)]              = (7, -7, 12, 2 , -3, 7, -11, -7, 0, -1)

        self.clouds_centers[(15, untreated_label, 24)] = (0, -9, 0, 0 , -3, -7, 11, -7, 0, 13)
        self.clouds_centers[(15, 1, 24)]              = (2, -7, 0, 0 , -3, -7, 11, -7, 0, 13)

        self.clouds_centers[(16, untreated_label, 24)] = (-5, -9, 10, 1 , -3, -7, 11, 2, 0, 13)
        self.clouds_centers[(16, 1, 24)]              = (-3, -7, 10, 1 , -3, -7, 11, 2, 0, 13)

        self.clouds_centers[(17, untreated_label, 24)] = (-10, -9, 4, 1 , -3, 2, 11, 0, 0, 13)
        self.clouds_centers[(17, 1, 24)]              = (-8, -7, 4, 1 , -3, 2, 11, 0, 0, 13)

        self.clouds_centers[(18, untreated_label, 24)] = (-10, -7, 4, 10 , 3, -2, -11, 0, 0, 13)
        self.clouds_centers[(18, 1, 24)]              = (-8, -5, 4, 10 , 3, -2, -11, 0, 0, 13)

        self.clouds_centers[(19, untreated_label, 24)] = (-10, -4, 5, 10 , -2, -2, -11, -5, 0, 13)
        self.clouds_centers[(19, 1, 24)]              = (-8, -2, 5, 10 , -2, -2, -11, -5, 0, 13)

        self.clouds_centers[(20, untreated_label, 24)] = (-10, 0, 4, 10 , 0, -2, -11, 5, 0, 13)
        self.clouds_centers[(20, 1, 24)]              = (-8, 2, 4, 10 , 0, -2, -11, 5, 0, 13)

        self.clouds_centers[(21, untreated_label, 24)] = (-10, -1, 4, 13 , 0, 6, -11, 5, 8, 13)
        self.clouds_centers[(21, 1, 24)]              = (-8, -13, 4, 13 , 0, 6, -11, 5, 8, 13)

        self.clouds_centers[(22, untreated_label, 24)] = (-5, -15, 4, 1 , 0, 6, 1, 5, -8, 13)
        self.clouds_centers[(22, 1, 24)]              = (-3, -13, 4, 1 , 0, 6, 1, 5, -8, 13)

        self.clouds_centers[(23, untreated_label, 24)] = (0, -15, -4, 1 , 0, -6, 1, 5, -8, 13)
        self.clouds_centers[(23, 1, 24)]              = (2, -13, -4, 1 , 0, -6, 1, 5, -8, 13)

        self.clouds_centers[(24, untreated_label, 24)] = (5, -15, 3, 1 , 0, -6, 1, 2, -6, 13)
        self.clouds_centers[(24, 1, 24)]              = (7, -13, 3, 1 , 0, -6, 1, 2, -6, 13)


    def _create_2PerturbationX3Tissues(self):

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers[(0, untreated_label, 24)] = (-10, 4)
        self.clouds_centers[(0, 1, 24)]               = (-8, 6)
        self.clouds_centers[(0, 2, 24)]               = (-12, 7)

        self.clouds_centers[(2, untreated_label, 24)] = (9, 4)
        self.clouds_centers[(2, 1, 24)]               = (11, 6)
        self.clouds_centers[(2, 2, 24)]               = (7, 7)

        self.clouds_centers[(1, untreated_label, 24)] = (-10, 0)
        self.clouds_centers[(1, 1, 24)]               = (-8, 2)
        self.clouds_centers[(1, 2, 24)]               = (-12, 3)

    def _create_4PerturbationX6Tissues(self):

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers[(0, untreated_label, 24)] = (-10, 4)
        self.clouds_centers[(0, 1, 24)]              = (-8, 6)
        self.clouds_centers[(0, 2, 24)]              = (-12, 7)
        self.clouds_centers[(0, 3, 24)]              = (-12, 5)
        self.clouds_centers[(0, 4, 24)]              = (-10, 2)

        self.clouds_centers[(1, untreated_label, 24)] = (0, 4)
        self.clouds_centers[(1, 1, 24)]              = (2, 6)
        self.clouds_centers[(1, 2, 24)]              = (-2, 7)
        self.clouds_centers[(1, 3, 24)]              = (-2, 5)
        self.clouds_centers[(1, 4, 24)]              = (0, 2)

        self.clouds_centers[(2, untreated_label, 24)] = (9, 4)
        self.clouds_centers[(2, 1, 24)]              = (11, 6)
        self.clouds_centers[(2, 2, 24)]              = (7, 7)
        self.clouds_centers[(2, 3, 24)]              = (7, 5)
        self.clouds_centers[(2, 4, 24)]              = (9, 2)

        self.clouds_centers[(3, untreated_label, 24)] = (-10, 0)
        self.clouds_centers[(3, 1, 24)]              = (-8, 2)
        self.clouds_centers[(3, 2, 24)]              = (-12, 3)
        self.clouds_centers[(3, 3, 24)]              = (-12, 1)
        self.clouds_centers[(3, 4, 24)]              = (-10, -2)

        self.clouds_centers[(4, untreated_label, 24)] = (0, 0)
        self.clouds_centers[(4, 1, 24)]              = (2, 2)
        self.clouds_centers[(4, 2, 24)]              = (-2, 3)
        self.clouds_centers[(4, 3, 24)]              = (-2, 1)
        self.clouds_centers[(4, 4, 24)]              = (0, -2)

        self.clouds_centers[(5, untreated_label, 24)] = (9, 0)
        self.clouds_centers[(5, 1, 24)]              = (11, 2)
        self.clouds_centers[(5, 2, 24)]              = (7, 3)
        self.clouds_centers[(5, 3, 24)]              = (7, 1)
        self.clouds_centers[(5, 4, 24)]              = (9, -2)


    def _create_4PerturbationX6Tissues_of_3_dim(self):

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers[(0, untreated_label, 24)] = (-10, 4, 3)
        self.clouds_centers[(0, 1, 24)] = (-8, 6, 5)   # Action: +2, +2, +2
        self.clouds_centers[(0, 2, 24)] = (-12, 7, 2)  # Action: -2, +3, -1
        #self.clouds_centers[(0, 3, 24)] = (-12, 5, 3)  # Action: -2, +1,  0
        #self.clouds_centers[(0, 4, 24)] = (-10, 2, 0)  # Action: 0, -2,  -3
        """
        self.clouds_centers[(1, untreated_label, 24)] = (0, 4, 3)
        self.clouds_centers[(1, 1, 24)] = (2, 6, 5)
        self.clouds_centers[(1, 2, 24)] = (-2, 7, 2)
        self.clouds_centers[(1, 3, 24)] = (-2, 5, 3)
        self.clouds_centers[(1, 4, 24)] = (0, 2, 0)
        """
        self.clouds_centers[(2, untreated_label, 24)] = (9, 4, 3)
        self.clouds_centers[(2, 1, 24)] = (11, 6, 5)
        self.clouds_centers[(2, 2, 24)] = (7, 7, 2)
        #self.clouds_centers[(2, 3, 24)] = (7, 5, 3)
        #self.clouds_centers[(2, 4, 24)] = (9, 2, 0)
        """
        self.clouds_centers[(3, untreated_label, 24)] = (-10, 0, -2)
        self.clouds_centers[(3, 1, 24)] = (-8, 2, 0)
        self.clouds_centers[(3, 2, 24)] = (-12, 3, -3)
        self.clouds_centers[(3, 3, 24)] = (-12, 1, -2)
        self.clouds_centers[(3, 4, 24)] = (-10, -2, -5)
        """
        self.clouds_centers[(4, untreated_label, 24)] = (0, 0, -2)
        self.clouds_centers[(4, 1, 24)] = (2, 2, 0)
        self.clouds_centers[(4, 2, 24)] = (-2, 3, -3)
        #self.clouds_centers[(4, 3, 24)] = (-2, 1, -2)
        #self.clouds_centers[(4, 4, 24)] = (0, -2, -5)
        """
        self.clouds_centers[(5, untreated_label, 24)] = (9, 0, -2)
        self.clouds_centers[(5, 1, 24)] = (11, 2, 0)
        self.clouds_centers[(5, 2, 24)] = (7, 3, -3)
        self.clouds_centers[(5, 3, 24)] = (7, 1, -2)
        self.clouds_centers[(5, 4, 24)] = (9, -2, -5)
        """


    def _create_1PerturbationX3Tissues_of_3_dim(self):

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers[(0, untreated_label, 6)] = (-10, 4, 3)
        self.clouds_centers[(0, 1, 24)] = (-4, 10, 7)  # Action: +2, +2, +2
        #self.clouds_centers[(0, 1, 24)] = (-8, 6, 5)  # Action: +2, +2, +2
        #self.clouds_centers[(0, 2, 24)] = (-12, 7, 2)  # Action: -2, +3, -1
        # self.clouds_centers[(0, 3, 24)] = (-12, 5, 3)  # Action: -2, +1,  0
        # self.clouds_centers[(0, 4, 24)] = (-10, 2, 0)  # Action: 0, -2,  -3

        self.clouds_centers[(1, untreated_label, 6)] = (20, -18, 3)
        self.clouds_centers[(1, 1, 24)] = (20, -18, 7)  # Action: +0, +4, +4
        # self.clouds_centers[(1, 1, 24)] = (2, 6, 5)
        # self.clouds_centers[(1, 2, 24)] = (-2, 7, 2)
        # self.clouds_centers[(1, 3, 24)] = (-2, 5, 3)
        # self.clouds_centers[(1, 4, 24)] = (0, 2, 0)

        self.clouds_centers[(2, untreated_label, 6)] = (9, 4, 3)
        self.clouds_centers[(2, 1, 24)] = (20, 4, 3)  # Action: +8, +0, +10
        # self.clouds_centers[(2, 1, 24)] = (11, 6, 5)
        # self.clouds_centers[(2, 2, 24)] = (7, 7, 2)
        # self.clouds_centers[(2, 3, 24)] = (7, 5, 3)
        # self.clouds_centers[(2, 4, 24)] = (9, 2, 0)


    def _create_1PerturbationX4Tissues_of_3_dim(self):

        untreated_label = 0
        # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
        self.clouds_centers[(0, untreated_label, 6)] = (-10, 4, 3)
        self.clouds_centers[(0, 1, 24)] = (-4, 10, 7)  # Action: +2, +2, +2
        #self.clouds_centers[(0, 1, 24)] = (-8, 6, 5)  # Action: +2, +2, +2
        #self.clouds_centers[(0, 2, 24)] = (-12, 7, 2)  # Action: -2, +3, -1
        # self.clouds_centers[(0, 3, 24)] = (-12, 5, 3)  # Action: -2, +1,  0
        # self.clouds_centers[(0, 4, 24)] = (-10, 2, 0)  # Action: 0, -2,  -3

        self.clouds_centers[(1, untreated_label, 6)] = (20, -18, 3)
        self.clouds_centers[(1, 1, 24)] = (20, -18, 7) # Action: +0, +4, +4
        #self.clouds_centers[(1, 1, 24)] = (2, 6, 5)
        #self.clouds_centers[(1, 2, 24)] = (-2, 7, 2)
        #self.clouds_centers[(1, 3, 24)] = (-2, 5, 3)
        #self.clouds_centers[(1, 4, 24)] = (0, 2, 0)

        self.clouds_centers[(2, untreated_label, 6)] = (9, 4, 3)
        self.clouds_centers[(2, 1, 24)] = (20, 4, 3) # Action: +8, +0, +10
        #self.clouds_centers[(2, 1, 24)] = (11, 6, 5)
        #self.clouds_centers[(2, 2, 24)] = (7, 7, 2)
        # self.clouds_centers[(2, 3, 24)] = (7, 5, 3)
        # self.clouds_centers[(2, 4, 24)] = (9, 2, 0)

        self.clouds_centers[(3, untreated_label, 6)] = (-10, 0, -2)
        self.clouds_centers[(3, 1, 24)] = (-10, 0, -12)# Action: +0, +0, +10
        #self.clouds_centers[(3, 1, 24)] = (-8, 2, 0)
        #self.clouds_centers[(3, 2, 24)] = (-12, 3, -3)
        #self.clouds_centers[(3, 3, 24)] = (-12, 1, -2)
        #self.clouds_centers[(3, 4, 24)] = (-10, -2, -5)


    def _create_7PerturbationX6Tissues_with_random_centr(self):

        untreated_label = 0

        for curr_tissue in range(6):
            x = random.randint(-15, 16)
            y = random.randint(-15, 16)
            # Dictionary of clouds centers, each key in format (tumor_name, perturbation_name, time_name).
            self.clouds_centers[(curr_tissue, untreated_label, 24)] = (x, y)
            self.clouds_centers[(curr_tissue, 1, 24)] = (x+2, y+2)
            self.clouds_centers[(curr_tissue, 2, 24)] = (x-2, y+3)
            self.clouds_centers[(curr_tissue, 3, 24)] = (x-2, y+1)
            self.clouds_centers[(curr_tissue, 4, 24)] = (x,   y-2)
            self.clouds_centers[(curr_tissue, 5, 24)] = (x, y + 5)
            self.clouds_centers[(curr_tissue, 6, 24)] = (x - 7, y)
            self.clouds_centers[(curr_tissue, 7, 24)] = (x + 7, y)


    @staticmethod
    def _set_logger():
        """
        Set the logger for DataOrganizer.
        """
        log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
        root_logger = logging.getLogger()
        root_logger.handlers = []

        print(config.organized_cmap_folder)
        file_handler = logging.FileHandler(os.path.join(config.organized_cmap_folder, 'data_organizer_log.txt'))
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
        if self.mock_configuration['column_numbers'] == 2:
            x = np.random.normal(center_positions[0], std, (samples_number, 1))
            y = np.random.normal(center_positions[1], 2 * std, (samples_number, 1))
            return np.concatenate([x, y], axis=1)
        elif self.mock_configuration['column_numbers'] == 3:
            x = np.random.normal(center_positions[0], std, (samples_number, 1))
            y = np.random.normal(center_positions[1], std, (samples_number, 1))
            z = np.random.normal(center_positions[2], std, (samples_number, 1))
            return np.concatenate([x, y, z], axis=1)


    def _create_data(self):
        """
        Create the mock data.
        :return: nd array of samples, tissue map and perturbations map.
        """
        # Initialize samples array and the maps
        samples_np        = np.zeros((self.total_samples_number,
                                      self.mock_configuration['column_numbers']), dtype=np.float64)
        tissues_map       = np.zeros(self.total_samples_number, dtype=int)
        perturbations_map = np.zeros(self.total_samples_number, dtype=int)
        time_map          = np.zeros(self.total_samples_number, dtype=int)

        std = self.mock_configuration['std']
        current_sample = 0
        for key, center in self.clouds_centers.items():
            # Create random points for the current cloud.
            samples_np[current_sample:current_sample + self.mock_configuration['samples_per_cloud'], :] = \
                self._create_random_samples_for_tissue(self.mock_configuration['samples_per_cloud'], center, std)
            # Set the tissue-map, perturbation-map and time-map, for the current cloud.
            tissues_map[current_sample:current_sample + self.mock_configuration['samples_per_cloud']] = key[0]
            perturbations_map[current_sample:current_sample + self.mock_configuration['samples_per_cloud']] = key[1]
            time_map[current_sample:current_sample + self.mock_configuration['samples_per_cloud']] = key[2]

            # Go to next cloud.
            current_sample += self.mock_configuration['samples_per_cloud']

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
        Add needed columns to data info.
        """
        self.info_df['perturbation'] = self.info_df['perturbation'].apply(str)
        self.info_df['tumor'] = self.info_df['tumor'].apply(str)
        self.info_df.loc[self.info_df['perturbation'] == '0', 'perturbation'] = 'DMSO'

        self.info_df['classifier_labels'] = self.info_df['tumor'] + ' ' + self.info_df["perturbation"] + ' ' + \
                                            self.info_df["pert_time"].map(str)


        labels, _ = pd.factorize(self.info_df["classifier_labels"])
        self.info_df["numeric_labels"] = labels

        self.info_df['inst_id'] = self.info_df.index

        # Remove all the data that not in the selected samples, after all the filtering.
        self.data_df = self.data_df.loc[self.info_df["inst_id"]]


    def _print_samples_statistics(self):
        """
        Print statistics for each tumor.
        """
        if self.info_df.shape[0] == 0:
            logging.info("After DataOrganizer actions, no data is left!! Check your filtering configuration...")
            os._exit(1)

        clouds_num = 0
        total_samples_in_start_time = 0
        total_samples_in_pert_time = 0
        for tumor in self.info_df['tumor'].unique():
            tumor_samples_df = self.info_df[self.info_df['tumor'] == tumor]
            for perturbation in tumor_samples_df['perturbation'].unique():
                perturbation_samples_df = tumor_samples_df[tumor_samples_df['perturbation'] == perturbation]
                all_perturbation_samples = perturbation_samples_df.shape[0]
                start_time_samples_df = perturbation_samples_df[
                    perturbation_samples_df['pert_time'].isin(config.untreated_times)]
                samples_in_start_time = start_time_samples_df.shape[0]
                start_time_list = start_time_samples_df.pert_time.unique()
                pert_time_samples_df = perturbation_samples_df[
                    ~perturbation_samples_df['pert_time'].isin(config.untreated_times)]
                samples_in_pert_time = pert_time_samples_df.shape[0]
                pert_time_list = pert_time_samples_df.pert_time.unique()

                logging.info('Tumor %s perturbation %s have: %d samples in start time, %d samples in'
                             ' pertubrations time, %d samples at all',
                             tumor, perturbation, samples_in_start_time, samples_in_pert_time, all_perturbation_samples)
                string = "Start time list={}, Pert time list={}."
                string_revised = string.rjust(len(string) + 70)
                logging.info(string_revised.format(start_time_list, pert_time_list))
                clouds_num += 1
                total_samples_in_start_time += samples_in_start_time
                total_samples_in_pert_time += samples_in_pert_time

        logging.info("Total number of start time samples={}".format(total_samples_in_start_time))
        logging.info("Total number of pert time samples={}".format(total_samples_in_pert_time))
        logging.info("Total number of loaded data samples={}".format(self.data_df.shape[0]))
        logging.info("Total number of loaded perturbations={}:".format(len((self.info_df['perturbation'].unique()))))
        logging.info("{}".format(list(self.info_df['perturbation'].unique())))
        logging.info("Total Number of loaded tissues={}:".format(len((self.info_df['tumor'].unique()))))
        logging.info("{}".format(list(self.info_df['tumor'].unique())))
        logging.info("Total number of clouds={}:".format(clouds_num))


    def organize_data(self):
        """
        Read and process all the data, and then save it to output files.
        """
        # Create the data.
        self._create_data()

        # Add columns.
        self._add_columns()

        # Data normalization.
        if self.mock_configuration['normalize_data']:
            self.data_df = (self.data_df - self.data_df.mean()) / (self.data_df.max() - self.data_df.min())

        # Print samples statistics.
        self._print_samples_statistics()

        # Save the data and the information in two organized files.
        data_path = os.path.join(config.organized_cmap_folder, config.data_file_name)
        info_path = os.path.join(config.organized_cmap_folder, config.information_file_name)
        self.data_df.to_hdf(data_path, key='df')
        self.info_df.to_csv(info_path, sep=',', index=False, columns=['inst_id',
                                                                      'perturbation',
                                                                      'tumor',
                                                                      'classifier_labels',
                                                                      'numeric_labels',
                                                                      'pert_time'])

        # Create the configuration file for the sbatch_runner running on the HPC.
        self._create_unique_clouds_file()

        logging.info('Version {} - Mock {}D data organization successfully done!!'.format(
            config.version, self.mock_configuration['column_numbers']))


    def _create_unique_clouds_file(self):
        """
        Create the configuration file for the sbatch_runner running on the HPC.
        """
        unique_clouds_file_name = config.unique_clouds_file_name
        unique_clouds_path = os.path.join(config.organized_cmap_folder, unique_clouds_file_name)
        unique_clouds_df = pd.concat([self.info_df['perturbation'], self.info_df['tumor']], axis=1).drop_duplicates()
        unique_clouds_df = unique_clouds_df[~unique_clouds_df.perturbation.isin(config.untreated_labels)]
        unique_clouds_df.index = range(len(unique_clouds_df.index))
        unique_clouds_df.to_csv(unique_clouds_path, sep=',', columns=['perturbation', 'tumor'])


if __name__ == '__main__':
    organizer = DataOrganizer()
    organizer.organize_data()