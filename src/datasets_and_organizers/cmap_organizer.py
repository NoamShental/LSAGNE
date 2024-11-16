"""
DATE: 9.12.19
FILE DESCRIPTION:
    This module is responsible for convert the CMap  raw data, to easily read format, loaded by the training stage.
    When CMap organizer is initialized, it sets the output directory, the logger and the required input files names.
    Later on, by calling  organize_data()  the raw CMap information is loaded, and the following filtering will be done:
        1.	Removing samples with no cell-line info.
        2.	Removing samples of tumors and drugs didn't configured in white-list.
        3.	Removing samples that can't be calculated:
                -	Not enough control samples.
                -	Not enough treated samples per drug per cell-line.
                -	No time measurements.
	Lastly, we read the data itself. This stage consumes a lot of RAM, so irrelevant data is filtered out.
	At this stage we also normalize the filtered data to be between 0 to 1.
	The organized info and data are dumped to .csv and .hdf files respectively.
"""
import os
from dataclasses import dataclass, field
from logging import Logger
from os import PathLike
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn import preprocessing
import joblib

from src.logger_utils import create_logger, add_file_handler
from src.os_utilities import create_dir_if_not_exists, delete_dir_if_exists
from src.configuration import config


CMAP_ORGANIZER_LOG_FILE_NAME = 'cmap_organizer_log.txt'


@dataclass
class CmapOrganizer:
    """
    Organize the data from raw CMap files into two files: data.h5 and info.csv.
    """
    raw_cmap_folder: str | PathLike
    organized_cmap_folder: str | PathLike
    perturbations_whitelist: list[str] | None
    untreated_labels: list[str]
    untreated_times: list[int]
    tissues_whitelist: list[str] | None
    perturbation_times: list[int]
    min_treat_conc: int
    untreated_labels_times: list[int]
    data_file_name: str
    information_file_name: str
    min_samples_per_cloud: int
    unique_clouds_file_name: str
    logger: Logger = None


    def __post_init__(self):
        delete_dir_if_exists(self.organized_cmap_folder)
        create_dir_if_not_exists(self.organized_cmap_folder)

        if not self.logger:
            self.logger = create_logger('cmap_organizer')
            add_file_handler(self.logger, Path(self.organized_cmap_folder) / CMAP_ORGANIZER_LOG_FILE_NAME)

        # This file has only CMap genes (minus one) - 977, and not all the genes.
        self.cmap_samples_data = os.path.join(self.raw_cmap_folder, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x977.h5')

        # This file contain the perturbation that each sample got.
        self.perturbation_to_samples_filename = os.path.join(self.raw_cmap_folder, 'GSE92742_Broad_LINCS_inst_info.txt.gz')

        # This file contain metadata about the cell lines.
        self.cell_lines_info_filename = os.path.join(self.raw_cmap_folder, 'GSE92742_Broad_LINCS_cell_info.txt')


    @staticmethod
    def _filter_by_series(df_to_filter, intersecting_series, column_name, negative=False):
        """
        Filter DataFrame by series - keep only the rows where their value in column_name is in the intersecting_series.
        :param df_to_filter: DataFrame to filter
        :param intersecting_series: series or list to filter by
        :param column_name: column in the DataFrame to check if exists in the series
        :param negative: if true - drop the columns the exists in intersecting_series
        :return: filtered DataFrame
        """
        if negative:
            return df_to_filter[~df_to_filter[column_name].isin(intersecting_series)]

        return df_to_filter[df_to_filter[column_name].isin(intersecting_series)]


    def _read_cmap_samples_data(self):
        """
        Read and normalize, the raw data samples.
        """
        raw_data_df = pd.read_hdf(self.cmap_samples_data).T

        self.logger.info("CMap raw data info:")
        self.logger.info(f"Number of genes per sample = {raw_data_df.shape[1]:,}")
        self.logger.info(f"Number of samples = {raw_data_df.shape[0]:,}")

        raw_data_df = raw_data_df.loc[self.info_df.index]
        raw_data_df = raw_data_df.astype(np.float64)

        #Michael - fill all outliers above 95% quantile or below 5% with corresponding quantile values
        raw_data_df = raw_data_df.apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))
        # Scale the data to 0-1.
        self.scaler = preprocessing.MinMaxScaler()
        # self.scaler = preprocessing.StandardScaler()
        scaled_data_np = self.scaler.fit_transform(raw_data_df)
        self.data_df = pd.DataFrame(scaled_data_np, columns=raw_data_df.columns, index=raw_data_df.index)


    def _read_cell_lines_info(self):
        """
        Read cell lines information.
        """
        # Read the file contain the cell lines info.
        cell_lines_df = pd.read_csv(self.cell_lines_info_filename, sep="\t")

        # Drop samples without primary site or type.
        # Also - drop all normal tissues.
        self.cell_lines_df = cell_lines_df[(~cell_lines_df["primary_site"].isin(["-666"])) &
                                           (~cell_lines_df["sample_type"].isin(["-666"]))]


    def _read_perturbation_of_samples(self):
        """
        Read samples information, and perform required filtering.
        """
        dtypes = {'inst_id ': str,
                  'rna_plate': str,
                  'rna_well': str,
                  'pert_id': str,
                  'pert_iname': str,
                  'pert_type': str,
                  'pert_dose': np.floating,
                  'pert_dose_unit': str,
                  'pert_time': np.int32,
                  'pert_time_unit': str,
                  'cell_id': str}
        self.info_df = pd.read_csv(self.perturbation_to_samples_filename, sep="\t", dtype=dtypes)

        # Set index of info_df.
        self.info_df.index = self.info_df["inst_id"]

        # Drop samples without cell line information.
        self.info_df = self._filter_by_series(self.info_df, self.cell_lines_df.cell_id, 'cell_id')

        # If configure - keep only perturbations whitelist.
        if self.perturbations_whitelist:
            # Add all control perturbations labels
            whitelist_perturbation = [*self.untreated_labels, *self.perturbations_whitelist]
            self.info_df = self.info_df[self.info_df['pert_iname'].isin(whitelist_perturbation)]

        # Add column for perturbation time + time unit
        self.info_df['pert_time_label'] = self.info_df["pert_time"].astype(str) + self.info_df["pert_time_unit"]

        # Add requested columns to perturbation_of_samples_df.
        self._create_cell_type_columns()
        self._create_classifier_labels()

        # Add columns for exact perturbation and exact tumor, and the numeric labels.
        self.info_df['perturbation'] = self.info_df["pert_iname"]
        # change DMSO 6+ to "time <pert_time><pert_time_unit>"
        untreated_non_initial_time_idx = self.info_df['perturbation'].isin(self.untreated_labels) & \
                                         ~self.info_df['pert_time'].isin(self.untreated_times)
        self.info_df.loc[untreated_non_initial_time_idx, ['perturbation']] = 'time ' + self.info_df.loc[
            untreated_non_initial_time_idx]['pert_time_label']

        self.info_df['tumor']        = self.info_df["cell_id"] + ' ' + self.info_df["primary_site"] + \
                                       ' ' + self.info_df["subtype"]

        # If needed, remove tumors not in whitelist.
        if self.tissues_whitelist:
            self.info_df = self.info_df[self.info_df['tumor'].isin(self.tissues_whitelist)]

        # Remove samples with low occurrence.
        self._remove_samples_with_low_occurrence()

        # Remove samples that we can't use for our calculations.
        self._remove_none_calculated_samples()


    def _remove_none_calculated_samples(self):
        """
        Remove samples that are not relevant for the latent-space calculation.
        It can be in this cases:
        1. Perturbated samples on start time.
        2. Tissues without perturbated samples at pert time.
        3. Tissues without control samples at pert time.
        4. Tissues without control samples at start time.
        """
        # Drop perturbated samples that their pert time is not in config.config_map['perturbation_times']
        self.info_df = self.info_df[(self.info_df['pert_type'] != 'trt_cp') |
                                    (self.info_df['pert_time'].isin(self.perturbation_times))]

        # Drop perturbated samples that the pert dose is less than in config.config_map['min_treat_conc']
        self.info_df = self.info_df[(self.info_df['pert_type'] != 'trt_cp') |
                                    (self.info_df['pert_dose'] >= self.min_treat_conc)]

        # Drop Control samples that their pert time is not in  config.config_map['untreated_labels_times']
        self.info_df = self.info_df[(self.info_df['pert_type'] == 'trt_cp') |
                                    (self.info_df['pert_time'].isin(self.untreated_labels_times))]

        tumors_list = self.info_df.tumor.unique()
        for tumor in tumors_list:
            # Extract the current cell samples.
            samples_of_tumor_df = self.info_df[self.info_df.tumor == tumor]

            # Extract the current tumor untreated samples.
            untreated_samples_of_tumor_df = samples_of_tumor_df[samples_of_tumor_df.pert_type == 'ctl_vehicle']

           # Extract the current tumor treated samples.
            treated_samples_of_tumor_df = samples_of_tumor_df[samples_of_tumor_df.pert_type == 'trt_cp']

            # Actually filtering:
            if (treated_samples_of_tumor_df.shape[0] == 0) or (untreated_samples_of_tumor_df.shape[0] == 0):
                # Drop the whole cell - if there are no untreated samples (in pert time) or there are no treated samples.
                self.info_df = self.info_df[self.info_df.tumor != tumor]


    def _remove_samples_with_low_occurrence(self):
        """
        Remove samples that have less occurrences than min_treat_per_cell for the same cell_id and pert_iname.
        :param --
        """
        min_treat_per_cell = self.min_samples_per_cloud

        group = self.info_df.groupby(['cell_id', 'pert_iname', 'pert_time'])['inst_id']
        id_series = group.filter(lambda x: len(x) >= min_treat_per_cell)
        self.info_df = self._filter_by_series(self.info_df, id_series, 'inst_id')


    @staticmethod
    def _create_converting_table(df, index_column, column_to_get):
        """
        Create converting table from some index to column (i.e: dictionary {"id_1" : "type_a"...}.
        :param df: data frame to create the converting table.
        :param index_column: index column of the table.
        :param column_to_get: value column of the table.
        :return: the converting table.
        """
        converting_table = {}
        for index, row in df.iterrows():
            converting_table[row[index_column]] = row[column_to_get]
        return converting_table


    @classmethod
    def _add_column_from_another_df(cls, df, other_df, index_column, column_name):
        """
        Add custom column to df, from another df, based on:
        :param df: df to add column to.
        :param other_df:df to read the column from.
        :param index_column: the index column to join between those 2 df.
        :param column_name: column name to add.
        """
        converting_table = cls._create_converting_table(other_df, index_column, column_name)
        df[column_name] = df.apply(lambda row: converting_table[row[index_column]], axis=1)


    def _create_cell_type_columns(self):
        """
        Create columns in perturbation_of_samples_df that will contain the cell tissue and state.
        """
        self._add_column_from_another_df(self.info_df, self.cell_lines_df, "cell_id", "sample_type")
        self._add_column_from_another_df(self.info_df, self.cell_lines_df, "cell_id", "primary_site")
        self._add_column_from_another_df(self.info_df, self.cell_lines_df, "cell_id", "subtype")


    def _create_classifier_labels(self):
        """
        Create "classifier_labels" column
        """
        # Assign classifier_labels to all the rows.
        self.info_df['classifier_labels'] = self.info_df['cell_id'] + ' ' + self.info_df["primary_site"] + ' ' + \
                                            self.info_df["sample_type"] + ' ' + self.info_df["pert_iname"] + ' ' + \
                                            self.info_df["pert_time_label"]


    def _print_samples_statistics(self):
        """
        Print statistics for each tumor.
        """
        if self.info_df.shape[0] == 0:
            self.logger.info("After DataOrganizer actions, no data is left!! Check your filtering configuration...")
            os._exit(1)

        self.logger.info("CMap data info:")
        cmap_clouds_num = 0
        total_samples_in_start_time = 0
        total_samples_in_pert_time = 0
        cmap_info_df = self.info_df

        for tumor in cmap_info_df['tumor'].unique():
            tumor_samples_df = cmap_info_df[cmap_info_df['tumor'] == tumor]
            for perturbation in tumor_samples_df['perturbation'].unique():
                perturbation_samples_df = tumor_samples_df[tumor_samples_df['perturbation'] == perturbation]
                all_perturbation_samples = perturbation_samples_df.shape[0]
                start_time_samples_df = perturbation_samples_df[
                    perturbation_samples_df['pert_time'].isin(self.untreated_times)]
                samples_in_start_time = start_time_samples_df.shape[0]
                start_time_list = start_time_samples_df.pert_time.unique()
                pert_time_samples_df = perturbation_samples_df[
                    ~perturbation_samples_df['pert_time'].isin(self.untreated_times)]
                samples_in_pert_time = pert_time_samples_df.shape[0]
                pert_time_list = pert_time_samples_df.pert_time.unique()
                self.logger.info(f'Tumor {tumor} perturbation {perturbation} have: {samples_in_start_time:,} samples '
                                 f'in start time, {samples_in_pert_time:,} samples in pertubrations time, '
                                 f'{all_perturbation_samples:,} samples at all')
                string = "Start time list={}, Pert time list={}."
                string_revised = string.rjust(len(string) + 70)
                self.logger.info(string_revised.format(start_time_list, pert_time_list))
                cmap_clouds_num += 1
                total_samples_in_start_time += samples_in_start_time
                total_samples_in_pert_time += samples_in_pert_time
        self.logger.info("All data info:")
        self.logger.info(f"Total number of start time samples={total_samples_in_start_time:,}")
        self.logger.info(f"Total number of pert time samples={total_samples_in_pert_time:,}")
        self.logger.info(f"Total number of loaded CMap data samples={cmap_info_df.shape[0]:,}")
        self.logger.info(f"Total number of loaded perturbations={len((self.info_df['perturbation'].unique())):,}: {list(self.info_df['perturbation'].unique())}")
        self.logger.info(f"Total Number of loaded tissues={len((self.info_df['tumor'].unique()))}: {list(self.info_df['tumor'].unique())}")


    def organize_data(self):
        """
        Read and process all the data, and save it to output files.
        """
        # Read data info.
        self._read_cell_lines_info()
        self._read_perturbation_of_samples()

        # Actually read the CMap data.
        self._read_cmap_samples_data()

        # Remove all the data that not in the selected samples, after all the filtering.
        self.data_df = self.data_df.loc[self.info_df["inst_id"]]

        # Create numeric labels, this must be placed after all building and filtering the data.
        labels, _ = pd.factorize(self.info_df["classifier_labels"])
        self.info_df["numeric_labels"] = labels

        # Print samples statistics.
        self._print_samples_statistics()

        # Save the data and the information in 2 organized files.
        organized_data_folder = self.organized_cmap_folder
        data_path = os.path.join(organized_data_folder, self.data_file_name)
        info_path = os.path.join(organized_data_folder, self.information_file_name)
        self.data_df.to_hdf(data_path, key='df')
        self.info_df.to_csv(info_path, sep=',', columns=['perturbation',
                                                         'tumor',
                                                         'classifier_labels',
                                                         'numeric_labels',
                                                         'pert_time',
                                                         'pert_time_unit',
                                                         'pert_time_label'])

        # Save  data for scaling inversion.
        scaler_path = os.path.join(organized_data_folder, "cmap_scaler")
        joblib.dump(self.scaler, scaler_path)

        # Create and save the configuration file for the sbatch_runner running on the HPC.
        self._create_unique_clouds_file()

        self.logger.info('CMap data organization successfully done.')


    def _create_unique_clouds_file(self):
        """
        Create the all clouds file for training controler and  sbatch_runner (running on the HPC).
        """
        unique_clouds_path = os.path.join(self.organized_cmap_folder, self.unique_clouds_file_name)
        unique_clouds_df = pd.concat([self.info_df['perturbation'], self.info_df['tumor']], axis=1).drop_duplicates()
        unique_clouds_df = unique_clouds_df[~unique_clouds_df.perturbation.isin(self.untreated_labels)]
        unique_clouds_df.index = range(len(unique_clouds_df.index))
        unique_clouds_df.to_csv(unique_clouds_path, sep=',', columns=['perturbation', 'tumor'])


if __name__ == '__main__':
    organizer = CmapOrganizer(
        raw_cmap_folder=config.raw_cmap_folder,
        organized_cmap_folder=config.organized_cmap_folder,
        perturbations_whitelist=config.perturbations_whitelist if config.use_perturbations_whitelist else None,
        untreated_labels=config.untreated_labels,
        untreated_times=config.untreated_times,
        tissues_whitelist=config.tissues_whitelist if config.use_tissues_whitelist else None,
        perturbation_times=config.perturbation_times,
        min_treat_conc=config.min_treat_conc,
        untreated_labels_times=config.cmap_organizer_untreated_labels_times,
        data_file_name=config.data_file_name,
        information_file_name=config.information_file_name,
        min_samples_per_cloud=config.min_samples_per_cloud,
        unique_clouds_file_name=config.unique_clouds_file_name
    )
    organizer.organize_data()
