from configuration import config
from tcga_gtex_organizer import HealtyTissuesDataOrganizer

from sklearn import preprocessing
import joblib
import pandas as pd
import numpy as np
import os
import shutil
import logging


class DataOrganizer:
    """
    Organize the data from raw CMAP files into 2 files - data.h5 and info.csv.
    """

    def __init__(self):
        """
        Initializer - set all files constants.
        """
        # Create output root folder if not exists
        if not os.path.isdir(config.config_map['root_output_folder']):
            os.makedirs(config.config_map['root_output_folder'])

        # Set the logger configuration
        self._set_logger()

        self.data_folder = config.config_map['raw_data_folder']

        # This file has only CMAP genes - 977, and not all the genes.
        self.cmap_samples_data = os.path.join(self.data_folder,
                                              'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x977.h5')

        # This file contain the perturbation that each sample got.
        self.perturbation_to_samples_filename = os.path.join(self.data_folder, 'GSE92742_Broad_LINCS_inst_info.txt.gz')

        # This file contain metadata about the cell lines
        self.cell_lines_info_filename = os.path.join(self.data_folder, 'GSE92742_Broad_LINCS_cell_info.txt')

        # This file contain live-dead data for part of the cell lines
        self.live_dead_filename = os.path.join(self.data_folder, 'CMAP_LiveDead_v3.txt')

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

    @staticmethod
    def _filter_by_series(df_to_filter, intersecting_series, column_name, negative=False):
        """
        Filter DataFrame by series - keep only the rows where their value in column_name is in the intersecting_series
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
        index_to_scale = self.info_df.index.append(self.test_info_df.index)
        raw_data_df = pd.read_hdf(self.cmap_samples_data).T
        raw_data_df = raw_data_df.loc[index_to_scale]
        raw_data_df = raw_data_df.astype(np.float64)

        # Scale the data to 0-1.
        self.scaler = preprocessing.MinMaxScaler()
        scaled_data_np = self.scaler.fit_transform(raw_data_df)
        all_data_df = pd.DataFrame(scaled_data_np, columns=raw_data_df.columns, index=raw_data_df.index)
        self.data_df = all_data_df.loc[self.info_df.index]
        self.test_data_df = all_data_df.loc[self.test_info_df.index]

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

    @staticmethod
    def _add_column_from_another_df(df, other_df, index_column, columns_names):
        """
        Add custom column to df, from another df, based on:
        :param df: df to add column to.
        :param other_df:df to read the column from.
        :param index_column: the index column to join between those 2 df.
        :param columns_names: list of columns names to add
        """
        other_df = other_df.set_index(index_column)
        return df.join(other_df[columns_names], on=index_column, how='inner')

    def _remove_uncalculated_samples(self):
        """
        Remove samples that are not relevant for the latent-space calculation.
        It can be in this cases:
        1. Perturbations that don't have minimum big tumors
        2. Tissues without perturbated samples at pert time.
        3. Tissues without control samples at pert time.
        4. Tissues without control samples at start time.
        """
        # Filter perturbations that don't have minimum big tumors per perturbation
        groups = self.info_df.groupby(by=['perturbation', 'tumor'])
        count_df = groups.count()[['inst_id']]
        perts = count_df.index.get_level_values(0)
        min_samples_per_clouds = config.config_map['organizer_min_samples_per_cell_line']
        perts_to_drop = []
        print('Number of clouds before removing uncalculated samples: {}'.format(len(perts)))
        i = 0
        for p in perts:
            i += 1
            if i % 1000 == 0:
                print('Check perturbation {}'.format(i))
            pert_df = count_df.loc[p]
            big_clouds = pert_df[pert_df.inst_id >= min_samples_per_clouds].shape[0]
            if big_clouds < config.config_map['organizer_min_cell_lines_per_perturbation']:
                perts_to_drop.append(p)
        self.info_df.drop(self.info_df[self.info_df.perturbation.isin(perts_to_drop)].index, axis=0, inplace=True)
        print('Finish dropping uncalculated perturbations')

        # Filter tumors that don't have the 3 clouds
        tumors_list = self.info_df.tumor.unique()
        for tumor in tumors_list:
            # Extract the current cell samples.
            samples_of_tumor_df = self.info_df[self.info_df.tumor == tumor]

            # Extract the current cell untreated samples.
            untreated_samples_of_tumor_df = samples_of_tumor_df[samples_of_tumor_df.pert_type == 'ctl_vehicle']

            untreated_samples_start_time_df = untreated_samples_of_tumor_df[
                untreated_samples_of_tumor_df['pert_time'].isin(config.config_map['untreated_times'])]

            untreated_samples_pert_time_df = untreated_samples_of_tumor_df[
                untreated_samples_of_tumor_df['pert_time'].isin(config.config_map['perturbation_times'])]

            # Extract the current cell treated samples.
            treated_samples_of_cell_df = samples_of_tumor_df[samples_of_tumor_df.pert_type == 'trt_cp']

            # Actually filtering:
            if (untreated_samples_start_time_df.shape[0] == 0) or (treated_samples_of_cell_df.shape[0] == 0) or (
                    untreated_samples_pert_time_df.shape[0] == 0):
                # Drop the whole cell - if there are no untreated samples in both pert time and start time
                # or there are no treated samples.
                self.info_df = self.info_df[self.info_df.tumor != tumor]

        print ("Finish dropping uncalculated tumors")
        groups = self.info_df.groupby(by=['perturbation', 'tumor'])
        count_df = groups.count()[['inst_id']]
        perts = count_df.index.get_level_values(0)
        print('Number of clouds after removing uncalculated samples: {}'.format(len(perts)))

    def _read_perturbation_of_samples(self):
        """
        Read the perturbations that each samples passed
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

        # Set index of info_df
        self.info_df.index = self.info_df["inst_id"]

        # Drop samples without cell line information.
        self.info_df = self._filter_by_series(self.info_df, self.cell_lines_df.cell_id, 'cell_id')

        # Add requested columns to perturbation_of_samples_df.

        self.info_df = self._add_column_from_another_df(self.info_df, self.cell_lines_df, "cell_id", ["sample_type",
                                                                                                      "primary_site",
                                                                                                      "subtype"])
        self.info_df['perturbation'] = self.info_df["pert_iname"]
        self.info_df['tumor'] = self.info_df["cell_id"] + ' ' + self.info_df["primary_site"] + ' ' + self.info_df[
            "subtype"]

        # Add columns for exact perturbation and exact tumor, and the numeric labels.
        self.info_df.pert_dose = self.info_df['pert_dose'].map(str) + self.info_df['pert_dose_unit'].map(str)

        # If configure - keep only drug whitelist.
        if config.config_map['organizer_use_perturbations_whitelist']:
            whitelist_perturbation = config.config_map['organizer_perturbations_whitelist']
            whitelist_perturbation.extend(config.config_map['untreated_labels'])
            self.info_df = self.info_df[self.info_df['pert_iname'].isin(whitelist_perturbation)]

        # If configure - keep only tissues whitelist.
        if config.config_map['organizer_use_tissues_whitelist']:
            whitelist_tissues = config.config_map['organizer_tissues_whitelist']
            self.info_df = self.info_df[self.info_df['tumor'].isin(whitelist_tissues)]

        # Remove samples that we can't use for our calculations.
        self._remove_uncalculated_samples()

        # Drop Control samples that their pert time is not in config.config_map['untreated_times']
        # or config.config_map['perturbation_times'] or "test_pert_time"
        self.test_info_df = self.info_df[self.info_df['pert_time'].isin(config.config_map['test_pert_times'])]
        self.info_df = self.info_df[(self.info_df['pert_type'] == 'trt_cp') |
                                    (self.info_df['pert_time'].isin(config.config_map['untreated_times'])) |
                                    (self.info_df['pert_time'].isin(config.config_map['perturbation_times']))]
        self.info_df = self.info_df[(self.info_df['pert_type'] != 'trt_cp') |
                                    (self.info_df['pert_time'].isin(config.config_map['perturbation_times']))]

    def _print_samples_statistics(self):
        """
        Print statistics for each tumor
        """
        for t in self.info_df['tumor'].unique():
            tumor_samples_df = self.info_df[self.info_df['tumor'] == t]
            for p in tumor_samples_df['perturbation'].unique():
                perturbation_samples_df = tumor_samples_df[tumor_samples_df['perturbation'] == p]
                all_perturbation_samples = perturbation_samples_df.shape[0]
                samples_in_start_time = perturbation_samples_df[
                    perturbation_samples_df['pert_time'].isin(config.config_map['untreated_times'])].shape[0]
                samples_in_pert_time = all_perturbation_samples - samples_in_start_time
                logging.info('Tumor %s perturbation %s have: %d samples in start time, %d samples in'
                             ' pertubrations time, %d samples at all',
                             t, p, samples_in_start_time, samples_in_pert_time, all_perturbation_samples)

    def read_live_dead_data(self):
        """
        Read information regarding the live and dead for the samples
        :return: DataFrame with perturbation, tumor and death rate
        """
        # Read the file contain the cell lines info.
        live_dead_df = pd.read_csv(self.live_dead_filename, sep="\t")
        live_dead_df.index = live_dead_df.inst
        live_dead_df = self.info_df.join(live_dead_df, how='inner')
        self.live_dead_df = pd.DataFrame(columns=['perturbation', 'tumor', 'death_rate'])
        for p in live_dead_df.perturbation.unique():
            pert_cloud = live_dead_df[live_dead_df.perturbation == p]
            for t in pert_cloud.tumor.unique():
                tumor_cloud = pert_cloud[pert_cloud.tumor == t]
                death_rate = tumor_cloud.livedead.mean()
                self.live_dead_df = self.live_dead_df.append(
                    {'perturbation': p, 'tumor': t, 'death_rate': death_rate},
                    ignore_index=True)

    def organize_data(self):
        """
        Read and process all the data, and save it to output files.
        """
        # Read the data.
        self._read_cell_lines_info()
        self._read_perturbation_of_samples()
        self.read_live_dead_data()

        # Print samples statistics
        #self._print_samples_statistics()

        # Read the data
        self._read_cmap_samples_data()

        # Also save gtex if needed.
        if config.config_map['healthy_tissues_data_is_needed']:
            self._read_gtex_samples_data_and_info()

        # Save the data and the information that needed.
        organized_data_folder = config.config_map['organized_data_folder']

        # Delete old folder.
        if os.path.isdir(organized_data_folder):
            shutil.rmtree(organized_data_folder)

        # Create the folder.
        os.makedirs(organized_data_folder)

        # Save the data.
        data_path = os.path.join(organized_data_folder, config.config_map['data_file_name'])
        info_path = os.path.join(organized_data_folder, config.config_map['information_file_name'])
        self.data_df.to_hdf(data_path, key='df')
        self.info_df.to_csv(info_path, sep=',', columns=['inst_id', 'perturbation', 'tumor', 'pert_time', 'pert_dose'], index=False)

        test_data_path = os.path.join(organized_data_folder,
                                           config.config_map['test_data_file_name'])
        test_info_path = os.path.join(organized_data_folder,
                                           config.config_map['test_info_file_name'])
        self.test_data_df.to_hdf(test_data_path, key='df')
        self.test_info_df.to_csv(test_info_path, sep=',', columns=['perturbation', 'tumor',
                                                                   'pert_time',
                                                                   'pert_dose'])

        live_dead_path = os.path.join(organized_data_folder, config.config_map['live_dead_file_name'])
        self.live_dead_df.to_csv(live_dead_path, sep=',')

        scaler_path = os.path.join(organized_data_folder, "cmap_scaler")
        joblib.dump(self.scaler, scaler_path)

        # Create and save the configuration file for the sbatch_runner running on the HPC.
        self._create_unique_clouds_file()

        logging.info('Cmap data organization successfully done!!')

    def _read_gtex_samples_data_and_info(self):
        """
        Read and concat gtex data and info.
        """
        # Handle to healty tissues organizer.
        healty_tissue_organizer = HealtyTissuesDataOrganizer()

        # Load gtex info and data.
        healty_tissue_organizer.read_gtex_info_and_data(self.data_df.columns)

        # Scale gtex data if needed.
        self.info_df = pd.concat([self.info_df, healty_tissue_organizer.info_df], sort=False)
        self.data_df = pd.concat([self.data_df, healty_tissue_organizer.data_df])
        logging.info("Total number of GTEx samples={}".format(healty_tissue_organizer.data_df.shape[0]))
        logging.info(
            'Triangle version {} - TCGA-GTEx data organization successfully done!!\n'.format(
                config.config_map['version']))

    def _create_unique_clouds_file(self):
        """
        Create the all clouds file for training controler and  sbatch_runner (running on the HPC).
        """
        unique_clouds_file_name = config.config_map['unique_clouds_file_name']
        unique_clouds_path = os.path.join(config.config_map['organized_data_folder'], unique_clouds_file_name)
        info_df = self.info_df[~self.info_df.perturbation.isin(config.config_map['untreated_labels'])]
        g = info_df.groupby([ 'perturbation', 'tumor'])
        occurences = g.agg('count').inst_id
        filtered = occurences[occurences > config.config_map['minimum_samples_to_classify']]
        unique_df = filtered.reset_index()[['tumor', 'perturbation']]
        unique_df.to_csv(unique_clouds_path)


if __name__ == '__main__':
    organizer = DataOrganizer()
    organizer.organize_data()
