from configuration import config
import itertools
from latent_space_arithmetic import LatentSpaceArithmetic
from post_processing.death_predictor import DeathPredictor
import logging
import os
import pandas as pd
import numpy as np


class DrugCombinations:
    """
    Find new killing perturbations based on known killing perturbations
    """

    def __init__(self):
        self.test_name = None
        self.data = None
        self.output_folder = None
        self.model = None
        self.latent_space_arithmetic = None
        self.data_df, self.info_df, self.reference_points = None, None, None
        self.death_predictor = None
        self.meta_index = None
        self.encoded_df = None
        self.alpha_scaled_vectors = None

    @staticmethod
    def add_drug_combination_columns(df, cell_line, drugs):
        """
        Add columns to drug combinations df by drugs
        :param df: df to add columns
        :param cell_line: cell line of df data
        :param drugs: drugs that applied
        """
        df['cell_line'] = cell_line
        for j in range(len(drugs)):
            df['drug_{}'.format(j)] = drugs[j]
        for j in range(len(drugs), config.config_map['drug_combination_max_drugs']):
            df['drug_{}'.format(j)] = "-"

    @staticmethod
    def get_drugs_from_info(info_df):
        """
        Get list of treatable drugs (no DMSO) from information df
        :param info_df: DataFrame with info
        :return: list of drugs
        """
        drugs_list = list(info_df.perturbation.unique())
        for d in config.config_map['untreated_labels']:
            if d in drugs_list:
                drugs_list.remove(d)
        return drugs_list

    def initialize_properties(self, test_name, data, model):
        self.test_name = test_name
        self.data = data
        self.model = model
        self.latent_space_arithmetic = LatentSpaceArithmetic(data, model)
        self.data_df, self.info_df, self.reference_points = self.data.get_all_data_and_info()
        self.encoded_df = self.model.predict_latent_space(self.data.data_df)
        if config.config_map['drug_combination_predict_death']:
            self.death_predictor = DeathPredictor(self.data)
        self.meta_index = ['cell_line']
        for i in range(config.config_map['drug_combination_max_drugs']):
            self.meta_index.append('drug_{}'.format(i))
        self.alpha_scaled_vectors = {}

    def set_output_dir(self, dir_postfix):
        self.output_folder = os.path.join(config.config_map['output_folder'], 'drug_combinations' + dir_postfix)
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

    def meta_columns_to_index(self, df):
        df.index.name = 'sample_id'
        df.set_index(self.meta_index, drop=True, append=True, inplace=True)
        return df.reorder_levels(self.meta_index + ['sample_id'])

    def create_results_df(self, results_dict):
        data_df, _, _ = self.data.get_all_data_and_info()

        genes_columns = list(self.meta_index) + list(data_df.columns)
        predictions_columns = self.meta_index + ['death']

        if config.config_map['drug_combination_run_combined_prediction']:
            if config.config_map['drug_combination_save_genes']:
                results_dict['combinations'] = pd.DataFrame(columns=genes_columns)
            if config.config_map['drug_combination_predict_death']:
                combinations_predictions_df = pd.DataFrame(columns=predictions_columns)
                results_dict['combinations_predictions'] = self.meta_columns_to_index(combinations_predictions_df)
        if config.config_map['drug_combination_sample_sphere_by_drug_vector']:
            if config.config_map['drug_combination_save_genes']:
                results_dict['spheres'] = pd.DataFrame(columns=genes_columns)
            if config.config_map['drug_combination_predict_death']:
                combinations_spheres_by_vector_predictions_df = pd.DataFrame(columns=predictions_columns)
                results_dict['spheres_predictions'] = self.meta_columns_to_index(combinations_spheres_by_vector_predictions_df)           

    @staticmethod
    def print_number_of_combinations(drugs_list, cell_line_list):
        combinations_num = 0
        for i in range(1, config.config_map['drug_combination_max_drugs'] + 1):
            for comb in itertools.combinations(drugs_list, i):
                combinations_num += 1
        combinations_num *= len(cell_line_list)
        logging.info("Drug combinations: having %d combinations at all", combinations_num)

    def add_results(self, encoded_points, decoded_points, c, drugs, results_dict):
        """
        Calculate and add results to results dictionary
        :param encoded_points: encoded points of drug combinations
        :param decoded points: decoded points of drug combinations
        :param c: current cell line
        :param drugs: list of drugs that run
        """  
        should_unscale = config.config_map['drug_combinations_unscale_before_convert_to_12k']
        if 'combinations' in results_dict or 'combinations_predictions' in results_dict:
            self.add_drug_combination_columns(decoded_points, c, drugs)
            decoded_points = self.meta_columns_to_index(decoded_points)
            decoded_points = self.data.get_12k_data(decoded_points, should_unscale)
            if 'combinations' in results_dict:
                results_dict['combinations'] = results_dict['combinations'].append(decoded_points, sort=False)
            if 'combinations_predictions' in results_dict:
                death_predictions = self.death_predictor.predict(decoded_points)
                results_dict['combinations_predictions'] = results_dict['combinations_predictions'].append(death_predictions, sort=False)
        
        if 'spheres' in results_dict or 'spheres_predictions' in results_dict:
            decoded_points = self.sample_sphere_by_drug_vector(encoded_points, c, drugs[-1])
            self.add_drug_combination_columns(decoded_points, c, drugs)
            decoded_points = self.meta_columns_to_index(decoded_points)
            decoded_points = self.data.get_12k_data(decoded_points, should_unscale)
            if 'spheres' in results_dict:
                results_dict['spheres'] = results_dict['spheres'].append(decoded_points, sort=False)
            if 'spheres_predictions' in results_dict:
                death_predictions = self.death_predictor.predict(decoded_points)
                results_dict['spheres_predictions'] = results_dict['spheres_predictions'].append(death_predictions, sort=False)

    def calculate_combinations_by_method(self, cell_line_list, prediction_function):
        """
        Calculate drug combinations by given prediction function
        :param cell_line_list: list of cell lines to predict
        :param prediction_function: function to predict drug combinations
        :param save_prefix: prefix string for output files
        """
        data_df, info_df, reference_points = self.data.get_all_data_and_info()
        drugs_list = self.get_drugs_from_info(info_df)

        # Create DF with genes columns and columns for metadata: cell line and the used drugs
        results = {}
        self.create_results_df(results)

        # Print how many combinations we have at all
        self.print_number_of_combinations(drugs_list, cell_line_list)

        # For each cell line, gather base samples that will be used to each drug combination
        combinations_done = 0
        for c in cell_line_list:
            control_info_df = info_df[
                (info_df.perturbation.isin(config.config_map['untreated_labels'])) & (info_df.tumor == c)]
            base_samples_info_df = control_info_df[control_info_df.pert_time.isin(config.config_map['untreated_times'])]
            if config.config_map['drug_combination_samples_per_cloud'] == 0:
                base_samples_df = data_df.loc[base_samples_info_df.index]
            else:
                base_samples_df = data_df.loc[base_samples_info_df.index].sample(
                    config.config_map['drug_combination_samples_per_cloud'], replace=True)
            encoded_base_samples_df = self.model.predict_latent_space(base_samples_df)

            # Run over all possible drug combinations that contains between 1 to max_drugs drugs
            for i in range(1, config.config_map['drug_combination_max_drugs'] + 1):
                for drugs in itertools.combinations(drugs_list, i):
                    logging.info("Drug combinations: Working on combination %d", combinations_done)
                    combinations_done += 1

                    # Calculate drugs effect
                    encoded_points, decoded_points = prediction_function(encoded_base_samples_df, control_info_df, drugs, reference_points)
                    self.add_results(encoded_points, decoded_points, c, drugs, results)

        # save all outputs
        self.save_output(results)

    def save_output(self, output_dict):
        for k in output_dict.keys():
            output_filename = os.path.join(self.output_folder, '{}_{}'.format(k, self.test_name))
            output_dict[k].to_hdf(output_filename, 'df')

    def run(self, test_name, data, model):
        """
        Run drug combinations and create hdf with samples for each combinations
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        logging.info("Starting drug combinations tests: %s", str(test_name))
        self.initialize_properties(test_name, data, model)
        if config.config_map['drug_combinations_calculate_cmap']:
            if config.config_map['drug_combination_cell_lines']:
                cell_line_list = config.config_map['drug_combination_cell_lines']
            else:
                _, info_df, _ = self.data.get_all_data_and_info()
                cell_line_list = self.info_df[self.info_df.perturbation == 'DMSO'].tumor.unique()
            prediction_function = self.latent_space_arithmetic.calculate_perturbations_effect_for_given_data
            self.set_output_dir('_cmap')
            self.calculate_combinations_by_method(cell_line_list, prediction_function)
        if config.config_map['drug_combinations_calculate_gtex']:
            if config.config_map['drug_combination_cell_lines']:
                cell_line_list = config.config_map['drug_combination_cell_lines']
            else:
                _, info_df, _ = self.data.get_all_data_and_info()
                cell_line_list = self.info_df[self.info_df.perturbation == 'GTEX'].tumor.unique()
            prediction_function = self.calculate_drugs_by_alpha_from_cmap
            self.set_output_dir('_gtex')
            self.calculate_combinations_by_method(cell_line_list, prediction_function)


    def sample_sphere_by_drug_vector(self, encoded_points, cell_line, drug):
        control_info_df = self.info_df[(self.info_df['tumor'] == cell_line) &
                                       (self.info_df['perturbation'].isin(config.config_map['untreated_labels']))]
        start_time_control_info_df = control_info_df[
            control_info_df['pert_time'].isin(config.config_map['untreated_times'])]
        pert_time_control_info_df = control_info_df[
            ~control_info_df['pert_time'].isin(config.config_map['untreated_times'])]

        start_time_sample_ref = self.reference_points[5].loc[start_time_control_info_df.index].iloc[0]
        pert_time_sample_ref = self.reference_points[5].loc[pert_time_control_info_df.index].iloc[0]
        ref_points_df = pd.DataFrame([start_time_sample_ref, pert_time_sample_ref])
        encoded_control_ref_df = self.model.predict_latent_space(ref_points_df)
        control_6h_np = encoded_control_ref_df.iloc[0].values
        control_24h_np = encoded_control_ref_df.iloc[1].values

        drug_vector, _ = self.latent_space_arithmetic.get_drug_vectors_from_base_points(drug, self.reference_points,
                                                                                        control_6h_np, control_24h_np)

        return self.sample_sphere_by_std(encoded_points, np.linalg.norm(drug_vector) / 2)

    def sample_sphere_by_std(self, encoded_points, std_to_calculate):
        """
        Sample sphere by std
        :param encoded_points: Cloud to sample from it's sphere
        :param std_to_calculate: number, std to sample
        :return: Dataframe of samples data, decoded
        """
        center_point = encoded_points.mean(axis=0)
        num_of_samples = config.config_map['drug_combination_num_of_spheres_samples']
        samples_left_to_sample = num_of_samples
        columns_num = encoded_points.shape[1]
        samples = np.zeros(shape=(num_of_samples, columns_num))
        tolerance = config.config_map['drug_combination_sample_std_tolerance']
        std_vector = np.full(shape=center_point.shape, fill_value=std_to_calculate)
        std_tolerance = np.linalg.norm(std_vector) * tolerance

        # Keep sampling until we have enough
        while samples_left_to_sample > 0:
            new_samples = np.random.normal(center_point, std_vector, [num_of_samples, columns_num])
            distances = np.linalg.norm(new_samples - center_point.values, axis=1)
            new_samples = new_samples[distances >= std_tolerance]
            samples_to_take = min(new_samples.shape[0], samples_left_to_sample)
            new_samples = new_samples[0:samples_to_take]
            start_index = num_of_samples - samples_left_to_sample
            end_index = start_index + samples_to_take
            samples[start_index:end_index] = new_samples
            samples_left_to_sample -= new_samples.shape[0]

        samples_df = pd.DataFrame(samples)
        decoded_df = self.model.predict_decoder(samples_df)
        decoded_df.columns = self.data.data_df.columns
        return decoded_df
    
    def get_alpha_for_drug(self, drug):
        data = self.encoded_df
        info = self.info_df
        tumors = info[info.perturbation == drug].tumor.unique()
        length_list = []
        for t in tumors:
            control_start_time = info[(info.perturbation.isin(config.config_map['untreated_labels'])) &
                                      (info.tumor == t) & 
                                      (info.pert_time.isin(config.config_map['untreated_times']))].iloc[0].name
            treated_sample = info[(info.perturbation == drug) & (info.tumor == t)].iloc[0].name
            
            start_ref = self.reference_points[5].loc[control_start_time]
            end_ref = self.reference_points[5].loc[treated_sample]
            ref_points_df = pd.DataFrame([start_ref, end_ref])
            encoded_ref_df = self.model.predict_latent_space(ref_points_df)
            length_list.append(np.linalg.norm(encoded_ref_df.iloc[0] - encoded_ref_df.iloc[1]))
        return np.mean(length_list)

    def get_treatment_vector_for_drug(self, drug):
        data = self.encoded_df
        info = self.info_df
        treated_sample = info[(info.perturbation == drug)].iloc[0].name
        start_ref = self.reference_points[1].loc[treated_sample]
        end_ref = self.reference_points[2].loc[treated_sample]
        ref_points_df = pd.DataFrame([start_ref, end_ref])
        encoded_ref_df = self.model.predict_latent_space(ref_points_df)
        return (encoded_ref_df.iloc[0] - encoded_ref_df.iloc[1]).values

    def calculate_drugs_by_alpha_from_cmap(self, encoded_base_samples_df, control_info_df, drugs, reference_points):
        """
        This function is a mock of Alpha method - calculate drugs by alpha version only
        :param encoded_base_samples_df: base samples to add drugs, in latent space
        :param control_info_df: info_df of control points
        :param drugs: list of drugs to add
        :param reference_points: list of reference points
        """
        # calculate vector for each drug
        calculated_vectors = []
        for d in drugs:
            if d not in self.alpha_scaled_vectors:
                alpha = self.get_alpha_for_drug(d)
                vector = self.get_treatment_vector_for_drug(d)
                scaled_vector = np.linalg.norm(vector) / alpha * vector
                self.alpha_scaled_vectors[d] = scaled_vector
            calculated_vectors.append(self.alpha_scaled_vectors[d])

        # Apply all the calculated vectors we gather
        calculated_samples_df = encoded_base_samples_df
        for cv in calculated_vectors:
            calculated_samples_df = calculated_samples_df + cv

        # Create copy of the data in real space
        decoded_calculated_df = self.model.predict_decoder(calculated_samples_df)
        decoded_calculated_df.columns = self.data.data_df.columns
        return calculated_samples_df, decoded_calculated_df


