from configuration import config
from latent_space_arithmetic import LatentSpaceArithmetic
import numpy as np
import pandas as pd
import logging
import os


class TrajectoriesTests:
    """
    This class handles the creation of trajectories
    """

    def __init__(self, use_random_pairs=False):
        """
        Initialize the class
        :param use_random_pairs: If set, use random pairs instead of reference points
        """
        self.use_random_pairs = use_random_pairs

    def get_start_points(self, data_df, model, control_df, reference_points, samples_per_cloud):
        if config.config_map['trajectories_startpoint_is_24']:
            control_index = control_df[control_df['pert_time'].isin(config.config_map['untreated_times'])].index
        else:
            control_index = control_df[~control_df['pert_time'].isin(config.config_map['untreated_times'])].index

        if self.use_random_pairs:
            rs_start_df = data_df.loc[control_index].sample(n=samples_per_cloud, replace=True)
            ls_start_df = model.predict_latent_space(rs_start_df)
        else:
            rs_start_df = reference_points[5].loc[control_index].sample(n=samples_per_cloud, replace=True)
            ls_start_df = model.predict_latent_space(rs_start_df)
        return ls_start_df

    def get_end_points(self, data_df, model, latent_space_arithmetic, info_df, tumor, perturbation, reference_points,
                       samples_per_cloud, ls_start_df):
        control_df = info_df[
            (info_df['perturbation'].isin(config.config_map['untreated_labels'])) &
            (info_df['tumor'] == tumor)]
        if config.config_map['trajectories_endpoint_is_predicted']:
            # In case start point is DMSO6 -> use them as base points
            if config.config_map['trajectories_startpoint_is_24']:
                ls_base_df = ls_start_df
            else:
                # Otherwise, find random samples as requested
                base_index = control_df[
                    control_df['pert_time'].isin(config.config_map['untreated_times'])].index
                if self.use_random_pairs:
                    base_info_df = data_df.loc[base_index].sample(n=samples_per_cloud, replace=True)
                else:
                    base_info_df = reference_points[5].loc[base_index].sample(n=samples_per_cloud, replace=True)
                rs_base_data_df = data_df.loc[base_info_df]
                ls_base_df = model.predict_latent_space(rs_base_data_df)

            perturbations = [perturbation]
            ls_dest_df, _ = \
                latent_space_arithmetic.calculate_perturbations_effect_for_given_data(ls_base_df, control_df,
                                                                                      perturbations, reference_points)
        else:
            # Use treated samples
            dest_index = info_df[(info_df['perturbation'] == perturbation) & (info_df['tumor'] == tumor)].index
            # In case of random pairs: choose randomly from treated samples
            if self.use_random_pairs:
                rs_dest_df = data_df.loc[dest_index].sample(n=samples_per_cloud, replace=True)
            else:
                # Otherwise - just take center of cloud
                rs_dest_df = reference_points[5].loc[dest_index].sample(n=samples_per_cloud,replace=True)
            ls_dest_df = model.predict_latent_space(rs_dest_df)
        return ls_dest_df

    def get_vector_before_and_after_distances(self, model, latent_space_arithmetic, tumor, perturbation,
                                              data_df, info_df, reference_points):
        """
        For current perturbation and tumor, print points on the perturbation vector from control to treated,
        in latent space and after decoding-encoding
        :param model: model handler
        :param latent_space_arithmetic: latent space arithmetic class
        :param tumor: tumor to test
        :param perturbation: perturbation to test
        :return: list of distances for each vector between one and another
        """
        samples_per_cloud = 1
        if self.use_random_pairs:
            samples_per_cloud = config.config_map['trajectories_samples_to_choose']

        # Find start point indexes
        control_df = info_df[
                (info_df['perturbation'].isin(config.config_map['untreated_labels'])) &
                (info_df['tumor'] == tumor)]

        ls_start_df = self.get_start_points(data_df, model, control_df, reference_points, samples_per_cloud)
        ls_dest_df = self.get_end_points(data_df, model, latent_space_arithmetic, info_df, tumor, perturbation,
                                         reference_points, samples_per_cloud, ls_start_df)

        # Create matrix that contains points between control start time and treated
        points_on_vector = config.config_map['trajectories_points_on_vector']
        points_np = np.linspace(ls_start_df, ls_dest_df, points_on_vector)
        points_np = np.transpose(points_np, axes=[1, 0, 2])
        points_np = points_np.reshape(samples_per_cloud * points_on_vector, ls_start_df.shape[1])

        idx = pd.MultiIndex.from_product([range(samples_per_cloud), range(points_on_vector)])
        points_df = pd.DataFrame(points_np, index=idx, columns=ls_start_df.columns)

        # Decode and encode
        decoded_df = model.predict_decoder(points_df)
        decoded_df.columns = data_df.columns

        # If not calculating random pairs, calculate decode-encode distance also
        decoded_encoded_points_df = None
        decoded_encoded_points_distances = None
        if not self.use_random_pairs:
            decoded_encoded_points_df = model.predict_latent_space(decoded_df)

            distances_df = decoded_encoded_points_df - points_df
            decoded_encoded_points_distances = np.linalg.norm(distances_df.values, axis=1)
            decoded_encoded_points_distances = decoded_encoded_points_distances / decoded_encoded_points_distances[0]

        return points_df, decoded_df, decoded_encoded_points_df, decoded_encoded_points_distances

    def run(self, test_name, data, model):
        """
        Run perturbations test - angles between each pair of perturbations,
        and create hdf with sets of samples that change for each perturbation
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        logging.info("Starting perturbation tests: %s", str(test_name))
        data_df, info_df, reference_points = data.get_all_data_and_info()
        clouds_types = info_df[['perturbation', 'tumor']].drop_duplicates()

        latent_space_arithmetic = LatentSpaceArithmetic(data, model)
        output_folder = os.path.join(config.config_map['output_folder'], 'trajectories_tests')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # In case of regular pairs, also calculate the noise distance
        if not self.use_random_pairs:
            idx = pd.MultiIndex.from_frame(clouds_types)
            points_list = list(range(config.config_map['trajectories_points_on_vector']))
            pert_vectors_decode_encode_df = pd.DataFrame(index=idx,
                                                         columns=points_list,
                                                         dtype=float)

        # Create index, it is <perturbation> <tumor> <sample number> <point on vector>
        decoded_index_df = clouds_types.copy()
        decoded_index_df['sample_num'] = 0
        decoded_index_df["point_num"] = 0

        # In case of random pairs, there might be more than 1 sample per cloud
        if self.use_random_pairs:
            samples_per_cloud = config.config_map['trajectories_samples_to_choose']
            decoded_index_df = decoded_index_df.append(
                [decoded_index_df] * (samples_per_cloud - 1), ignore_index=True)
            decoded_index_df = decoded_index_df.sort_values(by=['tumor', 'perturbation']).reset_index(
                drop=True)
            for i in range(samples_per_cloud):
                curr_indexes = decoded_index_df.loc[decoded_index_df.index % samples_per_cloud == i].index
                decoded_index_df.loc[curr_indexes, 'sample_num'] = i

        points_on_vector = config.config_map['trajectories_points_on_vector']
        decoded_index_df = decoded_index_df.append([decoded_index_df] * (points_on_vector - 1), ignore_index=True)
        decoded_index_df = decoded_index_df.sort_values(by=['tumor', 'perturbation', 'sample_num']).reset_index(drop=True)
        for i in range(points_on_vector):
            curr_indexes = decoded_index_df.loc[decoded_index_df.index % points_on_vector == i].index
            decoded_index_df.loc[curr_indexes, 'point_num'] = i

        idx = pd.MultiIndex.from_frame(decoded_index_df)
        decoded_df = pd.DataFrame(index=idx, columns=data_df.columns, dtype=float)
        encoded_df = pd.DataFrame(index=idx, columns=range(config.config_map['latent_dim']), dtype=float)

        # Calculate and save decode-encode distances normalized for each neuron
        for _, row in clouds_types.iterrows():
            tumor = row.tumor
            perturbation = row.perturbation
            curr_encoded_df, curr_decoded_df, _, distances =\
                self.get_vector_before_and_after_distances(model, latent_space_arithmetic, tumor, perturbation,
                                                           data_df, info_df, reference_points)
            decoded_df.loc[perturbation, tumor].update(curr_decoded_df)
            encoded_df.loc[perturbation, tumor].update(curr_encoded_df)
            if not self.use_random_pairs:
                pert_vectors_decode_encode_df.loc[(perturbation, tumor)] = distances

        pre_string = ''
        if self.use_random_pairs:
            pre_string = 'random_'
        decoded_df.to_hdf(os.path.join(output_folder, pre_string + 'decoded.h5'), 'df')
        encoded_df.to_hdf(os.path.join(output_folder, pre_string + 'encoded.h5'), 'df')
        if not self.use_random_pairs:
            pert_vectors_decode_encode_df.to_hdf(os.path.join(output_folder, 'pert_vectors_decode_encode.h5'), 'df')
