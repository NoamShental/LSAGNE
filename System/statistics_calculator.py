from configuration import config
import numpy as np
import pandas as pd
import os
import logging
import warnings
import string
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt


class StatisticsCalculator:
    """
    This class will compute the correlation between calculated and treated clouds
    """
    def __init__(self, treated_tuple):
        """
        Initialization of class - set treated cloud
        :param treated tuple: tuple of DF that the first 2 are: treated data in latent space, treated data in real space
        """
        self.treated_latent_space_df = treated_tuple[0]
        self.treated_real_space_df = treated_tuple[1]
        self.output_folder = os.path.join(config.config_map['pictures_folder'], 'Densities Diagrams')
        if not os.path.isdir(self.output_folder):
            os.makedirs(self.output_folder)

    @staticmethod
    def find_closest_points_between_clouds(cloud1, cloud2, use_times=0):
        """
        Create DataFrame with indexes of samples name in cloud 1, and their nearest samples in cloud 2
        :param cloud1: calculated cloud
        :param cloud2: treated cloud
        :param use_times: if not 0, use each point in cloud 2 maximum of use_times times
        :return: DataFrame
        """
        nearest_points_df = pd.DataFrame(index=cloud1.index, columns=['nearest_sample'])
        use_dict = {}
        if use_times != 0:
            cloud2 = cloud2.copy()

        # Shuffle the DataFrame
        calculated_cloud = cloud1.sample(frac=1)

        for index, row in calculated_cloud.iterrows():
            nearest_sample_index = np.linalg.norm(row.values - cloud2.values, axis=1).argmin()
            nearest_sample_index = cloud2.iloc[nearest_sample_index].name
            nearest_points_df.loc[index].nearest_sample = nearest_sample_index
            if use_times != 0:
                if nearest_sample_index in use_dict:
                    use_dict[nearest_sample_index] += 1
                else:
                    use_dict[nearest_sample_index] = 1
                if use_dict[nearest_sample_index] == use_times:
                    cloud2.drop(nearest_sample_index, inplace=True)
        return nearest_points_df

    @staticmethod
    def calculate_function_cloud_to_nearest_points(cloud1, cloud2, nearest_points, calculate_function):
        """
        For each point in cloud1, take the nearest point in cloud 2, and calculate a given function between them.
        :param cloud1: calculated cloud.
        :param cloud2: treated cloud
        :param nearest_points: Series with nearest treated sample to each sample calculated samples
        :param calculate_function: function to call when calculating
        :return: results array
        """
        results = np.zeros(cloud1.shape[0], dtype=np.float64)
        prev_config = np.seterr(invalid='ignore')
        for i in range(cloud1.shape[0]):
            curr_sample = cloud1.iloc[i]
            nearest_sample = cloud2.loc[nearest_points.loc[curr_sample.name]]
            results[i] = calculate_function(curr_sample.values, nearest_sample.values.flatten())
        np.seterr(invalid=prev_config['invalid'])
        return results

    @staticmethod
    def _file_name_escaping(filename):
        """
        Make escaping for file names (i.e: omit '|' or '\'.
        :param filename: filename to escape
        :return: escaped filename
        """
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        return ''.join(c for c in filename if c in valid_chars)

    @staticmethod
    def print_distributions(distributions_dict, figure_name, output_folder, figure_title=None):
        """
        Print multiple distributions on same figure
        :param distributions_dict: dict of distributions: {label: [distribution_np, color]
        :param figure_name: name of figure
        :param output_folder: output folder to save the figure
        :param figure_title: title of figure
        """
        prev_config = np.seterr(divide='ignore', invalid='ignore')
        for key, value in distributions_dict.items():
            label = key
            correlation = value[0]
            if correlation.shape[0] > 500:
                correlation = np.random.choice(correlation, size=500)
            color = value[1]
            try:
                sns.distplot(correlation, color=color, hist=False, label=label)
            except np.linalg.LinAlgError as e:
                logging.error('Encounter Singular Matrix')
        np.seterr(divide=prev_config['divide'], invalid=prev_config['invalid'])
        if figure_title is not None:
            plt.suptitle(figure_title, size=10)
        plt.savefig(os.path.join(output_folder, figure_name))
        plt.close("all")

    @staticmethod
    def calculate_distances(calculated_cloud, treated_cloud, nearest_points, colors, clouds_names):
        """
        Calculate distances between:
            1. calculated to nearest treated
            2. all calculated to all treated
            3. all treated to all treated
            4. all calculated to all calculated
        :param calculated_cloud: DataFrame with calculated data
        :param treated_cloud: DataFrame with treated data
        :param nearest_points: Series of coupling calculated to it's nearest treated point
        :param colors: tuple of strings to colors
        :param clouds_names: names of clouds
        :return: distributions dictionary, ready to print
        """
        distances_c_to_t = StatisticsCalculator.calculate_function_cloud_to_nearest_points(
            calculated_cloud, treated_cloud, nearest_points, (lambda x, y: np.linalg.norm(x - y)))
        distances_c_to_t_all = euclidean_distances(calculated_cloud, treated_cloud).flatten()
        distances_t_to_t_all = euclidean_distances(treated_cloud, treated_cloud).flatten()
        distances_c_to_c_all = euclidean_distances(calculated_cloud, calculated_cloud).flatten()
        name0, name1 = clouds_names
        distances_dict = {'{} to nearest {}'.format(name0, name1): [distances_c_to_t, colors[0]],
                          'all {} to all {}'.format(name0, name1): [distances_c_to_t_all, colors[1]],
                          '{} to itself'.format(name1): [distances_t_to_t_all, colors[2]],
                          '{} to itself'.format(name0): [distances_c_to_c_all, colors[3]]}
        c_t_mean, c_t_std = np.mean(distances_c_to_t), np.std(distances_c_to_t)
        c_t_all_mean, c_t_all_std = np.mean(distances_c_to_t_all), np.std(distances_c_to_t_all)
        t_t_mean, t_t_std = np.mean(distances_t_to_t_all), np.std(distances_t_to_t_all)
        c_c_mean, c_c_std = np.mean(distances_c_to_c_all), np.std(distances_c_to_c_all)
        numeric_dict = {'calculate to nearest treated dist': (c_t_mean, c_t_std),
                        'calculate to treated dist': (c_t_all_mean, c_t_all_std),
                        'treated to treated dist': (t_t_mean, t_t_std),
                        'calculate to calculated dist': (c_c_mean, c_c_std)}
        return distances_dict, numeric_dict

    @staticmethod
    def calculate_correlations(calculated_cloud, treated_cloud, nearest_points, colors, clouds_names):
        """
        Calculate correlations between:
            1. calculated to nearest treated
            2. all calculated to all treated
            3. all treated to all treated
            4. all calculated to all calculated
        :param calculated_cloud: DataFrame with calculated data
        :param treated_cloud: DataFrame with treated data
        :param nearest_points: Series of coupling calculated to it's nearest treated point
        :param colors: tuple of strings to colors
        :param clouds_names: names of clouds
        :return: distributions dictionary, ready to print
        """
        # Wrap each ndarray with nan to num, because pearson correlation may return nan in case the covariance
        # if one of the arrays are 0
        # Also, filter out warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        correlations_c_to_t = np.nan_to_num(StatisticsCalculator.calculate_function_cloud_to_nearest_points(
            calculated_cloud, treated_cloud, nearest_points, (lambda x, y: pearsonr(x, y)[0])), copy=False)
        correlations_c_to_t_all = np.nan_to_num(np.corrcoef(calculated_cloud, treated_cloud, rowvar=True).flatten(), copy=False)
        correlations_t_to_t_all = np.nan_to_num(np.corrcoef(treated_cloud, treated_cloud, rowvar=True).flatten(), copy=False)
        correlations_c_to_c_all = np.nan_to_num(np.corrcoef(calculated_cloud, calculated_cloud, rowvar=True).flatten(), copy=False)
        warnings.filterwarnings("default", category=RuntimeWarning)
        name0, name1 = clouds_names
        correlations_dict = {'{} to nearest {}'.format(name0, name1): [correlations_c_to_t, colors[0]],
                             'all {} to all {}'.format(name0, name1): [correlations_c_to_t_all, colors[1]],
                             '{} to itself'.format(name1): [correlations_t_to_t_all, colors[2]],
                             '{} to itself'.format(name0): [correlations_c_to_c_all, colors[3]]}
        c_t_mean, c_t_std = np.mean(correlations_c_to_t), np.std(correlations_c_to_t)
        c_t_all_mean, c_t_all_std = np.mean(correlations_c_to_t_all), np.std(correlations_c_to_t_all)
        t_t_mean, t_t_std = np.mean(correlations_t_to_t_all), np.std(correlations_t_to_t_all)
        c_c_mean, c_c_std = np.mean(correlations_c_to_c_all), np.std(correlations_c_to_c_all)
        numeric_dict = {'calculate to nearest treated corr': (c_t_mean, c_t_std),
                        'calculate to treated corr': (c_t_all_mean, c_t_all_std),
                        'treated to treated corr': (t_t_mean, t_t_std),
                        'calculate to calculated corr': (c_c_mean, c_c_std)}
        return correlations_dict, numeric_dict

    def do_statistical_tests(self, calculated_tuple, state, tumor, perturbation, data):
        """
        Calculate distances distribution and correlation distribution between treated and calculated clouds
        :param calculated_tuple: tuple of DF that the first 2 are: calculated latent space, calculated real space,
        :param labels_prefix: prefix to name of distance distribution
        """
        labels_prefix = self._file_name_escaping('{} {} {}'.format(state, tumor, perturbation))
        calculated_latent_space_df = calculated_tuple[0]
        calculated_real_space_df = calculated_tuple[1]
        calculated_cloud = calculated_latent_space_df
        treated_cloud = self.treated_latent_space_df
        latent_space_colors = ['k', 'b', 'c', 'g']
        names = ['calculated', 'treated']

        # Calculate nearest points
        # Because we can use each treated sample only use_times times, we have to take maximum
        # of  use_times * treated_cloud.shape[0] samples from calculated cloud
        calculated_samples = calculated_cloud.shape[0]
        treated_samples = treated_cloud.shape[0]
        use_times = calculated_samples / treated_samples + (calculated_samples % treated_samples > 0)
        nearest_points_df = self.find_closest_points_between_clouds(calculated_cloud, treated_cloud,
                                                                    use_times=use_times)

        # Correlation in latent space
        logging.info('%s correlation in latent space', labels_prefix)
        latent_space_correlations_dict, _ = self.calculate_correlations(calculated_cloud, treated_cloud,
                                                                        nearest_points_df['nearest_sample'],
                                                                        latent_space_colors, names)
        name = '{0} {1} latent space correlation distribution'.format(labels_prefix, config.config_map['test_number'])
        self.print_distributions(latent_space_correlations_dict, name, self.output_folder)

        # Distances in latent space
        logging.info('%s distances in latent space', labels_prefix)
        latent_space_distances_dict, _ = self.calculate_distances(calculated_cloud, treated_cloud,
                                                                  nearest_points_df['nearest_sample'],
                                                                  latent_space_colors, names)
        name = '{0} {1} latent space distance distribution'.format(labels_prefix, config.config_map['test_number'])
        self.print_distributions(latent_space_distances_dict, name, self.output_folder)

        # Correlation in real space
        calculated_cloud = calculated_real_space_df
        treated_cloud = self.treated_real_space_df

        # If configured - perform real space statistics only with dominant genes.
        if config.config_map['genes_filtering_is_needed']:
            dominates_genes_list = data.drugs_dominates_genes_dict[perturbation]
            logging.info('Number of {} dominat genes: {}'.format(perturbation, len(dominates_genes_list)))
            calculated_cloud = calculated_cloud[dominates_genes_list]
            treated_cloud = treated_cloud[dominates_genes_list]

        real_space_colors = ['k', 'b', 'c', 'g']
        logging.info('%s correlation in real space', labels_prefix)
        real_space_correlations_dict, results_dict = self.calculate_correlations(calculated_cloud, treated_cloud,
                                                                                     nearest_points_df['nearest_sample'],
                                                                                     real_space_colors, names)
        name = '{0} {1} real space correlation distribution'.format(labels_prefix, config.config_map['test_number'])
        self.print_distributions(real_space_correlations_dict, name, self.output_folder)

        # Distances in real space
        logging.info('%s distances in real space', labels_prefix)
        real_space_distances_dict, distance_dict = self.calculate_distances(calculated_cloud, treated_cloud,
                                                                            nearest_points_df['nearest_sample'],
                                                                            real_space_colors,
                                                                            names)
        name = '{0} {1} real space distance distribution'.format(labels_prefix, config.config_map['test_number'])
        results_dict.update(distance_dict)
        self.print_distributions(real_space_distances_dict, name, self.output_folder)
        return results_dict
