from System.data_handler import DataHandler
from System.statistics_calculator import StatisticsCalculator
from System.configuration import config
import pandas as pd
import string


def _file_name_escaping(filename):
    """
    Make escaping for file names (i.e: omit '|' or '\'.
    :param filename: filename to escape
    :return: escaped filename
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


def one_correlation(tumor, perturbation):
    """
    Do correlation test to one cloud
    :param tumor: tumor to test
    :param perturbation: perturbation to test
    """
    config.config_map['leave_out_tissue_name'] = tumor
    config.config_map['leave_out_perturbation_name'] = perturbation
    config.config_map['test_set_percent'] = 0
    config.config_map['leave_data_out'] = True

    data = DataHandler()

    control_pert_time_info_df = data.info_df[(data.info_df.tumor == tumor) &
                                             (data.info_df.perturbation.isin(config.config_map['untreated_labels'])) &
                                             (data.info_df.pert_time.isin(config.config_map['perturbation_times']))]
    control_pert_time_data_df = data.data_df.loc[control_pert_time_info_df.index]
    treated_data_df = data.left_data_df
    nearest_points_df = StatisticsCalculator.find_closest_points_between_clouds(treated_data_df,
                                                                                control_pert_time_data_df,
                                                                                1)
    colors = ['k', 'b', 'y', 'r']
    names = ['treated', 'control pert time']
    correlations_dict = StatisticsCalculator.calculate_correlations(treated_data_df, control_pert_time_data_df,
                                                                    nearest_points_df, colors, names)
    figure_name = _file_name_escaping('{0} {1} correlations'.format(tumor, perturbation))
    StatisticsCalculator.print_distributions(correlations_dict, figure_name, r'D:\Thesis\real space densities')
    distances_dict = StatisticsCalculator.calculate_distances(treated_data_df, control_pert_time_data_df,
                                                              nearest_points_df, colors, names)
    figure_name = _file_name_escaping('{0} {1} distances'.format(tumor, perturbation))
    StatisticsCalculator.print_distributions(distances_dict, figure_name, r'D:\Thesis\real space densities')


def main():
    unique_clouds_df = pd.read_csv('unique_clouds_cmap.csv')
    for i in range(unique_clouds_df.shape[0]):
        row = unique_clouds_df.iloc[i]
        tumor = row['tumor']
        pert = row['perturbation']
        one_correlation(tumor, pert)


if __name__ == '__main__':
    main()
