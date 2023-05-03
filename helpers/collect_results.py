import string
import os
import json
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.metrics.pairwise import euclidean_distances
from ast import literal_eval


def file_name_escaping(filename):
    """
    Make escaping for file names (i.e: omit '|' or '\'.
    :param filename: filename to escape
    :return: escaped filename
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


def read_svm_success_rates(results_folder, cloud_tests, results_to_p_value):
    """
    Read svm results for single cloud
    :param results_folder: path to results folder
    :param cloud_tests: DataFrame with all the tests
    :param results_to_p_value: DataFrame of results to p value tests
    :return: success and std of the cloud, filtered and unfiltered
    """
    unfiltered_results_df = pd.DataFrame(columns=['success', 'std'])
    filtered_results_df = pd.DataFrame(columns=['success', 'std'])
    summarize_np = None
    predicted_as_control = None
    latent_space_hdf_path = os.path.join('Tests', 'svm_latent_space_results.h5')

    # Read all the trials results, and save them in numpy arrays
    for i in range(cloud_tests.shape[0]):
        repeat_folder = file_name_escaping('_'.join(cloud_tests.iloc[i].map(str)))
        hdf_path = os.path.join(results_folder, repeat_folder, latent_space_hdf_path)
        success_df = pd.read_hdf(hdf_path, 'df')
        if summarize_np is None:
            summarize_np = np.zeros(shape=[success_df.shape[0], cloud_tests.shape[0]])
            predicted_as_control = np.zeros(shape=[success_df.shape[0], cloud_tests.shape[0]])
        summarize_np[:, i] = success_df['predicted'] == success_df['real']
        results_to_p_value.loc[cloud_tests.iloc[i].name, 'svm'] = summarize_np[:, i].mean()

        # The system will set the treated samples  that predicted as control, it comes from the test as -1
        predicted_as_control[:, i] = success_df['predicted'] == -1

    # Fill up unfiltered DataFrame
    unfiltered_results_df['success'] = summarize_np.mean(axis=1) * 100
    unfiltered_results_df['std'] = summarize_np.std(axis=1) * 100

    samples_to_filter = predicted_as_control.sum(axis=1) == cloud_tests.shape[0]
    filtered_results_df['success'] = unfiltered_results_df[~samples_to_filter]['success']
    filtered_results_df['std'] = unfiltered_results_df[~samples_to_filter]['std']
    unfiltered_success = unfiltered_results_df['success'].mean()
    unfiltered_std = unfiltered_results_df['std'].mean()
    filtered_success = filtered_results_df['success'].mean()
    filtered_std = filtered_results_df['std'].mean()
    return [[unfiltered_success, unfiltered_std], [filtered_success, filtered_std]]


def read_drop_statistics_to_repeat(test_folder):
    """
    Collect all the data about dropped samples from single test
    :param test_folder: folder of test
    :return: dictionary of dropped samples as: {perturbation: {tumor: dropped mean}}
    """
    drop_statistics_dict = {}

    # Read samples drop count
    drop_df = pd.read_csv(os.path.join(test_folder, 'Models', 'dropped_points.csv'))
    for statistic_index, row in drop_df.iterrows():
        drop_for_cloud = row.dropped / row.start
        if str(row.perturbation) not in drop_statistics_dict:
            drop_statistics_dict[str(row.perturbation)] = {}
        drop_statistics_dict[str(row.perturbation)][str(row.tumor)] = drop_for_cloud
    return drop_statistics_dict


def read_drop_statistics_to_cloud(cloud_tests, results_folder, drop_statistics_dict):
    """
    Collect the data from all the repeats to one cloud, and update the drop statistics dict
    :param cloud_tests: DataFrame of tests in current clouds
    :param results_folder: folder to read the results from
    :param drop_statistics_dict: dictionary with drop samples statistics, this dictionary will be updated
    """
    # For each repeat
    for index, single_test in cloud_tests.iterrows():
        test_folder = file_name_escaping('_'.join(single_test.map(str)))
        repeat_folder = os.path.join(results_folder, test_folder)

        # Collect drop statistics
        drop_statistics_for_test = read_drop_statistics_to_repeat(repeat_folder)
        for inner_p, t_dict in drop_statistics_for_test.items():
            for inner_t, value in t_dict.items():
                inner_t = file_name_escaping(inner_t)
                if inner_t not in drop_statistics_dict[inner_p]:
                    continue
                if drop_statistics_dict[inner_p][inner_t][0] == -10:
                    drop_statistics_dict[inner_p][inner_t] = [value, 1]
                else:
                    drop_statistics_dict[inner_p][inner_t][0] += value
                    drop_statistics_dict[inner_p][inner_t][1] += 1


def add_single_distribution(current_values, new_results):
    """
    Update current values with one more results
    :param current_values: tuple of (sum_of_mean, sum_of_std, count_of_repeats)
    :param new_results: tuple of (mean, std) of new result
    :return: updated values (add mean to mean, std to std, and accumulate count by 1).
    """
    current_values[0] += new_results[0]
    current_values[1] += new_results[1]
    current_values[2] += 1
    return current_values


def read_correlations_and_distances_to_cloud(cloud_tests, results_folder, results_to_p_value, correlations_to_nearest_dict,
                                             distances_to_nearest_dict, compare_corr, compare_distance,
                                             naive_results_df):
    """
    Read correlations and distances to one cloud
    :param cloud_tests: DataFrame of tests in current clouds
    :param results_folder: folder to read the results from
    :param results_to_p_value: dictionary with drop samples statistics, this dictionary will be updated
    :param correlations_to_nearest_dict: dictionary with correlation to nearest calculated
    :param distances_to_nearest_dict: dictionary with distance to nearest calculated
    :param compare_corr: Compared correlation dict
    :param compare_distance: Compared distance dict
    :param naive_results_df: DataFrame with naive results
    :return: 2 Dictionaries with results of cloud correlation and cloud distances
    """
    cloud_corr_dict = OrderedDict([("calculate to treated corr", [0., 0., 0]),
                                   ("treated to treated corr", [0., 0., 0])])
    cloud_dist_dict = OrderedDict([("calculate to treated dist", [0., 0., 0]),
                                   ("treated to treated dist", [0., 0., 0])])

    # For each repeat
    for index, single_test in cloud_tests.iterrows():
        repeat_folder = os.path.join(results_folder, file_name_escaping('_'.join(single_test.map(str))))
        test_results_json = os.path.join(repeat_folder, 'results.json')
        with open(test_results_json, 'r') as f:
            results_dict = json.load(f)
        statistics_dict = results_dict['Arithmetic']['statistics_results']
        for key, value in cloud_corr_dict.items():
            cloud_corr_dict[key] = add_single_distribution(value, statistics_dict[key])
        for key, value in cloud_dist_dict.items():
            cloud_dist_dict[key] = add_single_distribution(value, statistics_dict[key])

        results_to_p_value.loc[index, 'correlations'] = statistics_dict['calculate to treated corr'][0]
        results_to_p_value.loc[index, 'distances'] = statistics_dict['calculate to treated dist'][0]

    # Calculate mean of each of the values
    for key, value in cloud_corr_dict.items():
        cloud_corr_dict[key] = [np.around(value[0] / value[2], 4), np.around(value[1] / value[2], 4)]
    for key, value in cloud_dist_dict.items():
        cloud_dist_dict[key] = [np.around(value[0] / value[2], 4), np.around(value[1] / value[2], 4)]
    corr_results = list(cloud_corr_dict.values())
    distance_results = list(cloud_dist_dict.values())

    # Add pre compiled results
    t = cloud_tests.iloc[0].tumor
    p = cloud_tests.iloc[0].perturbation
    if correlations_to_nearest_dict is not None:
        corr_results.append(correlations_to_nearest_dict[t][p])
    if distances_to_nearest_dict is not None:
        distance_results.append(distances_to_nearest_dict[t][p])
    if compare_corr is not None:
        corr_results.append(compare_corr[p][t][0])
    if compare_distance is not None:
        distance_results.append(compare_distance[p][t][0])
    if naive_results_df is not None:
        row = naive_results_df[(naive_results_df.tumor.map(file_name_escaping) == t) &
                               (naive_results_df.perturbation == p)].iloc[0]
        corr_tuple = literal_eval(row['corr'])
        dist_tuple = literal_eval(row['distance'])
        corr_results.append(list(corr_tuple))
        distance_results.append(list(dist_tuple))
    return corr_results, distance_results


def mean_std_calculator(samples_df_list):
    """
    Calc mean and std for given list of dataframes
    :param samples_df_list: list of df
    :return: mean and std df
    """
    N = len(samples_df_list)
    mean_df = pd.DataFrame(0, samples_df_list[0].index, samples_df_list[0].columns)
    for i in range(N):
        sample_df = samples_df_list[i]
        mean_df += (sample_df / N)
    sum_df = pd.DataFrame(0, mean_df.index, mean_df.columns)
    for i in range(N):
        sample_df = samples_df_list[i]
        sum_df += (sample_df - mean_df)**2
    sum_df /= (N-1)
    return mean_df, sum_df.apply(np.sqrt)


def get_pd_read_function_for_path(df_path):
    """
    Get appropriate pandas read function for path
    :param df_path: path to read, can be relative
    :return: function pointer to use
    """
    if df_path.endswith('.csv'):
        return pd.read_csv
    elif df_path.endswith('.h5'):
        def read_hdf(path):
            return pd.read_hdf(path, 'df')
        return read_hdf
    else:
        raise TypeError("Unsupported extension")


def calc_mean_and_std_df(results_folder, tests_df, df_path):
    """
    Read DF from each test folder, and calculate mean and std of that dataframe among the results
    :param results_folder:  folder of results
    :param tests_df: DataFrame of tests
    :param df_path: path to df in each result folder
    :return: 1 matrix with average of euclidean distances for each cloud
    """
    perturbations_list = tests_df.perturbation.unique()
    df_list = []
    read_function = get_pd_read_function_for_path(df_path)
    for p in perturbations_list:
        pert_tests = tests_df[tests_df.perturbation == p]
        for t in pert_tests.tumor.unique():
            cloud_tests = pert_tests[pert_tests.tumor == t]
            for index, single_test in cloud_tests.iterrows():
                repeat_folder = os.path.join(results_folder, file_name_escaping('_'.join(single_test.map(str))))
                db_path = os.path.join(repeat_folder, df_path)
                current_df = read_function(db_path)
                df_list.append(current_df)
    mean_df, std_df = mean_std_calculator(df_list)
    return mean_df, std_df


def collect_confusion_table_by_method(test_folder, method, escaped_tumor_list, drugs):
    """
    Collect confusion tables for custom method
    :param test_folder: folder of current test
    :param method: svm or SystemClassifier
    :param escaped_tumor_list:  collect only for those tumors
    :param perturbations: collect only for those drugs
    :return: DataFrame with the results (2 indexes: tissue and perturbation)
    """
    result_folder = os.path.join(test_folder, 'ConfusionTable', method)
    index = pd.MultiIndex.from_product([escaped_tumor_list, drugs], names=['tissue', 'drug'])
    drug_columns = ['DMSO'] + drugs + ['other']
    results_df = pd.DataFrame(index=index, columns=drug_columns)
    results_df.sort_index(axis=0, level=['tissue', 'drug'], inplace=True)
    for tumor in escaped_tumor_list:
        tumor_df = pd.read_csv(os.path.join(result_folder, tumor) + '.csv')
        tumor_df.columns.values[0] = 'Perturbation'
        tumor_df = tumor_df.set_index('Perturbation')
        for d in drugs:
            try:
                results_df.loc[tumor, d] = tumor_df.loc[d]
            except:
                pass
    return results_df


def collect_confusion_table(results_folder, tests_df):
    perturbations_list = tests_df.perturbation.unique().tolist()
    tumor_list = tests_df.tumor.unique()

    # M2 matrix to calculate the std on the fly, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    svm_df_list = []
    system_classifier_df_list = []

    for p in perturbations_list:
        pert_tests = tests_df[tests_df.perturbation == p]
        for t in pert_tests.tumor.unique():
            cloud_tests = pert_tests[pert_tests.tumor == t]
            for index, single_test in cloud_tests.iterrows():
                repeat_folder = os.path.join(results_folder, file_name_escaping('_'.join(single_test.map(str))))
                svm_df = collect_confusion_table_by_method(repeat_folder, 'svm', tumor_list, perturbations_list)
                svm_df_list.append(svm_df.astype(np.float))
                sc_df = collect_confusion_table_by_method(repeat_folder, 'SystemClassifier', tumor_list, perturbations_list)
                system_classifier_df_list.append(sc_df.astype(np.float))
    svm_mean_df, svm_std_df = mean_std_calculator(svm_df_list)
    system_classifier_mean_df, system_classifier_std_df = mean_std_calculator(system_classifier_df_list)
    return svm_mean_df, svm_std_df, system_classifier_mean_df, system_classifier_std_df


def collect_semi_supervised(results_folder, tests_df):
    perturbations_list = tests_df.perturbation.unique().tolist()
    # M2 matrix to calculate the std on the fly, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    df_list = []
    sample_df = None
    n = 0

    for p in perturbations_list:
        pert_tests = tests_df[tests_df.perturbation == p]
        for t in pert_tests.tumor.unique():
            cloud_tests = pert_tests[pert_tests.tumor == t]
            for index, single_test in cloud_tests.iterrows():
                repeat_folder = os.path.join(results_folder, file_name_escaping('_'.join(single_test.map(str))))
                current_df = pd.read_csv(os.path.join(repeat_folder, 'SemiSupervised', 'svm.csv'))
                df_list.append(current_df[['Treated accuracy']])
                if sample_df is None:
                    sample_df = current_df
    mean_df, std_df = mean_std_calculator(df_list)
    sample_df['Treated accuracy'] = mean_df['Treated accuracy']
    sample_df['Treated accuracy std'] = std_df['Treated accuracy']
    return sample_df


def create_treated_to_treated_distance_df(data_path, info_path, tumors_list, perts_list):
    """
    Create dictionary with distances between treated clouds
    :param data_path: path to data df
    :param info_path: path to info df
    :param tumors_list: list of tumors to calculate
    :param perts_list: list of perturbations to calculate
    :return: dictionary where it's keys are tumors, and values are data frames
    """
    data_df = pd.read_hdf(data_path, 'df')
    info_df = pd.read_csv(info_path)
    info_df.set_index('inst_id', inplace=True)
    distances = {}
    for t in info_df.tumor.unique():
        escaped_tumor = file_name_escaping(t)
        tumor_info_df = info_df[info_df.tumor == t]
        if escaped_tumor not in tumors_list:
            continue

        perts_in_cloud = tumor_info_df[tumor_info_df.perturbation.isin(perts_list)].perturbation.unique()
        distances[escaped_tumor] = {}
        for p in perts_in_cloud:
            pert_cloud_info_df = tumor_info_df[tumor_info_df.perturbation == p]
            pert_data_cloud = data_df.loc[pert_cloud_info_df.index]
            min_mean = -1
            min_std = -1
            for p2 in perts_in_cloud:
                if p2 == p:
                    continue
                pert_cloud_2_info_df = tumor_info_df[tumor_info_df.perturbation == p2]
                pert_2_data_cloud = data_df.loc[pert_cloud_2_info_df.index]
                distances_matrix = euclidean_distances(pert_data_cloud, pert_2_data_cloud)
                current_mean = distances_matrix.mean()
                if min_mean > current_mean or min_mean == -1:
                    min_mean = current_mean
                    min_std = distances_matrix.std()
            distances[escaped_tumor][p] = [min_mean, min_std]
    return distances


def create_treated_to_treated_correlation_df(data_path, info_path, tumors_list, perts_list):
    """
    Create dictionary with correlations between treated clouds
    :param data_path: path to data df
    :param info_path: path to info df
    :param tumors_list: list of tumors to calculate
    :param perts_list: list of perturbations to calculate
    :return: dictionary where it's keys are tumors, and values are data frames
    """
    data_df = pd.read_hdf(data_path, 'df')
    info_df = pd.read_csv(info_path)
    info_df.set_index('inst_id', inplace=True)
    corrs = {}
    for t in info_df.tumor.unique():
        escaped_tumor = file_name_escaping(t)
        tumor_info_df = info_df[info_df.tumor == t]
        if escaped_tumor not in tumors_list:
            continue

        perts_in_cloud = tumor_info_df[tumor_info_df.perturbation.isin(perts_list)].perturbation.unique()
        corrs[escaped_tumor] = {}
        for p in perts_in_cloud:
            pert_cloud_info_df = tumor_info_df[tumor_info_df.perturbation == p]
            pert_data_cloud = data_df.loc[pert_cloud_info_df.index]
            min_mean = -1
            min_std = -1
            for p2 in perts_in_cloud:
                if p2 == p:
                    continue
                pert_cloud_2_info_df = tumor_info_df[tumor_info_df.perturbation == p2]
                pert_2_data_cloud = data_df.loc[pert_cloud_2_info_df.index]
                corr_matrix = np.corrcoef(pert_data_cloud, pert_2_data_cloud, rowvar=True)
                current_mean = np.absolute(corr_matrix).mean()
                if min_mean > current_mean or min_mean == -1:
                    min_mean = current_mean
                    min_std = corr_matrix.std()
            corrs[escaped_tumor][p] = [min_mean, min_std]
    return corrs


def collect_statistical_results(results_folder, tests_df, angles_between_perts, vectors_noise, tcga_results,
                                confusion_table, semi_supervised, data_path, info_path, compare_results, naive_results,
                                vae_noise):
    """
    Collect statistics results from tests
    :param results_folder: folder of results
    :param tests_df: DataFrame with all the tests
    :param angles_between_perts: if True, collect angles between perturbations
    :param vectors_noise: if True, collect noise on perturbations vectors.
    :param tcga_results: if True, collect TCGA results
    :param confusion_table: if True, collect confusion table
    :param semi_supervised: if True, collect semi supervised
    :param data_path: Path to data df
    :param info_path: Path to info df
    :param compare_results: Dictionary of previous results to compare, may be None
    :param naive_results: Path to csv with the naive results
    :param vae_noise: If True, collect and calculate vae noise
    :return: dictionaries of results results as:    {"results name": results}
    """
    svm_results = {}
    drop_statistics_dict = {}
    correlations_dict = {}
    distances_dict = {}
    results_to_p_value = tests_df.copy()
    results_to_p_value['svm'] = 0.0
    results_to_p_value['correlations'] = 0.0
    results_to_p_value['correlations_nearest'] = 0.0
    results_to_p_value['distances'] = 0.0
    results_to_p_value['distances_nearest'] = 0.0

    perturbations_list = tests_df.perturbation.unique()
    tumors_list = tests_df.tumor.unique()
    correlations_to_nearest_dict = None
    distances_to_nearest_dict = None
    with_t_to_nearest = False
    with_naive = False
    with_compare_results = False
    if data_path is not None and info_path is not None:
        correlations_to_nearest_dict = create_treated_to_treated_correlation_df(data_path, info_path, tumors_list,
                                                                                perturbations_list)
        distances_to_nearest_dict = create_treated_to_treated_distance_df(data_path, info_path, tumors_list,
                                                                          perturbations_list)
        with_t_to_nearest = True
    naive_df = None
    if naive_results is not None:
        naive_df = pd.read_csv(naive_results)
        with_naive = True
    if compare_results is not None:
        with_compare_results = True

    # Initiate the dictionaries with values
    for p in perturbations_list:
        print("Collecting results to perturbation %s" % p)

        # SVM dictionary format:
        # {perturbation: {tumor: ((unfiltered_mean, unfiltered_std), (filtered_mean, filtered_std))}}
        svm_results[p] = {}

        # Drop statistics dictionary format:
        # {perturbation: {tumor: (dropped samples percentage sum, number of tests with this cloud)}}
        drop_statistics_dict[p] = {}

        # Correlation dictionary format:
        # {perturbation: {tumor: ()}}
        correlations_dict[p] = {}
        distances_dict[p] = {}
        num_of_results_dist_and_corr = 2
        if data_path is not None and info_path is not None:
            num_of_results_dist_and_corr += 1
        if compare_results is not None:
            num_of_results_dist_and_corr += 1
        if naive_results is not None:
            num_of_results_dist_and_corr += 1
        for t in tumors_list:
            svm_results[p][t] = [[0, 0], [0, 0]]
            drop_statistics_dict[p][t] = [0, 0]
            correlations_dict[p][t] = [[0.0, 0.0]] * num_of_results_dist_and_corr
            distances_dict[p][t] = [[0.0, 0.0]] * num_of_results_dist_and_corr

    # Fill the dictionaries with real values
    for p in perturbations_list:
        pert_tests = tests_df[tests_df.perturbation == p]
        for t in pert_tests.tumor.unique():
            cloud_tests = pert_tests[pert_tests.tumor == t]
            svm_results[p][t] = read_svm_success_rates(results_folder, cloud_tests, results_to_p_value)
            read_drop_statistics_to_cloud(cloud_tests, results_folder, drop_statistics_dict)
            if compare_results is not None:
                correlations_dict[p][t], distances_dict[p][t] = \
                    read_correlations_and_distances_to_cloud(cloud_tests, results_folder, results_to_p_value,
                                                             correlations_to_nearest_dict, distances_to_nearest_dict,
                                                             compare_results['correlations'], compare_results['distances'],
                                                             naive_df)
            else:
                correlations_dict[p][t], distances_dict[p][t] = \
                    read_correlations_and_distances_to_cloud(cloud_tests, results_folder, results_to_p_value,
                                                             correlations_to_nearest_dict, distances_to_nearest_dict,
                                                             None, None, naive_df)

    for t_dict in drop_statistics_dict.values():
        keys = list(t_dict.keys())
        for t in keys:
            value = t_dict[t]
            if value[0] == 0 and value[1] == 0:
                t_dict[t] = [[0, 0]]
            else:
                t_dict[t] = [[value[0] / value[1] * 100, 0]]

    results = {'svm': svm_results, 'drop_statistics': drop_statistics_dict,
               'correlations': correlations_dict, "distances": distances_dict,
               'with_t_to_nearest': with_t_to_nearest, 'with_naive': with_naive, 'with_compare': with_compare_results}

    if angles_between_perts:
        results['angles'] = \
            calc_mean_and_std_df(results_folder, tests_df,
                                 os.path.join('perturbations_tests', 'angles_between_perturbations.h5'))
    if vectors_noise:
        results['vectors_noise'] = \
            calc_mean_and_std_df(results_folder, tests_df,
                                 os.path.join('perturbations_tests', 'pert_vectors_decode_encode.h5'))
    if tcga_results:
        results['TCGA'] = \
            calc_mean_and_std_df(results_folder, tests_df,
                                  os.path.join('tcga', 'tcga.h5'))
    if confusion_table:
        results['confusion_table'] = \
            collect_confusion_table(results_folder, tests_df)

    if semi_supervised:
        results['semi_supervised'] = \
            collect_semi_supervised(results_folder, tests_df)
    if vae_noise:
        results['vae_noise'] = \
            calc_mean_and_std_df(results_folder, tests_df,
                                 os.path.join('Tests', 'End percentage of std.csv'))

    return results, results_to_p_value
