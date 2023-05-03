from configuration import config
import logging
import os
from tessellation_tester import TessellationTester
from scipy.spatial.distance import cdist


def get_closest_cloud(data_df, info_df, test_data_df, measure_function, compare_function):
    """
    Get distance of closest cloud from given samples
    :param data_df:  all data with the clouds
    :param info_df:  info, to separate from cloud
    :param test_data_df: test samples
    :param measure_function: function to calculate the closeness between each 2 numbers in the given arrays
    :param compare_function: function that accepts 2 numbers, and decide which better
    :return: tuple of closest cloud tumor, perturbation and time, and distance
    """
    best_measure_a_to_a = -1
    best_cloud_a_to_a = ''
    best_measure_top = -1
    best_cloud_top = ''
    for t in info_df.tumor.unique():
        t_info_df = info_df[info_df.tumor == t]
        for p in t_info_df.perturbation.unique():
            p_info_df = t_info_df[t_info_df.perturbation == p]
            for pert_time in p_info_df.pert_time.unique():
                cloud_info_df = p_info_df[p_info_df.pert_time == pert_time]
                cloud_data_df = data_df.loc[cloud_info_df.index]
                measure_matrix = measure_function(cloud_data_df.values, test_data_df.values)
                current_measure = measure_matrix.mean()
                nearest_samples = compare_function(measure_matrix.max(axis=0), measure_matrix.min(axis=0))
                nearest_samples_mean = nearest_samples.mean()
                if best_measure_a_to_a == -1 or compare_function(current_measure, best_measure_a_to_a):
                    best_measure_a_to_a = current_measure
                    best_cloud_a_to_a = '{} {} {}'.format(t, p, pert_time)
                if best_measure_top == -1 or compare_function(nearest_samples_mean, best_measure_top):
                    best_measure_top = nearest_samples_mean
                    best_cloud_top = '{} {} {}'.format(t, p, pert_time)

    return best_cloud_a_to_a, best_measure_a_to_a, best_cloud_top, best_measure_top


def calculate_dmso_distances(data_df, info_df, reference_points, test_data_df, test_info_df,
                             latent_space_arithmetic, participating_tumors, output_csv_path):
    """
    Calculate Distance to DMSO
    :param data_df: data set of samples in latent space
    :param info_df: information of data_df
    :param reference_points: reference points for the data set
    :param test_data_df: DataFrame with test points
    :param test_info_df: DataFrame with test info
    :param latent_space_arithmetic: latent space arithmetic class
    :param participating_tumors: list of participating tumors
    :param output_csv_path: path to output csv
    """
    with open(output_csv_path, 'w') as f:
        f.write('Test tumor,Test perturbation,Distance from vector,Nearest cloud,Distance from nearest cloud\n')
    for t in participating_tumors:
        tumor_info_df = test_info_df[test_info_df.tumor == t]
        for p in config.config_map['untreated_labels']:
            control_info_df = tumor_info_df[tumor_info_df.perturbation == p]
            if control_info_df.shape[0] == 0:
                continue
            pert_ant_time_vector, pert_vector = latent_space_arithmetic.get_vectors(p, reference_points)
            start_time_control_cloud = info_df[(info_df.tumor == t) &
                                               (info_df.pert_time.isin(config.config_map['untreated_times']))]
            start_time_cloud_ref_point = reference_points[5].loc[[start_time_control_cloud.iloc[0].name]]
            first_point = latent_space_arithmetic.model.predict_latent_space(start_time_cloud_ref_point).values
            second_point = first_point + pert_vector
            distance = distance_and_correlation_from_point_to_line(test_data_df.loc[control_info_df.index],
                                                     (first_point, second_point))
            logging.info('Calculated distance DMSO from time vector:' + str(distance))
            nearest_cloud, smallest_distance, _, _ = get_closest_cloud(data_df, info_df, test_data_df, cdist)
            logging.info('Nearest cloud: {}, with distance {}'.format(nearest_cloud, smallest_distance))
            # Write results to csv
            with open(output_csv_path, 'a') as f:
                f.write('{0},{1},{2:.2f},{3},{4:.2f}\n'.format(t.replace(',', ' '), p, distance,
                                                               nearest_cloud.replace(',', ' '), smallest_distance))


def compare_perturbation_distances(data_df, info_df, reference_points, test_data_df, test_info_df,
                                   latent_space_arithmetic, output_folder_path):
    """
    Calculate Distance to test samples, from perturbation vector, and compared to all other clouds, for
    each cloud in test data and info
    :param data_df: data set of samples in latent space
    :param info_df: information of data_df
    :param reference_points: reference points for the data set
    :param test_data_df: DataFrame with test points
    :param test_info_df: DataFrame with test info
    :param latent_space_arithmetic: latent space arithmetic class
    :param output_folder_path: path to output folder
    :return: tuple of calculated encoded data, decoded data and info_df
    """
    all_calculated_encoded_data_df = None
    all_calculated_decoded_data_df = None
    all_calculated_info_df = None
    test_info_df = test_info_df.copy()
    test_info_df.numeric_labels.fillna(-1, inplace=True)
    curr_numeric_label = max(info_df.numeric_labels.max(), test_info_df.numeric_labels.max()) + 1
    for t in test_info_df.tumor.unique():
        tumor_info_df = test_info_df[test_info_df.tumor == t]
        for p in test_info_df.perturbation.unique():
            pert_info_df = tumor_info_df[tumor_info_df.perturbation == p]
            for pert_time in pert_info_df.pert_time.unique():
                cloud_info_df = pert_info_df[pert_info_df.pert_time == pert_time]
                if cloud_info_df.iloc[0].numeric_labels == -1:
                    test_info_df.loc[cloud_info_df.index, 'numeric_labels'] = curr_numeric_label
                    curr_numeric_label += 1
    svm_path = os.path.join(output_folder_path, 'svm.csv')
    with open(svm_path, 'w') as f:
        f.write(
            'Test tumor,Test perturbation,Number of DMSO0 Samples,Number of DMSO24 samples,Number of test samples,Treated accuracy\n')

    for t in test_info_df.tumor.unique():
        tumor_info_df = test_info_df[test_info_df.tumor == t]
        for p in tumor_info_df.perturbation.unique():
            logging.info('Calculated distance and correlation for {0} {1}'.format(t, p))
            cloud_info_df = tumor_info_df[tumor_info_df.perturbation == p]

            control_info_df = info_df[(info_df.tumor == t) &
                                      (info_df.perturbation.isin(config.config_map['untreated_labels']))]
            control_data_df = data_df.loc[control_info_df.index]
            start_time_control_info_df = control_info_df[
                control_info_df.pert_time.isin(config.config_map['untreated_times'])]
            base_samples_df = control_data_df.loc[start_time_control_info_df.index]

            calculated_df, decoded_calculated_df = \
                latent_space_arithmetic.calculate_perturbations_effect_for_given_data(base_samples_df,
                                                                                      control_info_df, p,
                                                                                      reference_points)

            # SVM test
            calculated_info_df = start_time_control_info_df.copy()
            calculated_info_df.index = 'calculated_' + p + '_' + calculated_info_df.index
            calculated_info_df.perturbation = cloud_info_df.iloc[0].perturbation
            calculated_info_df.pert_time = cloud_info_df.iloc[0].pert_time
            calculated_info_df.numeric_labels = cloud_info_df.iloc[0].numeric_labels
            calculated_df.index = 'calculated_' + p + '_' + calculated_df.index
            decoded_calculated_df.index = 'calculated_' + p + '_' + decoded_calculated_df.index

            # Save calculated and decoded to TSNE at the end
            if all_calculated_encoded_data_df is None:
                all_calculated_encoded_data_df = calculated_df
            else:
                all_calculated_encoded_data_df = all_calculated_encoded_data_df.append(calculated_df, sort=False)
            if all_calculated_decoded_data_df is None:
                all_calculated_decoded_data_df = decoded_calculated_df
            else:
                all_calculated_decoded_data_df = all_calculated_decoded_data_df.append(decoded_calculated_df,
                                                                                       sort=False)
            if all_calculated_info_df is None:
                all_calculated_info_df = calculated_info_df
            else:
                all_calculated_info_df = all_calculated_info_df.append(calculated_info_df, sort=False)
    data_to_fit_df = data_df[~data_df.index.isin(test_data_df.index)].append(all_calculated_encoded_data_df)
    info_to_fit_df = info_df[~info_df.index.isin(test_info_df.index)].append(all_calculated_info_df)
    logging.info("Start tessellation tester fit")
    tessellation_tester = TessellationTester()
    tessellation_tester.fit(data_to_fit_df, info_to_fit_df)
    train_accuracy, _ = tessellation_tester.get_accuracy(data_to_fit_df, info_to_fit_df)
    logging.info("Global tessellation tester train accuracy: {0:.2f}".format(train_accuracy))
    with open(svm_path, 'a') as f:
        for t in test_info_df.tumor.unique():
            tumor_info_df = test_info_df[test_info_df.tumor == t]
            control_samples_df = info_df[info_df.tumor == t]
            DMSO_0_number = control_samples_df[control_samples_df.pert_time.isin(config.config_map['untreated_times'])].shape[0]
            DMSO_24_number = control_samples_df.shape[0] - DMSO_0_number
            for p in tumor_info_df.perturbation.unique():
                cloud_info_df = tumor_info_df[tumor_info_df.perturbation == p]
                cloud_data_df = test_data_df.loc[cloud_info_df.index]
                treated_accuracy, _ = tessellation_tester.get_accuracy(cloud_data_df, cloud_info_df)
                f.write('{0},{1},{2},{3},{4},{5:.2f}\n'.format(
                    t.replace(',', ' '), p, DMSO_0_number, DMSO_24_number, cloud_info_df.shape[0], treated_accuracy))
    return all_calculated_encoded_data_df, all_calculated_decoded_data_df, all_calculated_info_df


def save_test_result(output_file, result_line, logging_level=logging.INFO):
    """
    Save result of test to output folder and log
    :param output_file: path to output file to save the result
    :param result_line: line of result to save
    :param logging_level: level of result logging, default to INFO
    """
    logging.log(logging_level, result_line)
    with open(output_file, 'at') as f:
        f.write(result_line + '\n')


def print_figures(printer, data_df, info_df, calculated_info_df, state, output_folder,
                  color_dict=None):
    """
    For each calculated cloud, print it with it's control clouds
    :param printer: printer handler
    :param data_df: DataFrame with all the data
    :param info_df: DataFrame with info of real samples
    :param calculated_info_df: DataFrame with info of calculated data
    :param state: state (latent space, decode, etc.
    :param output_folder: folder to save the figures
    :param color_dict: dictionary with colors for arithmetic column
    """
    # Do TSNE
    logging.info("{} start TSNE".format(state))
    data_df = printer.do_tsne(data_df)
    logging.info("{} end TSNE".format(state))

    printer.set_printing_plot_limits(data_df)

    # Copy info CF before changing it
    calculated_info_df = calculated_info_df.copy()
    info_df = info_df.copy()

    # Set arithmetic:
    # calculated ->calculated + pert time
    # fitted -> control/treated + pert time
    # test -> treated + pert time
    calculated_info_df['arithmetic'] = "calculated " + calculated_info_df['pert_time'].map(str)
    control_idx = info_df.perturbation.isin(config.config_map['untreated_labels'])
    info_df.loc[control_idx, 'arithmetic'] = "control " + info_df.loc[control_idx, 'pert_time'].map(str)
    info_df.loc[~control_idx, 'arithmetic'] = "treated " + info_df.loc[~control_idx, 'pert_time'].map(str)
    for t in calculated_info_df.tumor.unique():
        tumor_calculated_info_df = calculated_info_df[calculated_info_df.tumor == t]
        for p in tumor_calculated_info_df.perturbation.unique():

            # Take info of control + current perturbation
            cloud_info_df = info_df[(info_df.tumor == t) &
                                    (info_df.perturbation.isin([p] + config.config_map['untreated_labels']))]
            # Take info of current perturbation
            cloud_calculated_info_df = tumor_calculated_info_df[tumor_calculated_info_df.perturbation == p]
            print_info_df = cloud_info_df.append(cloud_calculated_info_df, sort=False)
            print_data_df = data_df.loc[print_info_df.index]

            # Print by arithmetic
            printer.sns_plot(print_data_df,
                             print_info_df,
                             ['arithmetic'],
                             '{0} {1} {2} calculated'.format(state, t, p),
                             color_dict=color_dict,
                             output_directory=output_folder)
