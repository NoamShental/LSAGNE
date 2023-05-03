import argparse
import os
import string
import pandas as pd


def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-results', help='Path to results folder', dest='results', required=True)
    parser.add_argument('-out', help='Path to output folder', dest='out', required=True)
    parser.add_argument('--traj', help='Collect and print the correlations between genes for each perturbation',
                        dest='traj', action="store_true")
    parser.add_argument('--random-traj', help='Collect and print the correlations between genes for each perturbation',
                        dest='random_traj', action="store_true")
    parser.add_argument('-tumor', help='Whitelist tumor to collect from')
    parser.add_argument('-drug', help='Whitelist drug to collect from')
    return parser.parse_args()


def create_tests_df(results_folder, args):
    """
    Create DataFrame with all the tests list, from path to results folder
    :param results_folder: path to results folder
    :return: DataFrame with all tests in that folder
    """
    sub_folders_list = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    tests = [x.split('_') for x in sub_folders_list]
    tests_df = pd.DataFrame(tests, columns=['repeat_number', 'test_number', 'tumor', 'perturbation'])
    tests_df["repeat_number"] = pd.to_numeric(tests_df["repeat_number"])
    tests_df["test_number"] = pd.to_numeric(tests_df["test_number"])
    tests_df.sort_values(['test_number', 'repeat_number'], inplace=True)
    if args.tumor:
        tests_df = tests_df[tests_df.tumor.str.match(args.tumor)]
    if args.drug:
        tests_df = tests_df[tests_df.perturbation == args.drug]
    return tests_df


def file_name_escaping(filename):
    """
    Make escaping for file names (i.e: omit '|' or '\'.
    :param filename: filename to escape
    :return: escaped filename
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


def calc_correlation_over_genes(results_folder, tests_df, decoded_df_path, encoded_df_path):
    """
    Calculate correlations for each pair of genes, for each cloud
    :param results_folder:  folder of results
    :param tests_df: DataFrame of tests
    :param decoded_df_path: path to decoded df in each result folder
    :param encoded_df_path: path to encoded df in each result folder
    :return: 1 matrix with correlations between each pair of genes
    """
    perturbations_to_gather = list(tests_df.perturbation.unique())
    full_df = None
    full_encoded_df = None
    for p in perturbations_to_gather:
        print('Gather correlation to {}'.format(p))
        pert_tests = tests_df[tests_df.perturbation == p]
        for t in pert_tests.tumor.unique():
            cloud_tests = pert_tests[pert_tests.tumor == t]
            for index, single_test in cloud_tests.iterrows():
                repeat_folder = os.path.join(results_folder, file_name_escaping('_'.join(single_test.map(str))))
                hdf_path = os.path.join(repeat_folder, decoded_df_path)
                current_df = pd.read_hdf(hdf_path, 'df')
                if full_df is None:
                    full_df = current_df
                else:
                    full_df = full_df.append(current_df)

                hdf_path = os.path.join(repeat_folder, encoded_df_path)
                current_df = pd.read_hdf(hdf_path, 'df')
                if full_encoded_df is None:
                    full_encoded_df = current_df
                else:
                    full_encoded_df = full_encoded_df.append(current_df)

    # Escape the tumor index (second level) in order to be the same as tests_df
    full_df.index = pd.MultiIndex.from_tuples([(x[0], file_name_escaping(x[1]), x[2]) for x in full_df.index])
    full_df = full_df.sort_index()
    genes_corr_df_dict = {}
    for p in full_df.index.get_level_values(0).unique():
        pert_samples = full_df.loc[p]
        tumors = pert_samples.index.get_level_values(0).unique()
        for t in tumors:
            t = file_name_escaping(t)
            print('Calculate correlation to {} {}'.format(p, t))
            cloud_df = full_df.loc[p, t]
            cloud_df = cloud_df.sample(n=min(cloud_df.shape[0], 2000))
            genes_corr_df_dict[p + '_' + t] = cloud_df.corr()
    return genes_corr_df_dict, full_df, full_encoded_df


def collect_trajectories(result_path, tests_df, collect_traj, collect_random_traj):
    results = {}
    if collect_traj:
        results['trajectories'] =\
            calc_correlation_over_genes(result_path, tests_df,
                                        os.path.join('trajectories_tests', 'decoded.h5'),
                                        os.path.join('trajectories_tests', 'encoded.h5'))
    if collect_random_traj:
        results['random_trajectories'] = \
            calc_correlation_over_genes(result_path, tests_df,
                                        os.path.join('trajectories_tests', 'random_decoded.h5'),
                                        os.path.join('trajectories_tests', 'random_encoded.h5'))
    return results


def save_trajectories(results, output_path):

    if 'trajectories' in results:
        # Save genes correlations matrices
        traj_df_dict, decoded_df, encoded_df = results['trajectories']
        matrices_dir = os.path.join(output_path, 'trajectories')
        if not os.path.isdir(matrices_dir):
            os.makedirs(matrices_dir)
        for key, df in traj_df_dict.items():
            file_name = key + '_trajectories.csv'
            df.to_csv(os.path.join(matrices_dir, file_name))
        decoded_df.to_hdf(os.path.join(output_path, 'decoded.h5'), 'df')
        encoded_df.to_hdf(os.path.join(output_path, 'encoded.h5'), 'df')

    if 'random_trajectories' in results:
        random_traj_df_dict, random_decoded_df, random_encoded_df = results['random_trajectories']
        # Save genes correlations matrices
        matrices_dir = os.path.join(output_path, 'random_trajectories')
        if not os.path.isdir(matrices_dir):
            os.makedirs(matrices_dir)
        for key, df in random_traj_df_dict.items():
            file_name = key + '_random_trajectories.csv'
            df.to_csv(os.path.join(matrices_dir, file_name))
        random_decoded_df.to_hdf(os.path.join(output_path, 'decoded.h5'), 'df')
        random_encoded_df.to_hdf(os.path.join(output_path, 'encoded.h5'), 'df')


def main():
    args = parse_arguments()
    tests_df = create_tests_df(args.results, args)
    results = collect_trajectories(args.results, tests_df, args.traj, args.random_traj)
    save_trajectories(results, args.out)


if __name__ == '__main__':
    main()
