import argparse
import pandas as pd
from scipy.stats import ranksums


def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--first', help='Path to first results pickle file', dest='first', required=True)
    parser.add_argument('--second', help='Path to first results pickle file', dest='second', required=True)
    parser.add_argument('--out', help='path to out csv', dest='out',required=True)
    return parser.parse_args()


def calculate_all_p_values(first_df, second_df, columns):
    results_columns = ['tumor', 'perturbation']
    results_columns.extend(columns)
    results_df = pd.DataFrame(columns=results_columns)

    for p in first_df.perturbation.unique():
        p_repeats = first_df[first_df.perturbation == p]
        for t in p_repeats.tumor.unique():
            first_repeats_df = p_repeats[p_repeats.tumor == t]
            second_repeats_df = second_df[(second_df.perturbation == p) & (second_df.tumor == t)]
            series = pd.Series(index=results_columns)
            series.tumor = t
            series.perturbation = p
            for test_column in columns:
                cloud_t_value, cloud_p_value = \
                    ranksums(first_repeats_df[test_column].values, second_repeats_df[test_column].values)
                series[test_column] = cloud_p_value
            results_df = results_df.append(series, ignore_index=True)

    return results_df


def main():
    args = parse_arguments()
    first_results = pd.read_csv(args.first)
    second_results = pd.read_csv(args.second)
    test_columns = ['svm', 'correlations', 'correlations_nearest', 'distances', 'distances_nearest']
    results_df = calculate_all_p_values(first_results, second_results, test_columns)
    results_df.to_csv(args.out)


if __name__ == '__main__':
    main()
