import os
import pandas as pd
import numpy as np
from pdf_creator import create_tumors_dict
from pdf_helper_functions import initiate_multi_subplots_figure, \
                                    finish_multi_subplots_figure, \
                                    print_bars_figure
WORK_PATH = "d://tes"
config = {
    'show_standard_error_of_mean': True,
    'tesselation_5_base_path': os.path.join(WORK_PATH, 'tesselation 5 base.csv'),
    'tesselation_8_path': os.path.join(WORK_PATH, 'tesselation 5+3.csv'),
    'basic_perturbations': ["geldanamycin", "raloxifene", "vorinostat", "trichostatin-a", "wortmannin"],
    'new_perturbations': ['sirolimus', 'isonicotinohydroxamic-acid', 'estriol']
}

def get_color_and_hatch(colors, hatches, index):
    color = None if colors is None else colors[index]
    hatch = None if hatches is None else hatches[index]
    return color, hatch


def print_bars_multi_figures(results_dict, fig_tuple, ind, labels, subfig_header,
                             y_ticks=None, colors=None, hatches=None):
    """
    Print bars figure.
    :param results_dict: dictionary of {label:number} or {label:list of numbers}
    :param fig_tuple: axis of the current drug.
    :param ind: index of the current drug.
    :param labels: labels of current sub-figure.
    :param subfig_header: text of current sub-figure.
    :param y_ticks: required y ticks (if any).
    """
    fig, axis = fig_tuple
    axis[ind].set_ylabel(labels[1], fontweight='semibold', fontsize=12)
    axis[ind].text(1.01, .5, subfig_header, verticalalignment='center', rotation=90, transform=axis[ind].transAxes,
                   color='red', fontsize=11, fontweight='semibold')
    dict_len = len(results_dict)
    number_of_bars = len(list(results_dict.values())[0])
    width = 0.2
    if number_of_bars == 2:
        first_bar_position = -(width)
    else:
        first_bar_position = -(width / 2)
    width_per_bar = width / number_of_bars*2

    for i in range(number_of_bars):
        left_bar = np.around(first_bar_position + i * width_per_bar, 4)
        right_bar = np.around(dict_len + first_bar_position + i * width_per_bar, 4)

        # Check if the data have std to print.
        std = None
        if isinstance(list(results_dict.values())[0][i], list):
            values = np.array([value[i][0] for _, value in sorted(results_dict.items())])
            values[values == 0] = np.nan
            if config['show_standard_error_of_mean']:
                std = pd.Series([value[i][1] for _, value in sorted(results_dict.items())])
                std[std == -10] = np.nan
        else:
            values = np.array([x[1][i] for x in sorted(results_dict.items())])
            values[values == 0] = np.nan

        color, hatch = get_color_and_hatch(colors, hatches, i)
        axis[ind].bar(np.arange(left_bar, right_bar, 1), values, yerr=std, width=width,
                                 align='edge', color=color, hatch=hatch)

    axis[ind].set_xticks(range(dict_len))
    axis[ind].tick_params(labelbottom=False)

    if y_ticks is not None:
        axis[ind].set_yticks(y_ticks, minor=False)
        axis[ind].set_ylim((y_ticks[0], y_ticks[-1]))
    axis[ind].set_xlim((-0.5, dict_len))
    if ind == 0 and labels[3][0] is not None:
            axis[ind].legend(labels[3])


def add_bar_diagrams_from_results_dict(results_dictionary, tests_df, tmp_img_path, results_legend, default_value,
                                       histogram_y_label, histogram_perturbation_title, fig_title, y_ticks=None,
                                       dont_delete_pics=True, print_multi_bars=False, colors=None, hatches=None):
    """
    add diagrams of results dictionaries to pdf
    :param pdf: fpdf object
    :param results_dictionary: dictionaries with the results to save, in format of
            {perturbation: {tumor: [[mean 1, std 1 (optional)], [mean 2, std 2 (optional)],...]}}
    :param tests_df: DataFrame with list of tests
    :param tmp_img_path: path to save tmp images (should contain {} to format)
    :param results_legend: list of labels for each of the results in the dictionaries,
            must be in the same length as number of results.
    :param default_value: value for cloud that does not exists
    :param histogram_y_label: y label of histogram
    :param histogram_perturbation_title: title of each perturbation histogram
    :param histogram_overall_title: title for overall histogram
    :param y_ticks: ticks for y axis in the histograms
    :param dont_delete_pics: If True, dont delete the statistics pics
    :param print_multi_bars: bool value.
    :param multi_bars_value_index: index of value in the dictionary to print in multi-bars mode.
    :param colors: list of colors to bars.
    """
    # Print tumors table
    tumors_label, tumor_to_number = create_tumors_dict(tests_df)
    number_to_tumor = {val: key.split(' ')[0] for (key, val) in tumor_to_number.items()}

    # Check if the data has std or not:
    if isinstance(list(list(results_dictionary.values())[0].values())[0][0], list):
        has_std = True
    else:
        has_std = False

    all_tests_dict = {}

    number_of_perturbations = len(results_dictionary.keys())
    success_rates = np.zeros(shape=(number_of_perturbations, len(results_legend)))

    if print_multi_bars:
        fig_tuple = initiate_multi_subplots_figure(number_of_perturbations, fig_title)

    # Print histogram for each perturbation
    for ind, (p, t_dict) in enumerate(sorted(results_dictionary.items())):
        print_dict = {}
        mean_list = []
        for i in range(len(results_legend)):
            if has_std:
                result_mean_to_perturbation = np.mean([x[i][0] for x in t_dict.values() if x is not None and x != default_value])
            else:
                result_mean_to_perturbation = np.mean([x[i] for x in t_dict.values() if x is not None and x != default_value])
            mean_list.append(result_mean_to_perturbation)
            success_rates[ind, i] = result_mean_to_perturbation

        # Add empty tests and all the tests to global tests bar
        for t, value in t_dict.items():
            if value is None:
                print_dict[tumor_to_number[t]] = default_value
            else:
                print_dict[tumor_to_number[t]] = value
                test = tests_df[(tests_df.drug == p) & (tests_df.tumor == t)]
                if test.shape[0] > 0:
                    test_number = test.iloc[0].test_number
                    all_tests_dict[int(test_number)] = value

        title_labels = mean_list
        labels = ['Cell ID', histogram_y_label, histogram_perturbation_title.format(*title_labels), results_legend, p]
        if print_multi_bars:
            print_bars_multi_figures(print_dict, fig_tuple, ind, labels, p, y_ticks=y_ticks, colors=colors, hatches=hatches)
            ind += 1
        else:
            xtables_ticks = [number_to_tumor[key] for key, _ in sorted(print_dict.items())]
            print_bars_figure(print_dict, tmp_img_path.format(p), labels, x_ticks=xtables_ticks,
                              y_ticks=y_ticks, colors=colors, hatches=hatches)

            if not dont_delete_pics:
                os.remove(tmp_img_path.format(p) + '.png')

    if print_multi_bars:
        output_path = tmp_img_path.format('all_drugs')
        xtables_ticks = [number_to_tumor[key] for key, _ in sorted(print_dict.items())]
        finish_multi_subplots_figure(fig_tuple, xtables_ticks, labels[0], output_path)
        if not dont_delete_pics:
            os.remove(output_path + '.png')


def print_figure(data_df, tmp_img_path):
    data_dict = {}
    for p in sorted(data_df.drug.unique()):
        pert_tests_df = data_df[data_df.drug == p]
        tumor_dict = {}
        for t in pert_tests_df.tumor.unique():
            cloud_tests = pert_tests_df[pert_tests_df.tumor == t]
            unfiltered_list = [cloud_tests.iloc[0].unfiltered_mean, cloud_tests.iloc[0].unfiltered_sem]
            filtered_list = [cloud_tests.iloc[0].filtered_mean, cloud_tests.iloc[0].filtered_sem]
            tumor_dict[t] = [unfiltered_list]
        data_dict[p] = tumor_dict

    results_legend = [None]
    default_value = [[0, 0], [0, 0]]
    colors = ['#1f77b4']
    hatches = ["...."]
    histogram_perturbation_title = 'Overall rate: {0:.2f}%'
    histogram_y_label = 'rate %'
    fig_title = "Unfilterd results of three new drugs"
    y_ticks = range(20, 101, 20)
    add_bar_diagrams_from_results_dict(data_dict, data_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, histogram_perturbation_title,
                                       fig_title, y_ticks, print_multi_bars=True, colors = colors, hatches=hatches)


def print_sum_all_cells_figure(data_df, tmp_img_path):
    data_dict = {}
    for p in sorted(data_df.drug.unique()):
        pert_tests_df = data_df[data_df.drug == p]
        unfiltered_mean_acc = 0
        num_of_cell_lines = 0
        for t in pert_tests_df.tumor.unique():
            cloud_tests = pert_tests_df[pert_tests_df.tumor == t]
            unfiltered_list = [cloud_tests.iloc[0].unfiltered_mean, cloud_tests.iloc[0].unfiltered_sem]
            unfiltered_mean_acc += unfiltered_list[0]
            if unfiltered_list[0] > 0 :
                num_of_cell_lines += 1
        data_dict[p] = [unfiltered_mean_acc/num_of_cell_lines]

    histogram_y_label = 'rate %'
    histogram_perturbation_title = 'Overall rate: {0:.2f}%'
    results_legend = [None]
    title_labels = np.array(list(data_dict.values())).mean()
    labels = ['Drugs', histogram_y_label, histogram_perturbation_title.format(title_labels), results_legend, "all"]
    xticks = data_dict.keys()
    xticks = [w.replace('-acid', "\n-acid") for w in xticks]
    y_ticks = range(20, 101, 20)
    colors = ['#1f77b4']
    hatches = ["...."]
    print_bars_figure(data_dict, tmp_img_path.format("all_drugs"), labels, x_ticks=xticks,
                      y_ticks=y_ticks, colors=colors, hatches=hatches)


def main():

    # Load 5 base drugs (stand alone).
    print("Load and print 5 basic drugs...")
    tesselation_5_base_path = config['tesselation_5_base_path']
    tesselation_5_base_df = pd.read_csv(tesselation_5_base_path)

    # Load 8 (5 basic + 3 new) drugs.
    print("Load and print 8 drugs (5 basics + 3 new)...")
    tesselation_8_path = config['tesselation_8_path']
    tesselation_8_df = pd.read_csv(tesselation_8_path)
    # Rermove new drugs from the data.
    tesselation_5_over_8_df = tesselation_8_df[~tesselation_8_df['drug'].isin(config['new_perturbations'])]

    tmp_img_path = os.path.join(WORK_PATH, 'tesselation_8_{}.png')
    print_sum_all_cells_figure(tesselation_8_df, tmp_img_path)

    # Load and print 3 new drugs.
    tesselation_3_over_8_df = tesselation_8_df
    for drug in config['basic_perturbations']:
        tesselation_3_over_8_df = tesselation_3_over_8_df[tesselation_3_over_8_df.drug != drug]
    tmp_img_path = os.path.join(WORK_PATH, 'tesselation_3_over_8.png')
    print_figure(tesselation_3_over_8_df, tmp_img_path)

    # Print 5 basics drugs only  Vs. 5 basics over 8.
    drug_list = tesselation_5_base_df.drug.unique()
    tumor_list = tesselation_5_base_df.tumor.unique()
    tesselation_5_vs_5_dict = {}
    for p in drug_list:
        tumor_dict = {}
        for t in tumor_list:
            five_over_8_cloud_tests_df = tesselation_5_over_8_df.loc[(tesselation_5_over_8_df.drug == p) &
                                                                     (tesselation_5_over_8_df.tumor == t)]
            five_over_8_unfiltered_list = [five_over_8_cloud_tests_df.iloc[0].unfiltered_mean,
                                           five_over_8_cloud_tests_df.iloc[0].unfiltered_sem]
            five_basic_cloud_tests_df = tesselation_5_base_df.loc[(tesselation_5_base_df.drug == p) &
                                                                     (tesselation_5_base_df.tumor == t)]
            five_basic_unfiltered_list = [five_basic_cloud_tests_df.iloc[0].unfiltered_mean,
                                          five_basic_cloud_tests_df.iloc[0].unfiltered_sem]

            tumor_dict[t] = [five_basic_unfiltered_list, five_over_8_unfiltered_list]
        tesselation_5_vs_5_dict[p] = tumor_dict

    tmp_img_path = os.path.join(WORK_PATH, '{}_five_only_vs_five_plus')
    results_legend = ['Five drugs only', 'Five drugs plus new drugs']
    default_value = [[0, 0], [0, 0]]
    colors = ['#1f77b4', '#1f77b4']
    hatches = [None, "...."]

    histogram_y_label = 'rate %'
    histogram_perturbation_title = 'Five drugs only rate: {0:.2f}%, Five drugs plus new drugs rate: {1:.2f}%'
    y_ticks = range(20, 101, 20)
    fig_title = "Unfilterd results of five basic drugs"

    add_bar_diagrams_from_results_dict(tesselation_5_vs_5_dict, tesselation_5_over_8_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, histogram_perturbation_title, fig_title, y_ticks,
                                       colors=colors, hatches=hatches, print_multi_bars=True)


if __name__ == '__main__':
    main()
