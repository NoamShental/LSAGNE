import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def print_horizontal_bar(histogram_np, output_path, title):
    """
    Print histogram to file.
    :param histogram_np: 1D array with histogram values
    :param output_path: path to save the figure
    :param title: title for the histogram
    """
    fig, ax = plt.subplots()
    plt.barh(range(histogram_np.shape[0]), histogram_np, 1)
    number_of_samples = np.sum(histogram_np)
    ax.set_xlabel('Number of samples out of {}'.format(number_of_samples))
    ax.set_ylabel('Number of correct answer out of {}'.format(histogram_np.shape[0] - 1))
    ax.set_title('{} svm success rates'.format(title))
    plt.yticks(range(histogram_np.shape[0]))
    for i, v in enumerate(histogram_np):
        ax.text(v + 3, i - 0.1, str(v), color='K', fontweight='bold')
    plt.savefig(output_path)
    plt.close('All')


def print_bars_figure(results_dict, output_path, labels, x_ticks=None, y_ticks=None, no_std=False,
                      colors=None, hatches=None):
    """
    Print bars figure.
    :param results_dict: dictionary of {label:number} or {label:list of numbers}
    :param output_path: path to save the figure
    :param labels: tuple of 4 strings and array: x label, y label, array of legends and figure title,
    :param x_ticks: set of ticks for x axis
    :param y_ticks: set of ticks for y axis
    :param no_std: don't print std even if it given
    :param colors: list of colors for the bars
    :param hatches: list of hatches for the bars.
    """
    dict_len = len(results_dict)
    number_of_bars = len(list(results_dict.values())[0])
    width = 0.8 if labels[0] != "Drugs" else 0.4
    width_per_bar = width / number_of_bars
    first_bar_position = -(width / 2)
    fig, ax = plt.subplots()

    def get_color_and_hatch(index):
        color = None if colors is None else colors[index]
        hatch = None if hatches is None else hatches[index]
        return color, hatch

    for i in range(number_of_bars):
        left_bar = np.around(first_bar_position + i*width_per_bar, 4)
        right_bar = np.around(dict_len + first_bar_position + i*width_per_bar, 4)

        # check if the data have std to print
        std = None
        if isinstance(list(results_dict.values())[0][i], list):
            values = np.array([value[i][0] for _, value in sorted(results_dict.items())])
            values[values == 0] = np.nan
            if not no_std:
                std = pd.Series([value[i][1] for _, value in sorted(results_dict.items())])
                std[std == -10] = np.nan
        else:
            values = np.array([x[1][i] for x in sorted(results_dict.items())])
            values[values == 0] = np.nan
        color, hatch = get_color_and_hatch(i)
        plt.bar(np.arange(left_bar, right_bar, 1), values, yerr=std, width=width_per_bar,
                align='edge', color=color, hatch=hatch)
    if x_ticks is None:
        plt.xticks(range(dict_len), [key for key, _ in sorted(results_dict.items())], color='k', size=6)
    elif labels[0] == "Drugs":
        plt.xticks(range(dict_len), x_ticks, fontweight='semibold', size=7, rotation=17, color='red',
                   horizontalalignment='right')
    else:
        plt.xticks(range(dict_len), x_ticks, fontweight='semibold', size=9)
    plt.xlim((-0.5, dict_len))

    if y_ticks is not None:
        plt.yticks(y_ticks)
        plt.ylim((y_ticks[0], y_ticks[-1]))
    #ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1], fontweight='semibold', fontsize=12)
    if labels[4] != 'all':
        # Print the label to the right of the figure.
        ax.text(1.02, .5, labels[4], verticalalignment='center', rotation=90, transform=ax.transAxes,
                color='red', fontsize=12, fontweight='semibold')

    ax.set_title(labels[2], fontsize=12)
    if labels[0] != "Drugs":
       ax.legend(labels[3])
    plt.savefig(output_path + '.eps', format='eps')
    plt.savefig(output_path + '.png')
    plt.close('All')

def create_tessellation_histogram(results_folder, trials_list, cloud_info_df):
    """
    Create and save histogram to one cloud, with results of svm over all iterations
    :param results_folder: root folder of results
    :param trials_list: list of sub folders with the trials
    :param cloud_info_df: DataFrame with info of treated cloud
    :return: numpy array of histogram of values
    """
    results_df = pd.DataFrame(columns=['success'])
    number_of_trials = len(trials_list)
    latent_space_hdf_path = os.path.join('Tests', 'svm_latent_space_results.h5')
    summarize_np = None

    # Read all the trials results, and save them in numpy arrays
    for i in range(number_of_trials):
        hdf_path = os.path.join(results_folder, trials_list[i], latent_space_hdf_path)
        success_df = pd.read_hdf(hdf_path, 'df')
        if summarize_np is None:
            summarize_np = np.zeros(shape=[success_df.shape[0], len(trials_list)])

        summarize_np[:, i] = success_df['predicted'] == success_df['real']

    # Fill up results DataFrame
    results_df['success'] = summarize_np.sum(axis=1)
    all_histogram_np = np.histogram(results_df['success'], bins=range(number_of_trials + 2))[0]
    histograms_dict = {'all': all_histogram_np}
    if cloud_info_df is not None:
        results_df.index = cloud_info_df.index
        for dose in cloud_info_df.pert_dose.unique():
            dose_results_df = results_df.loc[cloud_info_df[cloud_info_df.pert_dose == dose].index]
            histograms_dict[dose] = np.histogram(dose_results_df['success'], bins=range(number_of_trials + 2))[0]
    return histograms_dict


def initiate_multi_subplots_figure(subplots_num, title):
    """
    :param subplots_num: number of subplots in the figure
    :param title: title of subplot
    """
    fig, ax = plt.subplots(subplots_num, figsize=(10, 10))
    ax[0].text(.5, 1.08, title, horizontalalignment='center', transform=ax[0].transAxes,
            fontsize=13, fontweight='semibold')
    return fig, ax


def finish_multi_subplots_figure(fig_tuple, xtables_ticks, fig_x_label, save_path):
    """
    Save and close multi subplots figure
    :param fig_tuple: tuple of figure
    :param xtables_ticks: ticks of xtable
    :param save_path: path to save the figure
    """
    _, axis = fig_tuple
    axis[-1].tick_params(labelbottom=True)
    axis[-1].set_xticklabels(xtables_ticks, ha='center', fontweight='semibold', fontsize=12)
    # axis[-1].set_xlabel(fig_x_label)
    plt.savefig(save_path + '.eps', format='eps')
    plt.savefig(save_path + '.png')
    plt.close('All')


def print_bars_multi_figures(results_dict, fig_tuple, ind, subfig_y_label, subfig_header,
                             y_ticks=None, no_std=False):
    """
    Print bars figure.
    :param results_dict: dictionary of {label:number} or {label:list of numbers}
    :param fig_tuple: axis of the current drug.
    :param ind: index of the current drug.
    :param subfig_y_label: label of current sub-figure.
    :param subfig_header: text of current sub-figure.
    :param y_ticks: required y ticks (if any).
    :param no_std: If set, don't print std even if given
    """
    fig, axis = fig_tuple
    axis[ind].set_ylabel(subfig_y_label, fontweight='semibold', fontsize=12)
    axis[ind].text(1.01, .5, subfig_header, verticalalignment='center', rotation=90, transform=axis[ind].transAxes,
                   color='red', fontsize=11, fontweight='semibold')
    dict_len = len(results_dict)
    width = 0.4
    first_bar_position = -(width / 2)

    left_bar = np.around(first_bar_position, 4)
    right_bar = np.around(dict_len + first_bar_position, 4)

    # check if the data have std to print
    std = None
    if isinstance(list(results_dict.values())[0][0], list):
        values = np.array([value[0][0] for _, value in sorted(results_dict.items())])
        values[values == 0] = np.nan
        if not no_std:
            std = pd.Series([value[0][1] for _, value in sorted(results_dict.items())])
            std[std == 0] = np.nan
    else:
        values = np.array([x[1][0] for x in sorted(results_dict.items())])
        values[values == 0] = np.nan
    axis[ind].bar(np.arange(left_bar, right_bar, 1), values, yerr=std, width=width, align='edge')

    axis[ind].set_xticks(range(dict_len))
    axis[ind].tick_params(labelbottom=False)

    if y_ticks is not None:
        axis[ind].set_yticks(y_ticks, minor=False)
        axis[ind].set_ylim((y_ticks[0], y_ticks[-1]))
    axis[ind].set_xlim((-0.5, dict_len))
