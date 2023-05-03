from fpdf import FPDF
import os
import glob
import copy
import argparse
import pickle
import json
import pandas as pd
import numpy as np
from pdf_helper_functions import *
from collect_results import collect_statistical_results, file_name_escaping

# Globals.
figure_height = 45
figure_width = 45
trial_title_offset = 5

latent_space_path = os.path.join('Pictures', 'Latent space')
losses_path = os.path.join('Pictures', 'Losses')
real_space_path = os.path.join('Pictures', 'Real space')
statistics_path = os.path.join('Pictures', 'Densities Diagrams')

def parse_arguments():
    """
    Parse command line arguments
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-results', help='Path to results folder', dest='results', required=True)
    parser.add_argument('-out', help='Path to output PDF', dest='out', required=True)
    parser.add_argument('-header', help='Header of pdf', dest='header', required=True)
    parser.add_argument('--angles', help='Collect and print the angles between the perturbations',
                        dest='angles_between_perts', action="store_true")
    parser.add_argument('--vectors-noise', help='Collect and print the noise of decode-encode on perturbations vectors',
                        dest='vectors_noise', action="store_true")
    parser.add_argument('--traj', help='Collect and print the correlations between genes for each perturbation',
                        dest='traj', action="store_true")
    parser.add_argument('--random-traj', help='Collect and print the correlations between genes for each perturbation',
                        dest='random_traj', action="store_true")
    parser.add_argument('--tcga', help='Collect and print the tcga results',
                        dest='tcga', action="store_true")
    parser.add_argument('--confusion-table', help='Collect and print the confusion table results',
                        dest='confusion_table', action="store_true")
    parser.add_argument('--semi-supervised', help='Collect and print semi supervised results',
                        dest='semi_supervised', action="store_true")
    parser.add_argument('-data-path', help='Path to data df', dest='data_path')
    parser.add_argument('-info-path', help='Path to information df', dest='info_path')
    parser.add_argument('--vae-noise', help="Calculate and save the mean noise of passing through the system for genes",
                        dest='vae_noise', action='store_true')
    parser.add_argument('-compare', help='Path to results pickle file to compare', dest='compare')
    parser.add_argument('-naive', help='Path to results csv of naive', dest='naive')
    parser.add_argument('--no-std', help='Dont print std', dest='no_std', action='store_true')
    parser.add_argument('--sem', help='Print standart error of mean instead of std, number of runs that run',
                        dest='sem', default=None, type=int)

    parser.add_argument('--no-c-to-t', help='Remove c to t', dest='no_c_to_t', action='store_true')
    parser.add_argument('--no-t-to-t', help='Remove t to t', dest='no_t_to_t', action='store_true')
    parser.add_argument('--no-nearest-t', help='Remove nearest treated', dest='no_nearest_t', action='store_true')
    parser.add_argument('--no-compare', help='Remove compared', dest='no_compare', action='store_true')
    parser.add_argument('--no-naive', help='Remove naive', dest='no_naive', action='store_true')

    return parser.parse_args()


def PDF_create(header):
    pdf = FPDF()
    pdf.add_page(orientation='L')
    f_size = 40
    pdf.set_font("Courier", size=f_size, style='BU')
    pdf.multi_cell(0, f_size, txt=header, align="C")
    pdf.add_page(orientation='L')
    return pdf


def pdf_insert(pdf, row_offset, column_offset, image, height=figure_height, width=figure_width):
    """
    Insert picture into pdf
    :param pdf: pdf object
    :param row_offset: raw offset
    :param column_offset: column offset
    :param image: image to insert
    :param height: height of image
    :param width: width of image
    """
    # x,y is the up-left corner.
    x = 2 + (column_offset * width)
    y = 20 + trial_title_offset + (row_offset * height)
    pdf.image(image, x, y, w=width, h=height)


def pdf_print(pdf, out_path):
    """
    Save the pdf to file
    :param pdf: pdf object
    :param out_path: path to save
    """
    pdf.output(out_path, "F")
    print("Successfully made pdf file!")


def get_list_of_files(image_path):
    # create a list of files and sub directories names, in the given directory.
    all_files = list()
    # Iterate over all the entries.
    for entry in glob.glob(image_path):
        # Create full path.
        full_path = os.path.join(image_path, entry)
        if os.path.isdir(full_path):
            # If entry is a directory skip it.
            continue
        elif entry.endswith('.png'):
            all_files.append(entry)

    return all_files


def add_pictures_to_results_pdf(pdf, test_folder, row_offset):
    """
    Add pictures to pdf from single test folder
    :param pdf: fpdf object to add pictures to
    :param test_folder: folder of test to add pictures from
    :param row_offset: offset of row in fpdf object
    """
    # Print the figures of the test.
    curr_image_path = os.path.join(test_folder, latent_space_path,
                                   "End latent space classifier_labels.png")
    images_list = get_list_of_files(curr_image_path)[0]
    pdf_insert(pdf, row_offset, 0, images_list)

    curr_image_path = os.path.join(test_folder, latent_space_path, "End*calculated*")
    images_list = get_list_of_files(curr_image_path)[0]
    pdf_insert(pdf, row_offset, 1, images_list)

    curr_image_path = os.path.join(test_folder, real_space_path, "End*calculated*")
    images_list = get_list_of_files(curr_image_path)[0]
    pdf_insert(pdf, row_offset, 2, images_list)

    curr_image_path = os.path.join(test_folder, losses_path, "*")
    images_list = get_list_of_files(curr_image_path)[0]
    pdf_insert(pdf, row_offset, 3, images_list)

    curr_image_path = os.path.join(test_folder, statistics_path, "End * real space correlation*")
    images_list = get_list_of_files(curr_image_path)[0]
    pdf_insert(pdf, row_offset, 4, images_list)

    curr_image_path = os.path.join(test_folder, statistics_path, "End * real space distance*")
    images_list = get_list_of_files(curr_image_path)[0]
    pdf_insert(pdf, row_offset, 5, images_list)


def remove_unprintable_data(results_dict):
    """
    Remove unprintable keys from results dictionary
    :param results_dict: dictionary of results
    """
    results_dict['Arithmetic'].pop('statistics_results')


def create_results_pdf(pdf_header, results_folder, tests_df, output_path, info_path = None):
    """
    create pdf with all the tests results
    :param pdf_header: header of pdf
    :param results_folder: string, path to folder of results
    :param tests_df: DataFrame with all the tests
    :param output_path: path to save the pdf
    :param info_path: path to information DataFrame
    """
    pdf = PDF_create(pdf_header + ' Results')
    tmp_img = os.path.join(output_path, 'tmp{}{}.png')
    if info_path is not None:
        info_df = pd.read_csv(info_path)
    key_number = 0
    svm_results = {}
    perturbations_list = tests_df.perturbation.unique()
    tumors_list = list(tests_df.tumor.unique())
    empty_tumor_list = [None] * len(tumors_list)
    first_cloud = True
    configuration_content = ''
    for p in perturbations_list:
        svm_results[p] = dict(zip(tumors_list, empty_tumor_list))
        pert_tests = tests_df[tests_df.perturbation == p]
        for t in pert_tests.tumor.unique():
            # Add new page to each test
            if not first_cloud:
                pdf.add_page(orientation='L')
                key_number += 1
            else:
                first_cloud = False

            cloud_tests = pert_tests[pert_tests.tumor == t]
            row_offset = 0
            test_number = cloud_tests.iloc[0].test_number
            cloud_name = cloud_tests.iloc[0].tumor + ' ' + cloud_tests.iloc[0].perturbation
            title = "Test #{}: {}:".format(test_number, cloud_name)
            pdf.set_xy(10, 20 + (row_offset * figure_height))
            pdf.set_font("Courier", size=8, style='U')
            pdf.write(8, txt=title)
            row_offset += 0.1
            trial = 0
            for index, single_test in cloud_tests.iterrows():
                # Read repeat results
                repeat_folder = os.path.join(results_folder, '_'.join(single_test.map(str)))
                test_numeric = '{} {}'.format(single_test.test_number, single_test.repeat_number)
                test_results_json = os.path.join(repeat_folder, 'results.json')
                with open(test_results_json, 'r') as f:
                    results_dict = json.load(f)
                remove_unprintable_data(results_dict)
                results = json.dumps(results_dict)
                translation_table = dict.fromkeys(map(ord, r'{}[]"'), None)
                results = results.translate(translation_table)

                # Print title for the current test.
                title = "Trial #{}, results: {}:".format(str(trial + 1), results)
                pdf.set_xy(10, 20 + (row_offset * figure_height))
                pdf.set_font("Courier", size=6, style='U')
                pdf.write(5, txt=title)
                row_offset += 0.4

                # Add pictures to results pdf
                print("Adding pictures to (test number, repeat){}".format(test_numeric))
                add_pictures_to_results_pdf(pdf, repeat_folder, row_offset)
                row_offset += 1.2

                # Add next page if needed.
                if (trial + 1) % 2 == 0 and trial + 1 < cloud_tests.shape[0]:
                    pdf.add_page(orientation='L')
                    row_offset = 0
                trial += 1

                # Read configuration file if not read yet
                if configuration_content == '':
                    config_path = repeat_folder + "/configuration.txt"
                    with open(config_path, 'r') as content_file:
                        configuration_content = content_file.read()

            # Add empty page before svm results
            pdf.add_page(orientation='L')
            if info_path is not None:
                cloud_info_df = info_df[(info_df.perturbation == p) & (info_df.tumor.map(file_name_escaping) == t)]
            else:
                cloud_info_df = None

            row_offset = 0
            # Create svm success rates image
            current_tmp = tmp_img.format(key_number, "all")
            test_list = ['_'.join(row.map(str)) for _, row in cloud_tests.iterrows()]
            histograms_dict = create_tessellation_histogram(results_folder, test_list, cloud_info_df)
            all_histogram_np = histograms_dict.pop('all')
            print_horizontal_bar(all_histogram_np, current_tmp, 'All')
            pdf_insert(pdf, row_offset, 0, current_tmp, height=120, width=120)
            os.remove(current_tmp)
            if info_path is not None:
                pdf.add_page(orientation='L')
                column = 0
                row = 0
                index = 0
                number_of_doses = len(histograms_dict.values())
                for key, histogram_np in histograms_dict.items():
                    current_tmp = tmp_img.format(key_number, key)
                    print_horizontal_bar(histogram_np, current_tmp, 'Dose {}'.format(key))
                    pdf_insert(pdf, row, column, current_tmp, height=70, width=70)
                    os.remove(current_tmp)

                    column += 1
                    index += 1
                    if column == 3 and index + 1 < number_of_doses:
                        column = 0
                        row += 1
                        if row == 2:
                            pdf.add_page(orientation='L')
                            row = 0

            print("Finish test {}".format(cloud_tests.iloc[0].test_number))

    # Add the configuration file content to the end of the document.
    pdf.add_page(orientation='L')
    pdf.set_font("Courier", size=10, style='BU')
    pdf.cell(200, 3, txt="Configuration:", ln=1, align="L")
    pdf.set_font("Courier", size=6)
    pdf.write(3, configuration_content)

    pdf_print(pdf, os.path.join(output_path, 'results.pdf'))


def create_tumors_dict(tests_df):
    tumors_label = 'Tumors dictionary:\n\n'
    tumor_number = 0
    tumor_to_number = {}
    for t in tests_df.tumor.unique():
        tumors_label += '{}: {}\n\n'.format(tumor_number, t)
        tumor_to_number[t] = tumor_number
        tumor_number += 1
    return tumors_label, tumor_to_number


def create_drugs_dict(res_dict):
    """
    Create dict of drugs instead of dict of dicts
    :param res_dict: results dict
    :return: dict of format {"drug name": [all values tuple]}
    """
    out_dict = {}
    for drug, drug_dict in res_dict.items():
        mean_value = None
        to_divide = 0
        for tumor, value_list in drug_dict.items():
            # value_list is all the values of tumor, ie [[mean1, std1], [mean2, std2], ...]
            # Skip if default value
            if value_list[0][0] == 0 and (len(value_list[0]) == 0 or value_list[0][1] == 0):
                continue
            # Otherwise, append to mean values
            to_divide += 1
            if mean_value is None:
                mean_value = value_list
            else:
                for i in range(len(value_list)):
                    for j in range(len(value_list[i])):
                        mean_value[i][j] += value_list[i][j]
        for i in range(len(value_list)):
            for j in range(len(value_list[i])):
                mean_value[i][j] = mean_value[i][j] / to_divide
        out_dict[drug] = mean_value
    return out_dict


def add_bar_diagrams_from_results_dict(pdf, results_dictionary, tests_df, tmp_img_path, results_legend, default_value,
                                       histogram_y_label, histogram_perturbation_title, histogram_overall_title,
                                       y_ticks=None, dont_delete_pics=True, no_std = False, print_multi_bars=False,
                                       value_index=0, colors=None):
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
    :param no_std: If True, don't print std
    :param print_multi_bars: bool value.
    :param value_index: index of value in the dictionary to print in multi-bars mode and in drug summary
    :param colors: list of colors to bars
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
    column = 0
    row = 0
    number_of_perturbations = len(results_dictionary.keys())
    success_rates = np.zeros(shape=(number_of_perturbations, len(results_legend)))

    if print_multi_bars:
        fig_tuple = initiate_multi_subplots_figure(
            number_of_perturbations, results_legend[value_index])
    else:
        pdf.set_xy(215, 20)
        pdf.set_font("Courier", size=6, style='U')
        pdf.multi_cell(70, 2.5, txt=tumors_label)

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
                if print_multi_bars:
                    print_dict[tumor_to_number[t]] = [default_value[value_index]]
                else:
                    print_dict[tumor_to_number[t]] = default_value
            else:
                if print_multi_bars:
                    value = [value[value_index]]
                print_dict[tumor_to_number[t]] = value
                test = tests_df[(tests_df.perturbation == p) & (tests_df.tumor == t)]
                if test.shape[0] > 0:
                    test_number = test.iloc[0].test_number
                    all_tests_dict[int(test_number)] = value

        title_labels = mean_list
        labels = ['Cell ID', histogram_y_label, histogram_perturbation_title.format(*title_labels), results_legend, p]
        if print_multi_bars:
            print_bars_multi_figures(print_dict, fig_tuple, ind, labels[1], p, y_ticks=y_ticks, no_std=no_std)
            ind += 1
        else:
            xtables_ticks = [number_to_tumor[key] for key, _ in sorted(print_dict.items())]
            print_bars_figure(print_dict, tmp_img_path.format(p), labels, x_ticks=xtables_ticks, y_ticks=y_ticks,
                              no_std=no_std, colors=colors)
            pdf_insert(pdf, row, column, tmp_img_path.format(p) + '.png', height=70, width=70)
            if not dont_delete_pics:
                os.remove(tmp_img_path.format(p) + '.png')
            column += 1
            if column == 3 and ind + 2 < number_of_perturbations:
                column = 0
                row += 1
                if row == 2:
                    pdf.add_page(orientation='L')
                    row = 0

    if print_multi_bars:
        output_path = tmp_img_path.format('all_drugs')
        xtables_ticks = [number_to_tumor[key] for key, _ in sorted(print_dict.items())]
        finish_multi_subplots_figure(fig_tuple, xtables_ticks, labels[value_index], output_path)
        pdf_insert(pdf, row, column, output_path + '.png', height=150, width=150)
        if not dont_delete_pics:
            os.remove(output_path + '.png')

    global_mean = np.mean(success_rates, axis=0)
    labels = ['Test number', histogram_y_label, histogram_overall_title.format(*global_mean), results_legend, 'all']
    print("Add tester final histogram")
    print_bars_figure(all_tests_dict, tmp_img_path.format("summarize_bars"), labels, y_ticks=y_ticks, no_std=no_std,
                      colors=colors)
    pdf.add_page(orientation='L')
    pdf_insert(pdf, 0, 0, tmp_img_path.format("summarize_bars") + '.png', height=150, width=150)

    if print_multi_bars:
        tmp_res_dict = copy.deepcopy(results_dictionary)
        for i in range(len(list(list(tmp_res_dict.values())[0].values())[0])):
            if i == value_index:
                continue
            remove_unwanted_column(tmp_res_dict, 1)
        drugs_dict = create_drugs_dict(tmp_res_dict)
        labels = ['Drugs', histogram_y_label, "Overall rate {}".format(global_mean[value_index]), results_legend, 'all drugs']
        x_ticks = [key for key, _ in sorted(drugs_dict.items())]
        print_bars_figure(drugs_dict, tmp_img_path.format("summarize_drugs_bars"), labels, x_ticks=x_ticks, y_ticks=y_ticks,
                          no_std=no_std, colors=colors)

    if not dont_delete_pics:
        os.remove(tmp_img_path.format("summarize_bars") + '.png')


def print_table(pdf, header, df, to_sort=True, first_column_width_factor=6):
    """
    Print DataFrame to pdf
    :param pdf: pdf to write the table to
    :param header: header of table
    :param df: DataFrame to print
    :param to_sort: sort the df iff true
    :param first_column_width_factor: factor of width of first column against other columns
    """
    epw = pdf.w - 2 * pdf.l_margin
    df = df.sort_index()
    if to_sort:
        df = df.reindex(sorted(df.columns), axis=1)
    col_width = epw / (df.shape[1] + first_column_width_factor)
    pdf.set_font('Times', 'B', 15.0)
    pdf.cell(epw, 0.0, header, align='L')
    pdf.ln(10)
    pdf.set_font('Times', '', 10.0)

    # Text height is the same as current font size
    th = pdf.font_size

    # Print header row
    pdf.cell(col_width*first_column_width_factor, th, '', border=1)
    columns_line = df.columns
    for c in columns_line:
        if isinstance(c, float):
            s = '{0:.4f}'.format(c).rstrip('0').rstrip('.')
        else:
            s = str(c)
        pdf.cell(col_width, th, s, border=1)
    pdf.ln(th)

    # Print all the rows
    for i, row in df.iterrows():
        pdf.cell(col_width*first_column_width_factor, th, str(row.name), border=1)
        for v in row.values:
            if isinstance(v, float):
                s = '{0:.4f}'.format(v).rstrip('0').rstrip('.')
            else:
                s = str(v)
            pdf.cell(col_width, th, s, border=1)
        pdf.ln(th)


def remove_unwanted_column(results_dict, column):
    """
    Delete one column from dictionary
    :param results_dict: dictionary with columns
    :param column: column to delete
    """
    for drug_dict in results_dict.values():
        for v in drug_dict.values():
            del v[column]


def std_to_sem_dict(res_dict, num_of_returns):
    """
    Calculate standard mean of error instead of std
    :param res_dict: results dictionary
    :param num_of_returns: number of returs
    """
    sqrt_N = np.sqrt(num_of_returns)
    for pert, pert_dict in res_dict.items():
        for tumor, value in pert_dict.items():
            for i in range(len(value)):
                pert_dict[tumor][i][1] = pert_dict[tumor][i][1] / sqrt_N


def results_dict_to_csv(res_dict, fields, internal_fields, output_path):
    """
    save results dictionary as csv file
    :param res_dict: dictionary with results
    :param fields: list of fields to print
    :param internal_fields: list of 2nd level fields
    :param output_path: path to save the csv
    """
    columns = ['drug', "tumor"]
    for f in fields:
        for i_f in internal_fields:
            if i_f != '':
                columns.append(f + '_' + i_f)
            else:
                columns.append(f)

    df = pd.DataFrame(columns=columns)
    for drug, drug_dict in res_dict.items():
        for tumor, cloud_res in drug_dict.items():
            values = {'drug': drug, 'tumor': tumor}
            for i in range(len(fields)):
                f = fields[i]
                for j in range(len(internal_fields)):
                    i_f = internal_fields[j]
                    if i_f != '':
                        label = f + '_' + i_f
                    else:
                        label = f
                    values[label] = cloud_res[i][j]
            df = df.append(values, ignore_index=True)
    df.to_csv(output_path)


def create_statistics_pdf(args, results, tests_df):
    """
    create pdf with all the tests results
    :param args: arguments received from user
    :param results: dictionary of all the results for statistics pdf
    :param tests_df: DataFrame with all the results tumors and perturbations
    """
    pdf_header = args.header
    output_path = args.out
    svm_dict = results['svm']
    drop_samples_dict = results['drop_statistics']
    correlations_dict = results['correlations']
    distances_dict = results['distances']
    print_angles = False
    print_noise = False
    print_tcga_results = False
    print_confusion_table = False
    print_semi_supervised = False
    print_vae_noise = False
    if 'angles' in results:
        print_angles = True
        angles_results = results['angles']
    if 'vectors_noise' in results:
        print_noise = True
        pert_noise_results = results['vectors_noise']
    if 'TCGA' in results:
        print_tcga_results = True
        TCGA_results = results['TCGA']
    if 'confusion_table' in results:
        print_confusion_table = True
        confusion_table_results = results['confusion_table']
    if 'semi_supervised' in results:
        print_semi_supervised = True
        semi_supervised_results = results['semi_supervised']
    if 'vae_noise' in results:
        print_vae_noise = True
        vae_noise_results = results['vae_noise']

    second_label = 'std'
    if args.sem is not None:
        std_to_sem_dict(svm_dict, args.sem)
        std_to_sem_dict(drop_samples_dict, args.sem)
        std_to_sem_dict(correlations_dict, args.sem)
        std_to_sem_dict(distances_dict, args.sem)
        second_label = 'sem'

    # Create statistics pdf
    pdf = PDF_create(pdf_header + ' statistics')
    epw = pdf.w - 2 * pdf.l_margin

    # Print svm results
    print("Add histogram to SVM")
    pdf.set_font('Times', 'B', 20.0)
    pdf.cell(epw, 0.0, 'SVM results', align='C')
    statics_out_pics = os.path.join(output_path, "Statistics")
    if not os.path.isdir(statics_out_pics):
        os.makedirs(statics_out_pics)
    tmp_img_path = os.path.join(statics_out_pics, 'svm_{}')
    results_legend = ['Unfiltered result', 'Filtered result']
    default_value = [[0, 0], [0, 0]]
    histogram_y_label = 'rate %'
    histogram_perturbation_title = 'Overall rate: {0:.2f}%, filtered rate: {1:.2f}%'
    histogram_overall_title = 'Overall rate: {0:.2f}, filtered rate: {1:.2f}'
    y_ticks = range(0, 101, 20)
    add_bar_diagrams_from_results_dict(pdf, svm_dict, tests_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, histogram_perturbation_title,
                                       histogram_overall_title, y_ticks, no_std=args.no_std)
    pdf.add_page(orientation='L')
    results_dict_to_csv(svm_dict, ['unfiltered', 'filtered'], ['mean', second_label], os.path.join(statics_out_pics,
                                                                                                   "tesselation.csv"))

    print("Add multi-figure SVM results")
    tmp_img_path = os.path.join(statics_out_pics, 'multiple_svm_{}')
    add_bar_diagrams_from_results_dict(pdf, svm_dict, tests_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, histogram_perturbation_title,
                                       histogram_overall_title, y_ticks, no_std=args.no_std, print_multi_bars=True)
    pdf.add_page(orientation='L')
    print("Add multi-figure SVM filtered results")
    tmp_img_path = os.path.join(statics_out_pics, 'multiple_svm_filtered_{}')

    add_bar_diagrams_from_results_dict(pdf, svm_dict, tests_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, histogram_perturbation_title,
                                       histogram_overall_title, y_ticks, no_std=args.no_std, print_multi_bars=True,
                                       value_index=1)
    pdf.add_page(orientation='L')


    # Print dropped samples
    print("Add histogram to dropped samples")
    pdf.set_font('Times', 'B', 20.0)
    pdf.cell(epw, 0.0, 'Drop samples', align='C')
    tmp_img_path = os.path.join(statics_out_pics, 'drop_samples_{}')
    results_legend = ['Dropped samples percent']
    default_value = [[0, 0]]
    histogram_y_label = 'dropped %'
    histogram_perturbation_title = 'Dropped percent: {0:.2f}%'
    histogram_overall_title = 'Overall dropped percent: {0:.2f}%'
    y_ticks = range(0, 101, 20)
    add_bar_diagrams_from_results_dict(pdf, drop_samples_dict, tests_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, histogram_perturbation_title,
                                       histogram_overall_title, y_ticks, no_std=args.no_std)
    pdf.add_page(orientation='L')

    # Print correlations statistics
    print("Add histogram to correlations results")
    pdf.set_font('Times', 'B', 20.0)
    pdf.cell(epw, 0.0, 'Correlations results', align='C')
    tmp_img_path = os.path.join(statics_out_pics, 'correlations_{}')
    results_legend = []
    default_value = []
    colors = []
    histogram_perturbation_title = ' in real-space\n'
    histogram_overall_title = 'Overall:\n'

    addons_num = 0
    if args.no_c_to_t:
        remove_unwanted_column(correlations_dict, addons_num)
    else:
        results_legend.append('predicted to treated')
        default_value.append([0.0, 0.0])
        colors.append('b')
        histogram_perturbation_title += 'predicted to treated:{:.2f}'
        histogram_overall_title += 'predicted to treated:{:.2f}'
        addons_num += 1
    if args.no_t_to_t:
        remove_unwanted_column(correlations_dict, addons_num)
    else:
        results_legend.append('treated to treated')
        default_value.append([0.0, 0.0])
        colors.append('y')
        if addons_num % 2 == 1:
            histogram_perturbation_title += ', treated to treated:{:.2f}\n'
            histogram_overall_title += ', treated to treated:{:.2f}\n'
        else:
            histogram_perturbation_title += 'treated to treated:{:.2f}'
            histogram_overall_title += 'treated to treated:{:.2f}'
        addons_num += 1

    if not args.no_nearest_t and results['with_t_to_nearest']:
        default_value.append([0.0, 0.0])
        results_legend.append("treated to nearest")
        colors.append('g')
        if addons_num % 2 == 1:
            histogram_perturbation_title += ", treated to nearest:{:.2f}\n"
            histogram_overall_title += ", treated to nearest:{:.2f}\n"
        else:
            histogram_perturbation_title += "treated to nearest:{:.2f}"
            histogram_overall_title += "treated to nearest:{:.2f}"
        addons_num += 1

    elif args.no_nearest_t and results['with_t_to_nearest']:
        remove_unwanted_column(correlations_dict, addons_num)

    if not args.no_compare and results['with_compare']:
        default_value.append([0.0, 0.0])
        results_legend.append('One drug at a time')
        colors.append('r')
        if addons_num % 2 == 1:
            histogram_perturbation_title += ", one drug at a time:{:.2f}\n"
            histogram_overall_title += ", one drug at a time:{:.2f}\n"
        else:
            histogram_perturbation_title += "one drug at a time:{:.2f}"
            histogram_overall_title += "one drug at a time:{:.2f}"
        addons_num += 1
    elif args.no_compare and results['with_compare']:
        remove_unwanted_column(correlations_dict, addons_num)

    if not args.no_naive and results['with_naive']:
        default_value.append([0.0, 0.0])
        results_legend.append('naive predicted to treated')
        colors.append('r')
        if addons_num % 2 == 1:
            histogram_perturbation_title += ", naive predicted to treated:{:.2f}\n"
            histogram_overall_title += ", naive predicted to treated:{:.2f}\n"
        else:
            histogram_perturbation_title += "naive predicted to treated:{:.2f}"
            histogram_overall_title += "naive predicted to treated:{:.2f}"
        addons_num += 1
    elif args.no_naive and results['with_naive']:
        remove_unwanted_column(correlations_dict, addons_num)
    histogram_perturbation_title = histogram_perturbation_title.rstrip('\n')
    histogram_overall_title = histogram_perturbation_title.rstrip('\n')
    correlation_histogram_perturbation_title = "Correlation" + histogram_perturbation_title
    correlation_histogram_overall_title = "Correlation" + histogram_overall_title
    histogram_y_label = 'correlation'
    y_ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    add_bar_diagrams_from_results_dict(pdf, correlations_dict, tests_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, correlation_histogram_perturbation_title,
                                       correlation_histogram_overall_title, y_ticks, no_std=args.no_std,
                                       colors=colors)
    pdf.add_page(orientation='L')

    # Print distance statistics
    print("Add histogram to distances results")
    pdf.set_font('Times', 'B', 20.0)
    pdf.cell(epw, 0.0, 'Distance results', align='C')
    tmp_img_path = os.path.join(statics_out_pics, 'distances_{}')

    addons_num = 0
    if args.no_c_to_t:
        remove_unwanted_column(distances_dict, addons_num)
    else:
        addons_num += 1
    if args.no_t_to_t:
        remove_unwanted_column(distances_dict, addons_num)
    else:
        addons_num += 1

    if not args.no_nearest_t and results['with_t_to_nearest']:
        addons_num += 1
    elif args.no_nearest_t and results['with_t_to_nearest']:
        remove_unwanted_column(distances_dict, addons_num)

    if not args.no_compare and results['with_compare']:
        addons_num += 1
    elif args.no_compare and results['with_compare']:
        remove_unwanted_column(distances_dict, addons_num)

    if not args.no_naive and results['with_naive']:
        addons_num += 1
    elif args.no_naive and results['with_naive']:
        remove_unwanted_column(distances_dict, addons_num)

    distance_histogram_perturbation_title = "Distance" + histogram_perturbation_title
    distance_histogram_overall_title = "Distance" + histogram_overall_title
    histogram_y_label = 'distance'
    y_ticks = [2.5, 3, 4, 5, 6, 7]
    add_bar_diagrams_from_results_dict(pdf, distances_dict, tests_df, tmp_img_path, results_legend,
                                       default_value, histogram_y_label, distance_histogram_perturbation_title,
                                       distance_histogram_overall_title, y_ticks, no_std=args.no_std, colors=colors)

    if print_angles:
        pdf.add_page(orientation='L')
        # Print angles matrices
        print("Add angles between matrices")
        angles_mean_df, angles_std_df = angles_results

        pdf.set_font('Times', 'B', 20.0)
        pdf.cell(epw, 0.0, 'Angles between perturbations', align='C')
        pdf.ln(10)
        print_table(pdf, 'Mean matrix', angles_mean_df)
        pdf.ln(60)
        print_table(pdf, 'STD matrix', angles_std_df)

    if print_noise:
        pdf.add_page(orientation='L')

        # Print perturbations vectors noise matrices
        print("Add perturbations vectors noise matrices")
        pert_noise_mean_df, pert_noise_std_df = pert_noise_results

        pdf.set_font('Times', 'B', 20.0)
        pdf.cell(epw, 0.0, 'Euclidean distance after decode-encode for each neuron', align='C')
        pdf.ln(10)
        print_table(pdf, 'Mean matrix', pert_noise_mean_df)

        pdf.ln(60)
        print_table(pdf, 'STD matrix', pert_noise_std_df)

    if print_tcga_results:
        pdf.add_page(orientation='L')

        # Print perturbations vectors noise matrices
        print("Add TCGA results")
        tcga_mean_df, tcga_std_df = TCGA_results

        pdf.set_font('Times', 'B', 20.0)
        pdf.cell(epw, 0.0, 'TCGA classification success, only by tumor', align='C')
        pdf.ln(10)
        print_table(pdf, 'Mean matrix', tcga_mean_df)

        pdf.ln(60)
        print_table(pdf, 'STD matrix', tcga_std_df)

    if print_confusion_table:
        # Print perturbations vectors noise matrices
        print("Add confusion table results")
        conf_svm_mean_df, conf_svm_std_df, conf_class_mean_df, conf_class_std_df = confusion_table_results

        pdf.add_page(orientation='L')
        pdf.set_font('Times', 'B', 20.0)
        pdf.cell(epw, 0.0, 'Confusion table by SVM', align='C')
        pdf.ln(10)
        print_table(pdf, 'Mean matrix', conf_svm_mean_df, to_sort=False)

        pdf.ln(60)
        print_table(pdf, 'STD matrix', conf_svm_std_df, to_sort=False)

        pdf.add_page(orientation='L')
        pdf.set_font('Times', 'B', 20.0)
        pdf.cell(epw, 0.0, 'Confusion table by System classifier', align='C')
        pdf.ln(10)
        print_table(pdf, 'Mean matrix', conf_class_mean_df, to_sort=False)

        pdf.ln(60)
        print_table(pdf, 'STD matrix', conf_class_std_df, to_sort=False)

    if print_semi_supervised:
        # Print semi supervised results
        pdf.add_page(orientation='L')
        pdf.set_font('Times', 'B', 20.0)
        pdf.cell(epw, 0.0, 'Semi supervised by tessellation', align='C')
        pdf.ln(10)
        print_table(pdf, '', semi_supervised_results.set_index('Test tumor'), to_sort=False, first_column_width_factor=3)

    # Save the pdf to disk
    pdf_print(pdf, os.path.join(output_path, 'statistics.pdf'))


def save_results_dictionaries(results, results_to_p_value, output_folder):
    """
    Save results to disk
    :param results: results tuple
    :param results_to_p_value: results to save to p value
    :param angles_df: mean angles between pertubations
    :param output_folder: folder to save results to
    """
    with open(os.path.join(output_folder, 'results.p'), 'wb') as fp:
        pickle.dump(results, fp, protocol=4)
    results_to_p_value.to_csv(os.path.join(output_folder, 'tests.csv'))


def load_results_dictionaries(file_to_load):
    """
    Load results from disk
    :param file_to_load: path of file to load
    :return: results tuple
    """
    with open(file_to_load, 'rb') as fp:
        results = pickle.load(fp)
    return results


def create_tests_df(results_folder):
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
    return tests_df


def load_tests_df(saved_path):
    """
    Load tests DataFrame from results folder
    :param saved_path: path to the folder in which it saved
    :return: DataFrame with all tests
    """
    column_types = {'repeat_number': int, 'test_number': int, 'tumor': str, 'perturbation': str,
                    'svm': float, 'correlations': float, 'correlations_nearest': float,
                    'distances': float, 'distances_nearest': float}
    return pd.read_csv(saved_path, dtype=column_types)


def main():
    args = parse_arguments()
    tests_df = create_tests_df(args.results)
    if args.compare is not None:
        compare_results = load_results_dictionaries(args.compare)
    else:
        compare_results = None
    #tests_df = load_tests_df(r'd:\tests.csv')

    #create_results_pdf(args.header, args.results, tests_df, args.out, args.info_path)
    results, results_to_p_value = collect_statistical_results(args.results, tests_df, args.angles_between_perts,
                                                              args.vectors_noise, args.tcga, args.confusion_table,
                                                              args.semi_supervised, args.data_path, args.info_path,
                                                              compare_results, args.naive, args.vae_noise)
    save_results_dictionaries(results, results_to_p_value, args.out)
    #results = load_results_dictionaries(r'd:\results.p')

    create_statistics_pdf(args, results, tests_df)


if __name__ == '__main__':
    main()
