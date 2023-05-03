from configuration import config
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity
from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import numpy as np
import string
import os


class PrintingHandler:
    """
    Class to handle all the printing of graphs and figures
    """

    def __init__(self, model, data):
        """
        Initializer, save all parameters
        :param model: model to predict needed spaces
        :param data: data to show
        """
        self.model = model
        self.data = data
        self.default_output_folder = config.config_map['pictures_folder']
        self.plot_limits = [[-10, 10], [-10, 10]]  # list of limits, [[-X limit, +X limit], [-Y limit, +Y limit]]
        self.arrows_width = min(20, 20) / 300
        self.plot_kws = {"s": 80,  # size of markers
                         "alpha": 1}  # Color alpha
        if not os.path.isdir(self.default_output_folder):
            os.makedirs(self.default_output_folder)

    def set_printing_plot_limits(self, data_to_print_df):
        """
        Setting plotting limit in the printer object
        :param data_to_print_df: All the data that we are going to print, in 2D.
        """
        # Get maximum and minimum of the data.
        max_values = data_to_print_df.max(axis=0)
        x_max = max_values.iloc[0]
        y_max = max_values.iloc[1]
        min_values = data_to_print_df.min(axis=0)
        x_min = min_values.iloc[0]
        y_min = min_values.iloc[1]

        # Add 10% in each direction
        x_delta = x_max - x_min
        y_delta = y_max - y_min
        x_pad = x_delta / 10
        x_max += x_pad
        x_min -= x_pad
        y_pad = y_delta / 10
        y_max += y_pad
        y_min -= y_pad

        if data_to_print_df.shape[1] == 3:
            z_max = max_values.iloc[2]
            z_min = min_values[2]
            z_delta = z_max - z_min
            z_pad = z_delta / 10
            z_max += z_pad
            z_min -= z_pad

            self.plot_limits = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            self.arrows_width = min(x_max - x_min, y_max - y_min, z_max - z_min) / 300
        else:
            self.plot_limits = [[x_min, x_max], [y_min, y_max]]

            # Set arrows size
            self.arrows_width = min(x_max - x_min, y_max - y_min) / 300

    @staticmethod
    def _file_name_escaping(filename):
        """
        Make escaping for file names (i.e: omit '|' or '\'.
        :param filename: filename to escape
        :return: escaped filename
        """
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        return ''.join(c for c in filename if c in valid_chars)

    def losses_plot(self, data_df, data_columns, colors_dict, figure_name, figure_title=None, output_directory=None):
        """
        Plot losses DataFrame.
        :param data_df: DataFrame for ploting.
        :param data_columns: columns data to print.
        :param colors_dict: dictionary of colors to print
        :param figure_title: the title on the figure.
        :param figure_name: figure file name.
        :param output_directory: dictionary to put the figure in..
        """
        # Set current output directory, and make sure this directory exists.
        if output_directory is None:
            current_output = self.default_output_folder
        else:
            current_output = output_directory
        if not os.path.isdir(current_output):
            os.makedirs(current_output)

        ax = plt.gca()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("log2 of loss")

        for column_name in data_columns:
            if column_name == "loss":
                data_df.plot(x=data_df.index.name, y=column_name, kind='line', style=':', linewidth=1.0, color='black',
                             title=figure_title, ax=ax)
            else:
                data_df.plot(x=data_df.index.name, y=column_name, kind='line', linewidth=0.8,
                             color=colors_dict[column_name], title=figure_title, ax=ax)

        figure = ax.get_figure()
        file_path = os.path.join(current_output, figure_name)
        figure.savefig(file_path)
        figure.savefig(file_path + '.eps', format='eps')
        plt.close('all')

    def sns_plot(self, data_df, info_df, column_names, figure_name, figure_title="", output_directory=None, color_dict=None,
                 markers=None, vectors_to_print=None, show_legend=True):
        """
        create seaborn plot from a 2-dimensional DataFrame
        :param data_df: data DataFrame of the plot
        :param info_df: information DataFrame to take separation columns from
        :param column_names: list of column names, that the plotted dots will separate by.
        :param figure_name: name of figure
        :param figure_title: figure title
        :param output_directory: directory to save the figure at.
        :param color_dict: dictionary of colors to print.
        :param markers: list of markers style.
        :param vectors_to_print: list of vectors to print on the image
        :param show_legend: boolean or list in size of column_names, if True print legend of figure
        """
        if data_df.shape[1] == 3:
            dimension_3 = True
        else:
            dimension_3 = False

        # Set current output directory, and make sure this directory exists.
        if output_directory is None:
            current_output = self.default_output_folder
        else:
            current_output = output_directory
        if not os.path.isdir(current_output):
            os.makedirs(current_output)

        data_df = data_df.copy()

        # If show legend is bool - move it to list in size of column_names
        if isinstance(show_legend, bool):
            show_legend = [show_legend] * len(column_names)

        # Create figure to each seperation column
        for i in range(len(column_names)):
            separation_column = column_names[i]
            data_df[separation_column] = info_df[separation_column]
            unique_values = data_df[separation_column].unique()
            colors = cm.gist_rainbow(np.linspace(0, 1, len(unique_values)))
            # Create figure
            if show_legend[i]:
                fig = plt.figure(figsize=[12, 10])
            else:
                fig = plt.figure(figsize=[10, 10])

            # Create axs
            if dimension_3:
                axs = fig.add_subplot(111, projection='3d')
            else:
                axs = fig.add_subplot(111)

            for value, c in zip(unique_values, colors):
                samples = data_df[data_df[separation_column] == value]
                marker = None
                if color_dict is not None and value in color_dict:
                    c = color_dict[value]
                if markers is not None and value in markers:
                    marker = markers[value]
                if dimension_3:
                    axs.scatter(samples[0], samples[1], zs=samples[2], s=90, color=c, marker=marker, edgecolors='w', label=value)
                else:
                    axs.scatter(samples[0], samples[1], s=90, color=c, marker=marker, edgecolors='w', label=value)

            # Print vectors if asked to
            if vectors_to_print:
                for vector in vectors_to_print:
                    plt.arrow(vector[0][0], vector[0][1], vector[1][0], vector[1][1], width=self.arrows_width)

            # Cosmetics
            axs.set_xlim(self.plot_limits[0])
            axs.set_ylim(self.plot_limits[1])
            if dimension_3:
                axs.set_zlim(self.plot_limits[2])

            if show_legend[i]:
                axs.legend(frameon=False)
            axs.spines['right'].set_visible(False)
            axs.spines['top'].set_visible(False)
            fig.gca().set_aspect('auto', adjustable='box')
            axs.set_title(figure_title)

            # Save the figure
            figure_full_name = '{0} {1}'.format(self._file_name_escaping(figure_name),
                                                self._file_name_escaping(
                                                    separation_column))
            file_path = os.path.join(current_output, figure_full_name)
            fig.savefig(file_path)
            fig.savefig(file_path + '.eps', format='eps')
            plt.close(fig)

    @staticmethod
    def do_tsne(data_df):
        """
        Do 2 dimensional TSNE to DataFrame, if the given data already in 2D, just return it.
        :param data_df: DataFrame with data to do the tsne
        :return: DataFrame with the data after TSNE
        """
        if len(data_df.columns) < 4:
            return data_df

        tsne_function = TSNE(n_components=2,
                             random_state=0,
                             perplexity=20,
                             learning_rate=300,
                             n_iter=400,
                             n_jobs=4)
        tsne_out_np = tsne_function.fit_transform(data_df)
        tsne_out_df = pd.DataFrame(tsne_out_np, columns=[0, 1])
        tsne_out_df.index = data_df.index
        tsne_out_df.index.name = data_df.index.name

        return tsne_out_df

    def vectors_plot(self, vectors_df, info_df, figure_name, output_directory=None):
        """
        plot the perturbation vectors
        :param vectors_df: data DataFrame of the plot
        :param info_df: information of the data
        :param figure_name: figure title.
        :param output_directory: directory to save the figure at.
        """
        # Set current output directory, and make sure this directory exists.
        if output_directory is None:
            output_directory = self.default_output_folder
        else:
            output_directory = output_directory
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        unique_perturbations = info_df['perturbation'].unique()
        for perturbation in unique_perturbations:
            # Clean the plotter
            plt.close('all')
            current_axes = plt.axes()
            if perturbation in config.config_map['untreated_labels']:
                current_perturbation_info_df = info_df[(info_df['perturbation'] == perturbation) &
                                                       (~info_df['pert_time'].isin(config.config_map['untreated_times']))]
            else:
                current_perturbation_info_df = info_df[info_df['perturbation'] == perturbation]
            current_perturbation_data_np = vectors_df[vectors_df.index.isin(current_perturbation_info_df.index)].values
            for i in range(0, current_perturbation_data_np.shape[0]):
                current_axes.arrow(0, 0, current_perturbation_data_np[i, 0], current_perturbation_data_np[i, 1], head_width=0.02,
                                   head_length=0.05,
                                   color='b')

            plt.plot(0, 0, 'ok')  # Plot a black point at the origin.
            plt.axis('scaled')  # Set the axes to the same scale.
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.suptitle('{0} {1}'.format(figure_name, perturbation), size=10)
            file_path = os.path.join(output_directory, '{0} for pert {1}.png'.format(self._file_name_escaping(figure_name),
                                                                            self._file_name_escaping(perturbation)))
            plt.savefig(file_path)
            plt.close('all')

    @staticmethod
    def calculate_perturbation_vectors_latent_space(predicted_data_df, predicted_reference_points):
        """
        Calculate perturbation vectors in latent space - perturbation, and perturbations and time
        :param predicted_data_df: data in latent space
        :param predicted_reference_points: reference points in latent space
        :return: tuple of calculated perturbations and time vectors, and only perturbation vectors
        """
        # Calculate the perturbation vectors in the latent space.
        control_start_df, ref_treated_df, ref_control_start_df, control_pert_df, ref_control_pert_df = \
            predicted_reference_points[0:5]

        # Calculate pert and time vectors
        pert_and_time_vectors_df = predicted_data_df - control_start_df
        ref_pert_and_time_vectors_df = ref_treated_df - ref_control_start_df
        pert_and_time_cosine_similarity_table = cosine_similarity(pert_and_time_vectors_df.values,
                                                                  ref_pert_and_time_vectors_df.values)
        pert_and_time_angles_np = np.diagonal(pert_and_time_cosine_similarity_table)

        # Rotate vectors
        pert_and_time_arrows_np = np.zeros(shape=[pert_and_time_angles_np.shape[0], 2])
        for i in range(pert_and_time_angles_np.shape[0]):
            angle = pert_and_time_angles_np[i]
            rotation_matrix = [[np.cos(angle), -(np.sin(angle))],
                               [np.sin(angle), np.cos(angle)]]
            pert_and_time_arrows_np[i] = np.dot([1, 1], rotation_matrix)

        pert_and_time_vectors_df = pd.DataFrame(pert_and_time_arrows_np, index=predicted_data_df.index,
                                                columns=['x', 'y'])

        # Calculate pert vectors
        pert_vectors_df = predicted_data_df - control_pert_df
        ref_pert_vectors_df = ref_treated_df - ref_control_pert_df
        pert_cosine_similarity_table = cosine_similarity(pert_vectors_df.values,
                                                         ref_pert_vectors_df.values)
        pert_angles_np = np.diagonal(pert_cosine_similarity_table)

        # Rotate vectors
        pert_arrows_np = np.zeros(shape=[pert_angles_np.shape[0], 2])
        for i in range(pert_angles_np.shape[0]):
            angle = pert_angles_np[i]
            rotation_matrix = [[np.cos(angle), -(np.sin(angle))],
                               [np.sin(angle), np.cos(angle)]]
            pert_arrows_np[i] = np.dot([1, 1], rotation_matrix)

        pert_vectors_df = pd.DataFrame(pert_arrows_np, index=predicted_data_df.index, columns=['x', 'y'])

        return pert_and_time_vectors_df, pert_vectors_df

    def print_original_and_decoded(self, data_df, info_df, column_to_separate, is_decode_column, figures_prefix):
        """
        Create figures for each cloud, with it's original samples and that samples after decode.
        :param data_df: DataFrame with the values.
        :param info_df: information DataFrame.
        :param column_to_separate: column to separate to different figures.
        :param is_decode_column: column that say if the row is decoded or original.
        :param figures_prefix: prefix of figures to print.
        """
        # We're going to change that DataFrame, so make a copy of it.
        info_df = info_df.copy()

        # Create tuple of original and decoded classifier labels
        separation_labels = info_df[column_to_separate].unique()
        classifier_labels_list = []
        for label in separation_labels:
            classifier_labels_list.append((label, 'decoded_' + label))

        # Add "decoded_" to the start of separation label for each decoded label
        decoded_indexes = info_df[info_df[is_decode_column] == 1].index
        info_df.loc[decoded_indexes, [column_to_separate]] =\
            'decoded_' + info_df.loc[decoded_indexes, [column_to_separate]]

        # For each tuple, create figure
        for labels_tuple in classifier_labels_list:
            indexes = info_df[info_df[column_to_separate].isin(labels_tuple)].index
            cloud_data_df = data_df.reindex(indexes)
            name, _ = labels_tuple
            name = self._file_name_escaping(name)
            self.sns_plot(cloud_data_df, info_df,
                          [column_to_separate],
                          figures_prefix + ' ' + name,
                          output_directory=os.path.join(self.default_output_folder, 'Before and after'))

    def print_distribution(self, observations_dictionary, figure_name, output_directory=None):
        """
        Print distribution of observations.
        :param observations_dictionary: dictionary of observations, each key is name of observation, and the value is the observation samples.
        :param figure_name: name of figure to print
        :param output_directory: output directory to save the figure.
        """
        # Set current output directory, and make sure this directory exists.
        if output_directory is None:
            output_directory = self.default_output_folder
        else:
            output_directory = output_directory
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        for name, samples in observations_dictionary.items():
            sns.distplot(samples, hist=False, label=name)

        plt.savefig(os.path.join(output_directory, self._file_name_escaping(figure_name)))
        plt.close('all')

    def plot_decision_boundary(self, prediction_function, data_to_get_boundaries_df, real_labels, save_path):
        """
        Print the decision boundary for given prediction function
        This will work only to 2d data.
        :param prediction_function: function that get numpy array, and get prediction to it.
        :param data_to_get_boundaries_df: data to get it's boundaries.
        :param real_labels: real labels of the data, a DataFrame series
        :param save_path: where to save the pictures
        """
        # Set min and max values and give it some padding
        x_min, x_max = data_to_get_boundaries_df[0].min() - .5, data_to_get_boundaries_df[0].max() + .5
        y_min, y_max = data_to_get_boundaries_df[1].min() - .5, data_to_get_boundaries_df[1].max() + .5
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        labels_predicted = prediction_function(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=data_to_get_boundaries_df.columns))
        Z = labels_predicted.reshape(xx.shape)
        levels = np.unique(Z).shape[0]

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, levels, cmap=plt.get_cmap('plasma'))
        plt.contour(xx, yy, Z, levels, colors='black', linewidths=0.3)
        plt.scatter(data_to_get_boundaries_df[0], data_to_get_boundaries_df[1], c=real_labels,
                    cmap=plt.get_cmap('plasma'), linewidths=0.3, edgecolors='black')

        if not os.path.isdir(self.default_output_folder):
            os.makedirs(self.default_output_folder)

        file_path = os.path.join(self.default_output_folder, save_path)
        plt.savefig(file_path)
        plt.close('all')
