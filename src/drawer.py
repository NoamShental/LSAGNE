import os

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray

from src.scatter_data_2d import ScatterData2D, ScatterData3D


class Drawer:

    def __init__(self, logger, folder_path):
        self.logger = logger
        self.folder_path = folder_path

    def _save_figure_to_file_if_needed(self, file_name):
        if file_name:
            plt.savefig(os.path.join(self.folder_path, file_name), bbox_inches='tight')

    def plot_3d_scatter(self, array, display_labels, specific_display_label_colors=None, title=None, file_name=None):
        scatter_data = ScatterData3D(array[:, 0], '0',
                                     array[:, 1], '1',
                                     array[:, 2], '2',
                                     display_labels)
        # sns.color_palette("tab10", len(scatter_data.unique_labels))
        if not specific_display_label_colors:
            color_palette = sns.color_palette("tab10", len(scatter_data.unique_labels))
        else:
            color_palette = {}
            tab10 = sns.color_palette("tab10")
            i = 0
            for display_label in scatter_data.unique_labels:
                if display_label in specific_display_label_colors:
                    color_palette[display_label] = matplotlib.colors.to_rgb(
                        sns.crayons[specific_display_label_colors[display_label]])
                else:
                    color_palette[display_label] = tab10[i % 10]
                    i += 1

        if 'OTHER' in display_labels:
            indices0 = [i for i, s in enumerate(display_labels) if 'OTHER' in s]
            scatter_data0 =ScatterData3D(array[indices0, 0], '0',
                                         array[indices0, 1], '1',
                                         array[indices0, 2], '2',
                                         display_labels[indices0])
            indices1 = [i for i, s in enumerate(display_labels) if not 'OTHER' in s]
            scatter_data1 =ScatterData3D(array[indices1, 0], '0',
                                         array[indices1, 1], '1',
                                         array[indices1, 2], '2',
                                         display_labels[indices1])
        else:
            scatter_data0 = scatter_data

        unique_labels = np.unique(scatter_data0.labels)
        fig = plt.figure(dpi=200.0)
        ax = fig.add_subplot(projection='3d')
        for i, label in enumerate(np.sort(unique_labels)):
            label_indexes = scatter_data0.labels == label
            ax.plot(scatter_data0.x_data[label_indexes],
                    scatter_data0.y_data[label_indexes],
                    scatter_data0.z_data[label_indexes],
                    color=color_palette[label], marker='.', linestyle="None", label=label)
            if 'OTHER' in display_labels:
                ax.plot(scatter_data1.x_data[label_indexes],
                        scatter_data1.y_data[label_indexes],
                        scatter_data1.z_data[label_indexes],
                    color=color_palette[label], marker='.', linestyle="None", label=label)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        if title:
            ax.set_title(title)
        self._save_figure_to_file_if_needed(file_name)
        # plt.show(block=block)
        plt.close(fig)

    def plot_2d_scatter(
            self,
            array: NDArray[float],
            display_labels,
            specific_display_label_colors=None,
            display_labels_importance=None,
            title=None,
            file_name=None
    ) -> None:
        assert array.shape[1] == 2
        scatter_data = ScatterData2D(array[:, 0], 'dim 0',
                                     array[:, 1], 'dim 1',
                                     display_labels)
        # sns.color_palette("tab10", len(scatter_data.unique_labels))
        if not specific_display_label_colors:
            color_palette = sns.color_palette("tab10", len(scatter_data.unique_labels))
        else:
            color_palette = {}
            tab10 = sns.color_palette("tab10")
            i = 0
            for display_label in scatter_data.unique_labels:
                if display_label in specific_display_label_colors:
                    color_palette[display_label] = matplotlib.colors.to_rgb(
                        sns.crayons[specific_display_label_colors[display_label]])
                else:
                    color_palette[display_label] = tab10[i % 10]
                    i += 1

        if display_labels_importance:
            scatter_data = scatter_data.sort_by_label_importance(display_labels_importance)

        # fig = plt.figure(figsize=(5, 5), dpi=200.0)
        fig = plt.figure(dpi=200.0)
        ax = sns.scatterplot(
            x=scatter_data.x_legend, y=scatter_data.y_legend,
            hue=scatter_data.labels_legend,
            palette=color_palette,
            data=scatter_data.df,
            legend="full",
            alpha=0.5
        )

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        # manager = plt.get_current_fig_manager()
        # manager.window.showMaximized()
        if title:
            ax.set_title(title)
        self._save_figure_to_file_if_needed(file_name)
        # plt.show(block=block)
        plt.close(fig)

    def plot_curves(self, curves_dict, independent_axis_ticks, file_name=None):
        independent_axis = list(range(1, independent_axis_ticks + 1))
        fig = plt.figure(dpi=200.0)
        cmap = plt.get_cmap("tab10")
        for i, (label, curve) in enumerate(curves_dict.items()):
            plt.plot(independent_axis, curve, color=cmap(i), label=label)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        self._save_figure_to_file_if_needed(file_name)
        plt.close(fig)

    def plot_simple_bar_chart(self, df: pd.DataFrame, x_axis_field: str, y_axis_field: str, title: str,
                              file_name: str = None, show_figure: bool = True):
        sns.set(font_scale=1.4)
        fig = plt.figure()
        ax = sns.barplot(x=x_axis_field, y=y_axis_field, data=df, color='green')
        if title:
            ax.set_title(title)
        self._save_figure_to_file_if_needed(file_name)
        if show_figure:
            plt.show()
        plt.close(fig)

    def plot_bar_catplot(self, df: pd.DataFrame, x_axis_field: str, y_axis_field: str, hue_field_name: str,
                         col_separator_field_name: str, title: str, file_name: str = None, show_figure: bool = True):
        sns.set(font_scale=1.4)
        ax = sns.catplot(x=x_axis_field, y=y_axis_field, hue=hue_field_name, col=col_separator_field_name, data=df,
                         kind="bar", col_wrap=3, height=4, aspect=.7
                         # , dodge=False
                         )
        ax.set_titles(col_template='{col_name}')
        if title:
            ax.fig.suptitle(title)
            ax.fig.subplots_adjust(top=0.9)
        self._save_figure_to_file_if_needed(file_name)
        if show_figure:
            plt.show()
        plt.close(ax.fig)
