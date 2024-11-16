import logging

import pandas as pd

from src.drawer import Drawer
from src.evaluation.metrics_collector import collect_scores, collect_cdists


def print_scores_flat(results_dir: str, score_file_name: str, title: str, draw_file_name=None, drawing_dir=None,
                      y_axis_field: str='display name'):
    drawer = Drawer(logger=logging, folder_path=drawing_dir)
    df, scores_statistics = collect_scores(results_dir, score_file_name)
    print(f'score statistics are: {scores_statistics}')
    drawer.plot_simple_bar_chart(
        df=df,
        x_axis_field='score',
        y_axis_field=y_axis_field,
        title=title,
        file_name=draw_file_name
        )


def print_scores_bar_catplot(results_dir: str, score_file_name: str, title: str, draw_file_name=None, drawing_dir=None):
    drawer = Drawer(logger=logging, folder_path=drawing_dir)
    df, scores_statistics = collect_scores(results_dir, score_file_name)
    print(f'score statistics are: {scores_statistics}')
    # drawer.plot_bar_catplot(df=df,
    #                         x_axis_field='score',
    #                         y_axis_field='tissue_code',
    #                         hue_field_name='run name',
    #                         col_separator_field_name='perturbation',
    #                         title=title,
    #                         file_name=draw_file_name)
    # Add average "run name" to each cloud
    display_name_means = df.groupby(['display name', 'tissue_code', 'perturbation'])['score'].mean().reset_index()
    display_name_means['run name'] = 'avg'
    full_df = pd.concat([df, display_name_means])
    drawer.plot_bar_catplot(df=full_df,
                            x_axis_field='score',
                            y_axis_field='run name',
                            hue_field_name='tissue_code',
                            col_separator_field_name='display name',
                            title=title,
                            file_name=draw_file_name)


def print_cdist_bar_catplot(results_dir: str, cdist_file_name: str, metric: str, title: str, draw_file_name=None,
                            drawing_dir=None):
    drawer = Drawer(logger=logging, folder_path=drawing_dir)
    df = collect_cdists(results_dir, cdist_file_name, metric)
    drawer.plot_bar_catplot(df=df,
                            x_axis_field='score',
                            y_axis_field='tissue_code',
                            hue_field_name='corr_name',
                            col_separator_field_name='perturbation',
                            title=title,
                            file_name=draw_file_name)