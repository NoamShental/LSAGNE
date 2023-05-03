import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sn
from System.configuration import config, mock

def table_to_hetmaps(curr_drug, curr_tabel, max_value, tmp_img_path):
    # Load global table.
    table_df = pd.read_csv(curr_tabel)
    # Columns names to cell id.
    table_df.columns  = table_df.columns.str.split(n=1).str[0]
    # Tumors name to cell id.
    table_df["Tumor"] = table_df["Tumor"].str.split(n=1).str[0]
    table_df.reset_index(drop=True, inplace=True)
    heatmap_df = table_df.pivot_table(table_df, index=['Tumor'])
    # Create heatmap.
    # MAsk the upper triangle without the diagonal.
    heatmap_mask = np.zeros_like(heatmap_df, dtype=np.bool)
    heatmap_mask[np.triu_indices_from(heatmap_mask, k=1)] = True
    ax = sn.heatmap(heatmap_df, annot=True, cmap='coolwarm', linecolor='white', linewidths=1,
                    fmt='g', vmin=0, vmax=max_value, mask=heatmap_mask)
    ax.set_title(curr_drug, fontsize=12, fontweight='semibold')
    plt.yticks(va='center', size=8)
    plt.ylabel('Cell ID', fontweight='semibold')
    plt.xticks(rotation=12, ha='center', size=8)
    figure = ax.get_figure()
    figure.savefig(tmp_img_path.format(curr_drug))
    plt.close('all')

def angels_table_to_hetmaps(angels_table_df, tmp_img_path):
    #angels_table_df = angels_table_df.round(1)
    # Create heatmap for engels table.
    heatmap_df = angels_table_df.pivot_table(angels_table_df, index=['Perturbation'])
    # Mask the upper triangle without the diagonal.
    heatmap_mask = np.zeros_like(heatmap_df, dtype=np.bool)
    heatmap_mask[np.triu_indices_from(heatmap_mask, k=1)] = True
    ax = sn.heatmap(heatmap_df, annot=True, cmap='coolwarm', linecolor='white', linewidths=1,
                    mask = heatmap_mask, fmt='.1f')

    plt.yticks(va='center', size=8, rotation=78)
    plt.ylabel('Drug name', fontweight='semibold')
    plt.xticks(rotation=12, ha='center', size=8)
    figure = ax.get_figure()
    figure.savefig(tmp_img_path)
    plt.close('all')


def confusion_tables_to_hetmaps(confusion_table_df, tmp_img_path):
    confusion_table_df.drop('other', inplace=True, axis=1)
    confusion_table_df = confusion_table_df.round(2)

    # Create heatmap table for each cell line.
    for tumor in confusion_table_df.Tumor.unique():
        curr_tumor_confusion_table_df = confusion_table_df[confusion_table_df['Tumor'] == tumor]
        # Delete tumor column.
        curr_tumor_confusion_table_df.drop(['Tumor'], inplace=True, axis=1)
        curr_tumor_confusion_table_df.reset_index(drop=True, inplace=True)
        # Set drugs as index
        heatmap_df = curr_tumor_confusion_table_df.pivot_table(curr_tumor_confusion_table_df, index=['Perturbation'])
        # Move the DMSO to be last column in the table.
        columnsName = list(heatmap_df.columns)
        columnsName.remove('DMSO')
        columnsName.insert(len(columnsName), 'DMSO')
        heatmap_df = heatmap_df[columnsName]
        plt.figure(figsize=(8, 8))
        # Create heatmap.
        ax = sn.heatmap(heatmap_df, annot=True, cmap='coolwarm', linecolor='white', linewidths=1,
                            vmin=0, vmax=1)
        ax = sn.heatmap(heatmap_df, mask=heatmap_df < 2, cbar=False, linecolor='white', linewidths=1)

        ax.set_title(tumor, fontsize=12, fontweight='semibold')
        plt.yticks(va='center', size=8)
        plt.ylabel('Drug name', fontweight='semibold')
        plt.xticks(rotation=12, ha='center', size=8)
        figure = ax.get_figure()
        figure.savefig(tmp_img_path.format(tumor))
        plt.close('all')

def get_tables_max_value(tabels_path):
    max_val = 0
    for drug in config.config_map['perturbations_whitelist']:
        curr_drug_table_df = pd.read_csv(os.path.join(tabels_path, "{}.csv".format(drug)))
        curr_drug_table_df.drop(['Tumor'], inplace=True, axis=1)
        curr_table_max_val = curr_drug_table_df.values.max()
        if curr_table_max_val > max_val:
            max_val = curr_table_max_val

    return max_val


def main():
    off_line_prev_results_path = config.config_map['off_line_prev_results_path']

    # Handle svm confusion tables.
    svm_confusion_tabels_path = os.path.join(off_line_prev_results_path, 'confusion_tables_svm.csv')
    svm_confusion_tabels_output_path = os.path.join(off_line_prev_results_path, 'confusion_tables_svm')
    # Create output root folder if not exists.
    if not os.path.isdir(svm_confusion_tabels_output_path):
        os.makedirs(svm_confusion_tabels_output_path)
    tmp_img_path = os.path.join(svm_confusion_tabels_output_path, '{}_confusion_table_svm.png')
    # Load global table.
    svm_confusion_table_df = pd.read_csv(svm_confusion_tabels_path)
    confusion_tables_to_hetmaps(svm_confusion_table_df, tmp_img_path)

    # Handle angels tables.
    angels_tabel_path = os.path.join(off_line_prev_results_path, 'angles.csv')
    angels_tabel_output_path = os.path.join(off_line_prev_results_path, 'angels')
    # Create output root folder if not exists.
    if not os.path.isdir(angels_tabel_output_path):
        os.makedirs(angels_tabel_output_path)
    tmp_img_path = os.path.join(angels_tabel_output_path, 'angel_table.png')
    # Load angels table.
    angels_table_df = pd.read_csv(angels_tabel_path)
    angels_table_to_hetmaps(angels_table_df, tmp_img_path)


    """
    # Handle tester classifier confusion tables.
    system_classifier_confusion_tabels_path = os.path.join(off_line_prev_results_path, 'confusion_tables_system_classifier.csv')
    system_classifier_confusion_tabels_output_path = os.path.join(off_line_prev_results_path, 'confusion_tables_system_classifier')
    # Create output root folder if not exists.
    if not os.path.isdir(system_classifier_confusion_tabels_output_path):
        os.makedirs(system_classifier_confusion_tabels_output_path)
    tmp_img_path = os.path.join(system_classifier_confusion_tabels_output_path, '{}_confusion_table_system_classifier.png')
    # Load global table.
    system_classifier_confusion_table_df = pd.read_csv(system_classifier_confusion_tabels_path)
    confusion_tables_to_hetmaps(system_classifier_confusion_table_df, tmp_img_path)
    """

    # Handle naive results.
    naive_tabels_path = os.path.join(off_line_prev_results_path, 'naive')
    max_value = get_tables_max_value(naive_tabels_path)
    for drug in config.config_map['perturbations_whitelist']:
        curr_drug_naive_tabel_path = os.path.join(naive_tabels_path, "{}.csv".format(drug))
        tmp_img_path = os.path.join(naive_tabels_path,'{}_naive_results.png'.format(drug))
        table_to_hetmaps(drug, curr_drug_naive_tabel_path, max_value, tmp_img_path)


    
    # Handle trajectories results.
    trajectories_tabels_path = os.path.join(off_line_prev_results_path, 'trajectories')
    max_value = get_tables_max_value(trajectories_tabels_path)
    for drug in config.config_map['perturbations_whitelist']:
        curr_drug_trajectories_tabel_path = os.path.join(trajectories_tabels_path, "{}.csv".format(drug))
        tmp_img_path = os.path.join(trajectories_tabels_path, '{}_trajectories_results.png'.format(drug))
        table_to_hetmaps(drug, curr_drug_trajectories_tabel_path, max_value, tmp_img_path)



if __name__ == '__main__':
  main()


