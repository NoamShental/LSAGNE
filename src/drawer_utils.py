import numpy as np


def create_label_to_display_name(labels, display_name):
    # for index, info in train_cmap_dataset.info_df.iterrows():
    #     labels.append(f"p={info['perturbation']};t={info['tumor']}")
    label_to_display_name = {}
    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0][0]
        label_to_display_name[label] = display_name[label_idx]
    return label_to_display_name
