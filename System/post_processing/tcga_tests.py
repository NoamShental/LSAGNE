from configuration import config
import pandas as pd
import numpy as np
from tessellation_tester import TessellationTester
import logging
import os


class TCGATests:
    def __init__(self):
        """
        C'tor - Initialize all needed properties
        """
        self.tcga_data_df = pd.read_hdf(os.path.join(config.config_map['organized_tcga_folder'], 'data.h5'), 'df')
        self.tcga_info_df = pd.read_csv(os.path.join(config.config_map['organized_tcga_folder'], 'info.csv'))
        self.tcga_info_df.set_index('sample', inplace=True)

    def run(self, test_name, data, model):
        """
        Run tcga test
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        data_df, info_df, reference_points = data.get_all_data_and_info()
        cmap_tumors = info_df.tumor.unique()
        tcga_tissues = self.tcga_info_df.primary_site.unique()
        tcga_tissues_to_cmap_labels = {}
        for tissue in tcga_tissues:
            cmap_tumors_for_tissue = [t for t in cmap_tumors if tissue.lower() in t.lower()]
            tcga_tissues_to_cmap_labels[tissue] =\
                info_df[info_df.tumor.isin(cmap_tumors_for_tissue)].numeric_labels.unique()

        logging.info("Starting tcga classification: %s", str(test_name))
        output_folder = os.path.join(config.config_map['output_folder'], 'tcga')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        encoded_tcga_data_df = model.predict_latent_space(self.tcga_data_df)
        encoded_cmap_df = model.predict_latent_space(data_df)
        tessellation_tester = TessellationTester()
        tessellation_tester.fit(encoded_cmap_df, info_df)
        tcga_predictions = tessellation_tester.predict(encoded_tcga_data_df.values)
        predictions_df = pd.DataFrame(tcga_predictions, index=encoded_tcga_data_df.index, columns=['prediction'])
        success_rate_df = pd.DataFrame(index=self.tcga_info_df.primary_site.unique(), columns=['rate'])
        success_rate_df.index.name = 'tissue'
        global_succeed = 0
        for tissue in self.tcga_info_df.primary_site.unique():
            tcga_tissue_info_df = self.tcga_info_df[self.tcga_info_df.primary_site == tissue]
            tcga_tissue_predictions_df = predictions_df.loc[tcga_tissue_info_df.index]
            cmap_labels_np = tcga_tissues_to_cmap_labels[tissue]
            succed_np = np.isin(tcga_tissue_predictions_df.prediction.values, cmap_labels_np)
            succeed = np.sum(succed_np)
            global_succeed += succeed
            success_rate_df.loc[tissue] = succeed / tcga_tissue_predictions_df.shape[0]
            success_rate_df.to_hdf(os.path.join(output_folder, 'tcga.h5'), 'df')
