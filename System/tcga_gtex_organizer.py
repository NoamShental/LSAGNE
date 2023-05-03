from configuration import config, mock

import pandas as pd
import os
import shutil
import logging
import numpy as np

class HealtyTissuesDataOrganizer:
    """
    This class will organize the TCGA, according to CMAP organized data.
    """
    def __init__(self):
        """
        Initializer - set all files constants and other properties of the class.
        """

        self.tcga_folder = config.config_map['gtex_data_folder']

        # This file has TCGA-TARGET-GTEX samples, but only CMAP genes - 977, and not all the genes.
        self.tcga_samples_data = os.path.join(self.tcga_folder,
                                              'TcgaTargetGtex_RSEM_isoform_fpkm_CMAP978to977MADtranscripts_zeroone.tsv.gz')
        self.phenotype_filename = os.path.join(self.tcga_folder, 'TcgaTargetGTEX_phenotype.txt')
        self.data_df = None
        self.info_df = None

    def read_gtex_info_and_data(self, cmap_data_columns):
        """
        Read the information and the data of tcga.
        REM: It should be done in one function, because its mutual intersection.
        :param cmap_data_columns: list of columns for cmap data
        """
        info_df = pd.read_csv(self.phenotype_filename, index_col=0, encoding="ISO-8859-1", sep='\t')

        # Handle GTEX info:
        # Rename and delete some columns
        info_df['perturbation'] = info_df['_study']
        info_df['tumor'] = info_df['primary disease or tissue']

        logging.info("GTEx raw data info:")
        # Extract only GTEX.
        info_df = info_df[(info_df['perturbation'] == 'GTEX')]
        logging.info("Number of GTEx unfiltered info samples={}".format(info_df.shape[0]))

        # Extract only normal tissues.
        info_df = info_df[(info_df['_sample_type'] == 'Normal Tissue')]
        logging.info("Number of GTEx normal tisues info samples={}".format(info_df.shape[0]))

        info_df.drop(['_gender', '_study', '_primary_site', '_sample_type', 'primary disease or tissue',
                      'detailed_category'], axis=1, inplace=True)

        # Remove samples that have low occurrences.
        for disease_type in info_df['tumor'].unique():
            num_of_items = info_df[info_df['tumor'] == disease_type].shape[0]
            if (num_of_items < config.config_map['min_samples_per_gtex_class']):
                info_df = info_df[info_df["tumor"] != disease_type]
        logging.info("Number of GTEx info samples after dropping low occurrences classes={}".format(info_df.shape[0]))
        logging.info("Number of GTEx classes={}".format(len(info_df['tumor'].unique())))

        # Assign classifier_labels to all the rows.
        info_df['inst_id'] = info_df.index
        info_df['pert_time'] = 6
        info_df = info_df.astype({'pert_time': np.int32})
        info_df['pert_dose'] = '0.00%'
        self.info_df = info_df

        # Handle GTEX data:
        data_df = pd.read_csv(self.tcga_samples_data, index_col=0, sep='\t').transpose()
        logging.info("Number of TCGA_GTEx data samples before intersection={}".format(data_df.shape[0]))
        self.data_df = data_df.loc[data_df.index.intersection(self.info_df.index)]
        logging.info("Number of GTEx data samples after intersection={}".format(self.data_df.shape[0]))
        logging.info("Number of GTEx info samples before intersection={}".format(self.info_df.shape[0]))
        self.info_df = self.info_df.loc[self.info_df.index.intersection(data_df.index)]
        logging.info("Number of GTEx info samples after intersection={}".format(self.info_df.shape[0]))

        self.data_df = pd.DataFrame(self.data_df.values, columns=cmap_data_columns, index=self.data_df.index)
