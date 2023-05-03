from configuration import config
import pandas as pd
import statistics
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import numpy as np
import pickle


class DeathPredictor:
    """
    This class can predict death based on pre-trained SVM
    """
    def __init__(self, data):
        self.data = data
        self.dmso_data = {}
        svm_file = os.path.join(config.config_map['organized_data_folder'],
                                config.config_map['death_predictor_svm_file'])
        annonation_file = os.path.join(config.config_map['organized_data_folder'],
                                       config.config_map['death_predictor_annotation_file'])
        self.annotation_data = pd.read_csv(annonation_file, sep="\t", index_col=0)
        self.cell_cycles = self.get_cell_cycles()
        self.pathways_to_genes_dict = {}
        for pathway in list(self.cell_cycles.index):
            self.pathways_to_genes_dict[pathway] = [int(str(i).strip()) for i in self.cell_cycles.loc[pathway][2].split(",")]

        self.calculate_baseline_values()
        with open(svm_file, 'rb') as f:
            self.svm = pickle.load(f)

    @staticmethod
    def get_cell_cycles():
        reliable_pathways = ["GO:0000070", "GO:0000076", "GO:0000084", "GO:0000085", "GO:0000087", "GO:0000727",
                             "GO:0000729", "GO:0002438", "GO:0002526", "GO:0002675", "GO:0006957", "GO:0007051",
                             "GO:0007052", "GO:0007077", "GO:0007094", "GO:0007140", "GO:0008628", "GO:0008631",
                             "GO:0009414", "GO:0009635", "GO:0010544", "GO:0019731", "GO:0031571", "GO:0034121",
                             "GO:0034145", "GO:0042262", "GO:0042832", "GO:0043068", "GO:0043620", "GO:0045786",
                             "GO:0045787", "GO:0045954", "GO:0046599", "GO:0060561", "GO:0090201", "GO:0090331",
                             "GO:1900744", "GO:1902237", "GO:1903608", "GO:1903753", "GO:1990253", "GO:2000107",
                             "GO:2001236"]

        cell_cycles_file = os.path.join(config.config_map['organized_data_folder'],
                                        config.config_map['death_predictor_go_cycles_files'])
        cell_cycles = pd.read_csv(cell_cycles_file, sep="~|\t", index_col=0, header=None, engine='python')
        return cell_cycles.loc[reliable_pathways]

    def calculate_baseline_values_v0(self):
        inst_dmso = self.annotation_data[self.annotation_data["pert_iname"] == 'DMSO']
        dmso_grouped = inst_dmso.groupby(['cell_id'])
        data_df, info_df, _ = self.data.get_all_data_and_info()
        info_df = info_df[info_df.perturbation.isin(config.config_map['untreated_labels'])]
        dmso_df = data_df.loc[info_df.index]
        dmso_df = self.data.get_12k_data(dmso_df)

        for group_name, cell_group in dmso_grouped:
            self.dmso_data[group_name] = {}
            tdata = dmso_df.loc[cell_group.index]
            for pathway in list(self.cell_cycles.index):
                genes = [int(str(i).strip()) for i in self.cell_cycles.loc[pathway][2].split(",")]
                cell_pathway = tdata[genes]
                if len(genes) != len(cell_pathway.columns):
                    logging.error("Something wrong")
                if len(list(set(genes))) != len(genes):
                    logging.error("Something wrong")
                mmean = statistics.mean(list(cell_pathway.mean(axis=1)))
                self.dmso_data[group_name][pathway] = mmean

    def calculate_baseline_values(self):
        inst_dmso = self.annotation_data[self.annotation_data["pert_iname"] == 'DMSO']
        cell_lines = inst_dmso['cell_id'].unique()
        data_df, info_df, _ = self.data.get_all_data_and_info()
        info_df = info_df[(info_df.perturbation.isin(config.config_map['untreated_labels'])) &
                          (info_df.pert_time.isin(config.config_map['perturbation_times']))]
        dmso_df = data_df.loc[info_df.index]
        dmso_df = self.data.get_12k_data(dmso_df)
        inst_dmso = inst_dmso[inst_dmso.index.isin(dmso_df.index)]

        for c in cell_lines:
            self.dmso_data[c] = {}
            control_data_df = dmso_df.loc[inst_dmso[inst_dmso['cell_id'] == c].index]
            for pathway in list(self.cell_cycles.index):
                genes = self.pathways_to_genes_dict[pathway]
                cell_pathway = control_data_df[genes]
                self.dmso_data[c][pathway] = cell_pathway.values.mean()

    def predict_v0(self, test_df):
        test_df.columns = [str(i).strip() for i in test_df.columns]
        X_test = []
        for index, sample in test_df.iterrows():
            rdata = []
            for pathway in list(self.cell_cycles.index):
                genes = [str(i).strip() for i in self.cell_cycles.loc[pathway][2].split(",")]
                cell_pathway = sample.loc[genes]
                if len(genes) != cell_pathway.shape[0]:
                    logging.error("Something wrong")
                if len(list(set(genes))) != len(genes):
                    logging.error("Something wrong")
                mmean = cell_pathway.mean()
                rdata.append(self.dmso_data[index[0].split(' ')[0]][pathway] - mmean)
            X_test.append(rdata)
        X_test = np.array(X_test)
        y_pred = self.svm.predict(X_test)
        results_df = pd.DataFrame(y_pred, index=test_df.index, columns=['death'])
        return results_df

    def predict(self, test_df):
        test_df.columns = [int(str(i).strip()) for i in test_df.columns]
        X_test = []
        for index, sample in test_df.iterrows():
            rdata = []
            curr_cell_line = index[0].split(' ')[0]
            for pathway in list(self.cell_cycles.index):
                genes = self.pathways_to_genes_dict[pathway]
                cell_pathway = sample.loc[genes]
                if len(genes) != cell_pathway.shape[0]:
                    logging.error("Something wrong")
                if len(list(set(genes))) != len(genes):
                    logging.error("Something wrong")
                mmean = cell_pathway.mean()
                rdata.append(self.dmso_data[curr_cell_line][pathway] - mmean)
            X_test.append(rdata)
        X_test = np.array(X_test)
        y_pred = self.svm.predict(X_test)
        results_df = pd.DataFrame(y_pred, index=test_df.index, columns=['death'])
        return results_df
