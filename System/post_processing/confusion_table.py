from tessellation_tester import TessellationTester
from configuration import config
import logging
import os
import pandas as pd
import numpy as np


class ConfusionTable:
    def __init__(self):
        self.output_folder_name = 'ConfusionTable'
        self.output_folder = None

    def run(self, test_name, data, model):
        """
        Run semi supervised tests - Calculate distance from calculated samples to left out treated samples.
        :param test_name: name of running test
        :param data: data handler
        :param model: model handler to use
        """
        logging.info("Starting confusion table tests: %s", str(test_name))
        output_folder = os.path.join(config.config_map['output_folder'], self.output_folder_name)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # Calculate all the clouds
        data_df, info_df, reference_points = data.get_all_data_and_info()
        encoded_data_df = model.predict_latent_space(data_df)

        labels_predicted = np.argmax(
            model.predict_classifier(data_df).values,
            axis=-1)
        results_df = pd.DataFrame(labels_predicted, index=data_df.index, columns=['predicted'])
        numeric_to_label_df = pd.DataFrame(columns=['tumor', 'perturbation'])
        for n in info_df.numeric_labels.unique():
            samples = info_df[info_df.numeric_labels == n].iloc[0]
            numeric_to_label_df.loc[n] = [samples.tumor, samples.perturbation]
        system_classifier_folder = os.path.join(output_folder, "SystemClassifier")
        svm_folder = os.path.join(output_folder, "svm")
        try:
            os.mkdir(system_classifier_folder)
        except IOError:
            pass

        results_df = results_df.join(numeric_to_label_df, on='predicted', how='left')
        for t in info_df.tumor.unique():
            t_cloud = info_df[info_df.tumor == t]
            errors_df = pd.DataFrame(index=t_cloud.perturbation.unique(), columns=t_cloud.perturbation.unique())
            errors_df['other'] = 0
            out_path = os.path.join(system_classifier_folder, t + '.csv').replace('|', '')
            for p in t_cloud.perturbation.unique():
                info_cloud_df = t_cloud[t_cloud.perturbation == p]
                curr_results_df = results_df.loc[info_cloud_df.index]
                for predicted_p in t_cloud.perturbation.unique():
                    errors_df.loc[p, predicted_p] = curr_results_df[(curr_results_df.tumor == t) &
                                                              (curr_results_df.perturbation == predicted_p)].shape[0]
                errors_df.loc[p, 'other'] = info_cloud_df.shape[0] - errors_df.loc[p].sum()
                errors_df.loc[p] = errors_df.loc[p] / errors_df.loc[p].sum()
            errors_df.to_csv(out_path)

        try:
            os.mkdir(svm_folder)
        except IOError:
            pass
        tessellation_tester = TessellationTester()
        tessellation_tester.fit(encoded_data_df, info_df)
        accuracy, results_df = tessellation_tester.get_accuracy(encoded_data_df, info_df)
        results_df = results_df.join(numeric_to_label_df, on='predicted', how='left')
        for t in info_df.tumor.unique():
            t_cloud = info_df[info_df.tumor == t]
            errors_df = pd.DataFrame(index=t_cloud.perturbation.unique(), columns=t_cloud.perturbation.unique())
            errors_df['other'] = 0
            out_path = os.path.join(svm_folder, t + '.csv').replace('|', '')
            for p in t_cloud.perturbation.unique():
                info_cloud_df = t_cloud[t_cloud.perturbation == p]
                curr_results_df = results_df.loc[info_cloud_df.index]
                for predicted_p in t_cloud.perturbation.unique():
                    errors_df.loc[p, predicted_p] = curr_results_df[(curr_results_df.tumor == t) &
                                                                    (curr_results_df.perturbation == predicted_p)].shape[0]
                errors_df.loc[p, 'other'] = info_cloud_df.shape[0] - errors_df.loc[p].sum()
                errors_df.loc[p] = errors_df.loc[p] / errors_df.loc[p].sum()
            errors_df.to_csv(out_path)

