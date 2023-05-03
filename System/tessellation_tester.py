from sklearn.svm import LinearSVC
from configuration import config
import pandas as pd
import numpy as np


class TessellationTester:
    """
    This class will handle a tessellation tester
    """
    def __init__(self):
        """
        Initialize the tester
        """
        if config.config_map['should_use_class_weights']:
            self.svm = LinearSVC(tol=1e-5, max_iter=10000, class_weight='balanced')
        else:
            self.svm = LinearSVC(tol=1e-5, max_iter=10000)

    def predict(self, data_np):
        """
        Predict tessellation for a given data
        :param data_np: data to predict
        :return: predictions labels
        """
        return self.svm.predict(data_np)

    def get_accuracy(self, data_df, info_df):
        """
        Get accuracy rate of prediction
        :param data_df: data to predict
        :param info_df: DataFrame with real info
        :return: accuracy rate in percentage, DataFrame with the results
        """
        predictions = self.predict(data_df.values)
        results_df = pd.DataFrame({'predicted': predictions, 'real': info_df.numeric_labels.values},
                                  index=data_df.index)
        results_df['real'] = info_df.numeric_labels.values
        return (np.sum(results_df.predicted == info_df.numeric_labels.values) / results_df.predicted.shape[0]) * 100,\
            results_df

    def fit(self, data_df, info_df):
        """
        Fit the tester on given data and info
        :param data_df: data to fit
        :param info_df: labels of the data
        """
        self.svm.fit(data_df.values, info_df.numeric_labels.values)
