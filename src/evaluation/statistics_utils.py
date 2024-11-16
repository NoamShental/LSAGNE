import numpy as np
from scipy.stats import iqr


def calculate_detailed_statistics(arr):
    np_arr = np.array(arr)
    return {
        'mean': np_arr.mean(),
        'min': np_arr.min(),
        'max': np_arr.max(),
        'median': np.median(np_arr),
        'iqr': iqr(np_arr)
    }