import subprocess
import pandas as pd
from System.configuration import mock


if mock:
    ORGANIZED_DATA_FOLDER = r'D:\Thesis\Data\OrganizedMock'
    OUTPUT_FOLDER = r'D:\Thesis\output_mock'
    unique_clouds_df = pd.read_csv(r'helpers\unique_clouds_mock.csv')
else:
    ORGANIZED_DATA_FOLDER = r'D:\Thesis\Data\OrganizedCmap'
    OUTPUT_FOLDER = r'D:\Thesis\output'
    unique_clouds_df = pd.read_csv(r'helpers\unique_clouds_cmap.csv')

for i in range(unique_clouds_df.shape[0]):
    row = unique_clouds_df.iloc[i]
    tumor = row['tumor']
    pert = row['perturbation']
    command = 'python System\main.py --run -tumor "{0}" -pert "{1}" -output {2} -organized-folder {3} -start 0 -end 2 -test-num {4} 1>&2'.format(
        tumor, pert, OUTPUT_FOLDER, ORGANIZED_DATA_FOLDER, str(i))
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
