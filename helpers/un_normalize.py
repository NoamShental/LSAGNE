import pandas as pd
from sklearn.externals import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-in', help='Path to input df', dest='inp', required=True)
parser.add_argument('-out', help='Path to output df', dest='out', required=True)
args = parser.parse_args()
scaler = joblib.load('/RG/compbio/groupData/organized_data/bashan/1/cmap_scaler')
df = pd.read_hdf(args.inp, 'df')
unscaled_data_np = scaler.inverse_transform(df)
unscaled_df = pd.DataFrame(unscaled_data_np, df.index, df.columns)
unscaled_df.to_hdf(args.out, 'df')
