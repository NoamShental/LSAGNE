import argparse
import os
import pandas as pd
import string


def file_name_escaping(filename):
    """
    Make escaping for file names (i.e: omit '|' or '\'.
    :param filename: filename to escape
    :return: escaped filename
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


parser = argparse.ArgumentParser()
parser.add_argument('-in', help='Path to input df', dest='inp', required=True)
parser.add_argument('-out', help='folder to save csv', dest='out', required=True)
args = parser.parse_args()

df = pd.read_hdf(args.inp, 'df')
for d in df.index.get_level_values(0).unique():
    drug_samples = df.loc[d]
    tumors = drug_samples.index.get_level_values(0).unique()
    for t in tumors:
        cloud_samples = drug_samples.loc[t]
        t = file_name_escaping(t)
        print('Saving samples to csv {}_{}.csv'.format(d, t))
        cloud_samples.to_csv(os.path.join(args.out, "{}_{}_decoded_samples.csv").format(d, t))
