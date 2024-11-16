import os.path
import sys
from os.path import abspath, isfile, join, isdir
from glob import glob

from natsort import natsorted
import re
import pandas as pd


def extract_rep(exp: str, cell_line: str, drug: str) -> int | None:
    pattern = r'\d+-' + re.escape(f'{cell_line}-{drug}')  # Create pattern
    match = re.search(pattern, exp)

    if match:
        return int(match.group(0).split('-')[0])  # Get number before the '-'
    raise RuntimeError(f'Cloud not parse "{exp}"')


def main():
    """
    Main entry point, create sbatch and run it, for each cloud.
    """
    print('Starting to collect...')
    version = 'baseline'
    if len(sys.argv) < 3:
        print("Please provide directory name and output file name")
    runs = [abspath(x) for x in glob(join(sys.argv[1], '*')) if isdir(x)]
    runs = natsorted(runs, key=lambda y: y.lower())
    print(f'Found {len(runs):,} runs.')
    res=[]
    for expr in runs:
        try:
            tname = expr.split(version + '_')[1]
            tname1 = tname.split('_')
            cell_line = tname1[0]
            drug = tname1[1]
            rep = extract_rep(expr, cell_line, drug)
        except Exception as e:
            print(f'Could not parse the experiment {expr}, error: {e}')
            continue
        print(f'Collecting {cell_line} {drug} {rep}')
        with open(os.path.join(expr, 'log.txt')) as file:
            lines = [line.rstrip() for line in file]
        if 'Flow run SUCCESS: all reference tasks succeeded' not in lines[-1]:
            print(f'Run {cell_line} {drug} {rep} has not finished or has failed...')
            continue
        lines0 = [re.sub(' +', ' ',i).split(' ') for i in lines if (('  Left Out' in i) | (i.endswith('CV')))]
        if len(lines0) < 2:
            print("Empty file")
            continue
        lines0 = [[i[-2],i[-1]] if (i[-1]=='CV') else [i[-3],i[-2]] for i in lines0]
        lines1 = lines0[::2]
        lines2 = lines0[1::2]
        lines3 = [[float(a[0]), float(b[0])] for (a, b) in zip(lines1, lines2)]
        df = pd.DataFrame(lines3, columns=['LeftOut', 'CV'])
        # print(df.head())
        tres = [cell_line, drug, rep]
        tres.extend(df.iloc[df['LeftOut'].idxmax()].to_list())
        tres.extend(df.iloc[df['CV'].idxmax()].to_list())
        tres.extend(df.iloc[df.shape[0] - 1].to_list())
        res.append(tres)
    resdf = pd.DataFrame(res, columns=['Cell', 'Drug', 'Rep', 'LOMax_Left_Out','LOMax_CV','CVMax_Left_Out','CVMax_CV','LastEpoch_Left_Out','LastEpoch_CV'])
    resdf.to_csv(sys.argv[2],sep='\t',index=False)
    bestdf = resdf.groupby(['Cell','Drug'])['LOMax_Left_Out'].max()
    bestdf.reset_index()
    bestdf.to_csv(sys.argv[2].replace('.csv','_best.csv'),sep='\t')

if __name__ == '__main__':
    main()
