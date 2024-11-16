from pathlib import Path
import pandas as pd

if __name__ == '__main__':
    results_path = Path('/Users/emil/MSc/lsagne-1/output/aug_test_params_root/output')

    results = []
    for file_path in results_path.iterdir():
        print(f'Loading {file_path}...')
        results.append(pd.read_hdf(file_path, key='df'))
        # results[-1].drop(['B'0, 'C'], axis=1, inplace=True)
    all_res = pd.concat(results)
    print('done')
