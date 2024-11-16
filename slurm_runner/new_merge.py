import os
import pandas as pd

def merge_h5_chunks(root_dir, output_file):
    # Initialize an empty list to hold dataframes
    dataframes = []

    # Walk through the directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.h5') and 'results' in file:
                # Construct full file path
                file_path = os.path.join(root, file)
                print(file_path)
                # Read the dataframe and append to the list
                df = pd.read_hdf(file_path)
                dataframes.append(df)

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Write the merged dataframe to a tab-separated text file
    merged_df.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    # Define the root directory and output file path
    root_dir = '/RG/compbio/emil/augmentation_optimization_root_v4/results'
    root_dir = '/Users/emil/MSc/lsagne-1/output/from_cloud/augmentation_optimization_v4_0-1'
    output_file = os.path.join(root_dir, 'merged_results.tsv')

    # Call the function
    merge_h5_chunks(root_dir, output_file)
    print('Done!')
