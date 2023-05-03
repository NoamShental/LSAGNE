import pandas as pd
MINIMUM_SAMPLES_PER_CLOUD = 30
info_df = pd.read_csv('info.csv')
info_df = info_df[info_df.perturbation != 'DMSO']
g = info_df.groupby([ 'perturbation', 'tumor'])
occurences = g.agg('count').inst_id
filtered = occurences[occurences > MINIMUM_SAMPLES_PER_CLOUD]
unique_df = filtered.reset_index()[['tumor', 'perturbation']]
unique_df.to_csv('unique_clouds.csv')