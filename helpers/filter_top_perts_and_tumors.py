import pandas as pd

info_df = pd.read_csv('info.csv')

# Remove perturbation of that tumors with less then 100 samples
minimum_samples_per_cloud = 30
grouped = info_df.groupby(['tumor', 'perturbation', 'pert_time'])
top_clouds_info_df = grouped.filter(lambda x: x.shape[0] > minimum_samples_per_cloud)
all_tumors = list(top_clouds_info_df.tumor.unique())
#  all_tumors = list(info_df.tumor.unique())
top_clouds_info_df = info_df

# Remove tumors without DMSO on time 6h
minimum_control_samples_time_6 = 50
for t in all_tumors:
    if top_clouds_info_df[(top_clouds_info_df.tumor == t) &
                          (top_clouds_info_df.perturbation == 'DMSO') &
                          (top_clouds_info_df.pert_time == 6)].shape[0] < minimum_control_samples_time_6:
        all_tumors.remove(t)

# Remove tumors without DMSO on time 24h
minimum_control_samples_time_24 = 50
for t in all_tumors:
    if top_clouds_info_df[(top_clouds_info_df.tumor == t) &
                          (top_clouds_info_df.perturbation == 'DMSO') &
                          (top_clouds_info_df.pert_time == 24)].shape[0] < minimum_control_samples_time_24:
        all_tumors.remove(t)

top_clouds_info_df = top_clouds_info_df[top_clouds_info_df.tumor.isin(all_tumors)]
all_perts = list(top_clouds_info_df.perturbation.unique())

best_perturbations = {}
tumors = []
for p in all_perts:
    pert_info_df = top_clouds_info_df[(top_clouds_info_df.perturbation == p) & (top_clouds_info_df.pert_time == 24)]
    pert_dict = {}
    for t in pert_info_df.tumor.unique():
        pert_dict[t] = pert_info_df[pert_info_df.tumor == t].shape[0]
    if len(pert_dict) > 5:
        best_perturbations[p] = pert_dict
        for t in pert_dict.keys():
            if t not in tumors:
                tumors.append(t)

perts_matrix = pd.DataFrame(index=best_perturbations.keys(), columns=tumors)
for p, tumors in best_perturbations.items():
    for t, number_of_samples in tumors.items():
        perts_matrix.loc[p][t] = number_of_samples
perts_matrix['total'] = perts_matrix.sum(axis=1)
perts_matrix.sort(columns='total', inplace=True)
perts_matrix.to_csv('perturbations matrix.csv')
