import pandas as pd
real_info = 'd:\\info.csv'
data = 'd:\\data.h5'
prediction = 'd:\\predicted_classifier.csv'

info_df = pd.read_csv(real_info)
predictions_df = pd.read_csv(prediction)

failed_df = predictions_df[predictions_df.real_number != predictions_df.predicted]

print('Number of samples: {}'.format(info_df.shape[0]))
print('Failed classification samples: {}'.format(failed_df.shape[0]))

failed_in_control = 0
failed_in_perturbated = 0
failed_in_control_as_same_tumor_perturbated = 0
failed_in_control_as_other_tumor_control = 0
failed_in_control_as_other_tumor_perturbated = 0
failed_in_perturbated_other_perturbation = 0
failed_in_perturbated_as_control = 0
failed_in_perturbated_as_other_tumor = 0
for i, row in failed_df.iterrows():
    current_sample_info = info_df.loc[row.name]
    predicted_cloud = info_df[info_df['numeric_labels'] == row.predicted].iloc[0]
    real_cloud = info_df[info_df['numeric_labels'] == row.real_number].iloc[0]
    predicted_tumor = predicted_cloud.tumor
    predicted_perturbation = predicted_cloud.perturbation
    real_tumor = real_cloud.tumor
    real_perturbation = real_cloud.perturbation
    if real_perturbation == 'DMSO':
        failed_in_control += 1
        if predicted_tumor == real_tumor:
            failed_in_control_as_same_tumor_perturbated += 1
        else:
            if predicted_perturbation == 'DMSO':
                failed_in_control_as_other_tumor_control += 1
            else:
                failed_in_control_as_other_tumor_perturbated += 1
    else:
        failed_in_perturbated += 1
        if predicted_tumor == real_tumor and predicted_perturbation == 'DMSO':
            info_df.drop(row.name, inplace=True)
            failed_in_treatment += 1
        elif predicted_tumor == real_tumor:
            failed_in_other_perturbation += 1
        else:
            failed_other_tumor += 1

print('Failed samples that are from control set: {}'.format(failed_in_control))
print('From which: {} as same tumor but perturbated, {} as other tumor control, {} as other tumor perturbated\n'.format(failed_in_control_as_same_tumor_perturbated, failed_in_control_as_other_tumor_control, failed_in_control_as_other_tumor_perturbated))
print('Failed samples that are from perturbated set: {}'.format(failed_in_perturbated))
print('From which: {} as same tumor and other perturbation, {} as same tumor control, {} as other tumor\n'.format(failed_in_perturbated_other_perturbation, failed_in_perturbated_as_control, failed_in_perturbated_as_other_tumor))

print('Samples after filter: {}'.format(info_df.shape[0]))

data_df = pd.read_hdf(data)
info_df.index = info_df["inst_id"]
data_df = data_df.loc[info_df.index]

data_df.to_hdf('d:\\data_filtered.h5', key='df')
info_df.to_csv('d:\\info_filtered.csv', sep=',', columns=['perturbation',
                                                          'tumor',
                                                          'classifier_labels',
                                                          'numeric_labels',
                                                          'pert_time'])
