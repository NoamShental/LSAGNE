import pandas as pd

tumors_whitelist = ["MCF7 breast adenocarcinoma", "PC3 prostate adenocarcinoma", "A375 skin malignant melanoma", "A549 lung non small cell lung cancer| carcinoma", "HT29 large intestine colorectal adenocarcinoma", "HA1E kidney normal kidney", "HCC515 lung carcinoma", "HEPG2 liver hepatocellular carcinoma", "VCAP prostate carcinoma"]
perturbation_whitelist = ['DMSO', 'vorinostat', 'wortmannin', 'geldanamycin', 'trichostatin-a', 'Unterated', 'raloxifene', 'curcumin', 'sulforaphane', 'sirolimus', 'withaferin-a', 'tozasertib', 'thioridazine', 'estriol', 'TL-HRAS-61', 'tanespimycin', 'tretinoin', 'panobinostat', 'olaparib', 'troglitazone', 'barasertib-HQPA', 'resveratrol', 'dexamethasone', 'gemcitabine', 'parthenolide', 'selumetinib']

# Count perts per cloud
info_df = pd.read_csv('info.csv')
info_df = info_df[info_df.tumor.isin(tumors_whitelist)]
info_df = info_df[info_df.perturbation.isin(perturbation_whitelist)]
grouped = info_df.groupby(['tumor', 'perturbation', 'pert_time'])
top_clouds_info_df = grouped.count()[['inst_id']]
perts_matrix = top_clouds_info_df['inst_id'].unstack(level=0)
perts_matrix.to_csv('perturbation_matrix.csv')