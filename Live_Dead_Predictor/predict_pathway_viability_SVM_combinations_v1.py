import pandas as pd
import statistics
import os
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import numpy as np
import pickle
from sklearn.utils import class_weight
from xgboost import XGBClassifier

infilenames=[
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Sirolimus_Raloxifene/drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt",
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial0-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial1-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial2-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial3-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial4-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial5-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial6-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial7-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial8-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    'E:/Noam/CMAP/drug_combinations/DUDI/AlphaAllDecodedData_I-Siro_Tamo_Vori/drug_combinations_Trial9-A549 lung non small cell lung cancer carcinoma_wortmannin_138Pathways_deltaDMSO_livedead_v1.txt',
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_0_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_1_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_2_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_3_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_4_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_5_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_6_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_7_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_8_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt",
    "E:/Noam/CMAP/allsamp/to_Michael_all_pred_Bashan.tar/8_with_ralox.tar/8_with_ralox/drug_combinations_9_50_HCC515 lung carcinoma_vorinostat_138Pathways_deltaDMSO_livedead_v1.txt"
    ]

print("Loading Model")
# SVMfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM08_LiveDead_deltaDMSO_138pthway_ALIVE_v2Roni.sav'
SVMfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM03_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni_PredictedSingle.sav'
SVM = pickle.load(open(SVMfilename, 'rb'))

# XGB = XGBClassifier(use_label_encoder=False,max_depth = 5)
# XGBfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB03_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni.model'
# XGBfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB08_LiveDead_deltaDMSO_138pthway_ALIVE_v2Roni_PredictedSingle.model'
# XGB.load_model(XGBfilename)

for infilename in infilenames:
    print(infilename)
    instComb = pd.read_csv(infilename, sep="\t", index_col=0)

    X_test = instComb.to_numpy()
    predsS = SVM.predict(X_test)
    # predsS = XGB.predict(X_test)
    y_predA = predsS
    # y_predA =  [(1 - 0.1 * yrevconv[j]) for j in predsS]
    instComb['non-viability'] = y_predA
    # instComb.to_csv(infilename.replace('_v1.txt','_PREDICTED_SVM08_ALIVE_v2Roni.txt'), sep="\t")
    # instComb.to_csv(infilename.replace('_v1.txt','_PREDICTED_XGB03_DEAD_v2Roni.txt'), sep="\t")
    # instComb.to_csv(infilename.replace('_v1.txt','_PREDICTED_XGB08_ALIVE_v2Roni_PredictedSingle.txt'), sep="\t")
    instComb.to_csv(infilename.replace('_v1.txt','_PREDICTED_SVM03_DEAD_v2Roni_PredictedSingle.txt'), sep="\t")

print("Done.")