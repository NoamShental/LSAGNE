import pandas as pd
import statistics
import os
import numpy as np
import pickle
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from collections import Counter
from sklearn.model_selection import GridSearchCV
import random

#from io import StringIO

#predicted data
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

if os.name == 'nt':
    ppref = 'E:/'
else:
    ppref = '/mnt/e/'
# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_43Pathways_deltaDMSO_livedead_v1.txt', sep="\t", index_col=0)
# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_75Pathways_deltaDMSO_livedead_v2.txt', sep="\t", index_col=0)
# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v2.txt', sep="\t", index_col=0)
# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_27Pathways_deltaDMSO_livedead_v2.txt', sep="\t", index_col=0)
# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_18Pathways_deltaDMSO_livedead_v2.txt', sep="\t", index_col=0)
# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_14Pathways_deltaDMSO_livedead_v2.txt', sep="\t", index_col=0)

# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v3.txt', sep="\t", index_col=0)
inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v3aRoni.txt', sep="\t", index_col=0)

# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_43Pathways_deltaDMSO_livedead_v1f.txt', sep="\t", index_col=0)

# instData = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2.txt', sep="\t", index_col=0)
instData = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2Roni.txt', sep="\t", index_col=0)

# inst = pd.read_csv('E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_decoded_12K_all9drugs_43Pathways_deltaDMSO_livedead_v2.txt', sep="\t", index_col=0)
# inst = pd.read_csv('E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_decoded_12K_all9drugs_43Pathways_deltaDMSO_livedead_v3.txt', sep="\t", index_col=0)
# inst = pd.read_csv('E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v1.txt', sep="\t", index_col=0)
# instData = pd.read_csv('E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.txt', sep="\t", index_col=0)

#make a dictionary lookup for generated samples
instDatadGroupped = instData.groupby(["cell_id","pert_iname","pert_dose"])['viability'].mean().reset_index()
instDatadGroupped = instDatadGroupped[instDatadGroupped["pert_dose"]==10]

#assign the experimental values to the predicted singletons
print("Adding predicted data for singletons to training set, currently :")
print(instData.shape)
for infilename in infilenames:
    print(infilename)
    instComb = pd.read_csv(infilename, sep="\t", index_col=0)
    instCombData = [[k.strip("(").strip(")").strip("'").split(",")[0].split(" ")[0],
                     k.strip("(").strip(")").strip("'").split(",")[1].strip(" '"),
                     k.strip("(").strip(")").strip("'").split(",")[2].strip(" '"),
                     k.strip("(").strip(")").strip("'").split(",")[3].strip(" '"),
                     k] for k in instComb.index]
    instCombData = [k for k in instCombData if ((k[2]=='-')&(k[3]=='-')) ]
    instCombDataF = pd.DataFrame([[k[1],k[0]] for k in instCombData],index=[k[-1] for k in instCombData], columns=["pert_iname","cell_id"])
    vbl = [-1 for k in instCombData]
    for i in range(len(instCombDataF)):
        lkup = instDatadGroupped[(instDatadGroupped["cell_id"]==instCombDataF.iloc[i]["cell_id"])&(instDatadGroupped["pert_iname"]==instCombDataF.iloc[i]["pert_iname"])]
        if lkup.shape[0]<1:
            vbl[i] = -1
        elif lkup.shape[0]==1:
            vbl[i] = float(lkup["viability"])
        else:
            print("Multiple data detected")
            print(i); print(lkup)
    instCombDataF["viability"] = vbl
    instCombDataF = instCombDataF[instCombDataF["viability"] > -1]
    vbl0 = [-1 for k in range(len(instCombDataF))]; instCombDataF["rna_plate"] = vbl0; instCombDataF["rna_well"] = vbl0;
    instCombDataF["pert_id"] = vbl0;instCombDataF["pert_type"] = vbl0;instCombDataF["pert_dose_unit"] = vbl0;
    instCombDataF["pert_time"] = vbl0;instCombDataF["pert_time_unit"] = vbl0;
    instComb = instComb.loc[instCombDataF.index]
    rstr="r{0}".format(random.randint(1,10000))
    # add random string because the names are the same in replicates
    instCombDataF.index = rstr + instCombDataF.index
    instComb.index = rstr + instComb.index

    inst = inst.append(instComb)
    instData = instData.append(instCombDataF)
    print(instData.shape)



print("Hiding 30% of data")
# inst = inst.sample(frac=1)      #shuffle
data = inst[inst.columns[1:]].to_numpy()
instDe = instData.loc[inst.index]

target = instDe['viability'].to_numpy()
target[target == 1.5] = 1
target[target == 1.95] = 1.3
target[target > 1.3] = 1.4

targetD = np.where(target<0.3, 1, 0)    #death predictor
targetA = np.where(target>0.8, 1, 0)    #alive predictor
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(data, targetA, test_size=0.3,random_state=109)

#Create a svm Classifier
clf0A = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
#Train the model using the training sets
clf0A.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf0A.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("SVM Accuracy of life:",1.0*sum([y_pred[i]*y_test[i] for i in range(len(y_test))])/sum(y_pred))
print("SVM Recall:",metrics.recall_score(y_test, y_pred,average='micro'))

#XGBoost classifier
XGB0A = XGBClassifier(use_label_encoder=False)
XGB0A.fit(X_train, y_train)
y_pred = XGB0A.predict(X_test)
print("XGB Base Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("XGB Base Accuracy of life:",1.0*sum([y_pred[i]*y_test[i] for i in range(len(y_test))])/sum(y_pred))
print("XGB Base Recall:",metrics.recall_score(y_test, y_pred,average='micro'))

X_train, X_test, y_train, y_test = train_test_split(data, targetD, test_size=0.3,random_state=109)

#Create a svm Classifier
clf0D = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
#Train the model using the training sets
clf0D.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf0D.predict(X_test)
# Model Accuracy: how often is the classifier correct?
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("SVM Accuracy of death:",1.0*sum([y_pred[i]*y_test[i] for i in range(len(y_test))])/sum(y_pred))
print("SVM Recall:",metrics.recall_score(y_test, y_pred,average='micro'))

#XGBoost classifier
XGB0D = XGBClassifier(use_label_encoder=False)
XGB0D.fit(X_train, y_train)
y_pred = XGB0D.predict(X_test)
print("XGB Base Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("XGB Base Accuracy of death:",1.0*sum([y_pred[i]*y_test[i] for i in range(len(y_test))])/sum(y_pred))
print("XGB Base Recall:",metrics.recall_score(y_test, y_pred,average='micro'))


print("Saving SVM model...")
X_train = data
y_train = targetA

clf0A = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
clf0A.fit(X_train, y_train)
outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM08_LiveDead_deltaDMSO_138pthway_ALIVE_v2Roni_PredictedSingle.sav'
pickle.dump(clf0A, open(outfilename, 'wb'))

XGB0A = XGBClassifier(use_label_encoder=False)
XGB0A.fit(X_train, y_train)
outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB08_LiveDead_deltaDMSO_138pthway_ALIVE_v2Roni_PredictedSingle.sav'
XGB0A.save_model(outfilename.replace('.sav','.model'))

X_train = data
y_train = targetD

clf0D = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
clf0D.fit(X_train, y_train)
outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM03_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni_PredictedSingle.sav'
pickle.dump(clf0D, open(outfilename, 'wb'))

XGB0D = XGBClassifier(use_label_encoder=False)
XGB0D.fit(X_train, y_train)
outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB03_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni_PredictedSingle.sav'
XGB0D.save_model(outfilename.replace('.sav','.model'))


####This is to tune hyperparameter of imbalanced classes - result no need
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(XGB0, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=3)
# print('Mean ROC AUC: %.5f' % np.mean(scores))
# counter = Counter(y_train)
# estimate = counter[0] / counter[1]
# print('Estimate: %.3f' % estimate)
#
# weights = [1, estimate, 10, 25, 50, 75, 100, 1000]
# param_grid = dict(scale_pos_weight=weights)
# grid = GridSearchCV(estimator=XGB0, param_grid=param_grid, n_jobs=3, cv=cv, scoring='roc_auc')
# grid_result = grid.fit(X_train, y_train)
# # report the best configuration
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# # report all configurations
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))



# outfilename = 'E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_SVM_LiveDead_deltaDMSO_138pthway_DEAD_v1.sav'
# outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM_LiveDead_deltaDMSO_138pthway_DEAD_v1.sav'
# outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM_LiveDead_deltaDMSO_138pthway_ALIVE_v1.sav'
# outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni.sav'
# pickle.dump(clf0, open(outfilename, 'wb'))
# outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni.sav'
# pickle.dump(XGB0, open(outfilename, 'wb'))

# print("\n\n\n")
# cells=list(set(instData['cell_id']))
# for cell in cells:
#     print("\n\n\nHiding cel-line {0}".format(cell))
#
#     iids=instData[instData['cell_id']!=cell].index
#     iidsTr=[i for i in iids if i in inst.index]
#     if len(iidsTr)<50:
#         print("Training set too small ({0}), skipping".format(len(iidsTr)))
#         continue
#     instTr=inst.loc[iidsTr]
#     X_train = instTr[instTr.columns[1:]].to_numpy()
#     y_train = instTr['viability'].to_numpy()
#     y_train = np.where(y_train < 0.3, 1, 0)
#     # y_train = np.where(y_train > 0.8, 1, 0)
#
#     iids=instData[instData['cell_id'] == cell].index
#     iidsTe=[i for i in iids if i in inst.index]
#     if len(iidsTe)<50:
#         print("Test set too small ({0}), skipping".format(len(iidsTe)))
#         continue
#     instTe=inst.loc[iidsTe]
#     X_test = instTe[instTe.columns[1:]].to_numpy()
#     y_test = instTe['viability'].to_numpy()
#     y_test = np.where(y_test < 0.3, 1, 0)
#     # y_test = np.where(y_test > 0.8, 1, 0)
#
#     clf = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
#     if(sum(y_pred)==0):
#         print("SVM Accuracy of death: 1" , " sum(y_test)=",sum(y_test))
#     else:
#         print("SVM Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
#     print("SVM Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
#
#     XGB = XGBClassifier(use_label_encoder=False)
#     XGB.fit(X_train, y_train)
#     y_pred = XGB.predict(X_test)
#     print("XGB Accuracy:",metrics.accuracy_score(y_test, y_pred))
#     if(sum(y_pred)==0):
#         print("XGB Accuracy of death: 1" , " sum(y_test)=",sum(y_test))
#     else:
#         print("XGB Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
#     print("XGB Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
#
#     y_pred0 = clf0.predict(X_test)
#     print("SVM Accuracy by base predictor:",metrics.accuracy_score(y_test, y_pred0))
#     if (sum(y_pred0) == 0):
#         print("SVM Accuracy of death: 1", " sum(y_test)=", sum(y_test))
#     else:
#         print("SVM Accuracy of death by base predictor:", 1.0 * sum([y_pred0[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred0))
#     print("SVM Recall by base predictor:",metrics.recall_score(y_test, y_pred0,average='micro'))
#
#     y_pred0 = XGB0.predict(X_test)
#     print("XGB Accuracy by base predictor:", metrics.accuracy_score(y_test, y_pred0))
#     if (sum(y_pred0) == 0):
#         print("XGB Accuracy of death: 1", " sum(y_test)=", sum(y_test))
#     else:
#         print("XGB Accuracy of death by base predictor:",
#               1.0 * sum([y_pred0[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred0))
#     print("XGB Recall by base predictor:", metrics.recall_score(y_test, y_pred0, average='micro'))
#
# print("\n\n\n")
# for cell in cells:
#     print("\n\n\nHiding 50% cel-line {0}".format(cell))
#
#     iids=instData[instData['cell_id'] == cell].index
#     iidsTe=[i for i in iids if i in inst.index]
#     ll=len(iidsTe)
#     ll = int(ll / 2)
#     if len(iidsTe)<50:
#         print("Test set too small ({0}), skipping".format(len(iidsTe)))
#         continue
#     instTe=inst.loc[iidsTe[:ll]]
#     X_test = instTe[instTe.columns[1:]].to_numpy()
#     y_test = instTe['viability'].to_numpy()
#     y_test = np.where(y_test < 0.3, 1, 0)
#     # y_test = np.where(y_test > 0.8, 1, 0)
#
#     iids=instData[instData['cell_id']!=cell].index
#     iidsTr=[i for i in iids if i in inst.index]
#     if len(iidsTr)<50:
#         print("Training set too small ({0}), skipping".format(len(iidsTr)))
#         continue
#     iidsTr.extend(iidsTe[ll:])
#     instTr=inst.loc[iidsTr]
#     X_train = instTr[instTr.columns[1:]].to_numpy()
#     y_train = instTr['viability'].to_numpy()
#     y_train = np.where(y_train < 0.3, 1, 0)
#     # y_train = np.where(y_train > 0.8, 1, 0)
#
#
#     clf = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#     if (sum(y_pred) == 0):
#         print("Accuracy of death: 1", " sum(y_test)=", sum(y_test))
#     else:
#         print("Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
#     print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
#
#     XGB = XGBClassifier(use_label_encoder=False)
#     XGB.fit(X_train, y_train)
#     y_pred = XGB.predict(X_test)
#     print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#     if (sum(y_pred) == 0):
#         print("Accuracy of death: 1", " sum(y_test)=", sum(y_test))
#     else:
#         print("Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
#     print("Recall:", metrics.recall_score(y_test, y_pred, average='micro'))
#
#
#
# print("\n\n\n")
# drugs=list(set(instData['pert_iname']))
# for drug in drugs:
#     print("\n\n\nHiding drug {0}".format(drug))
#
#     iids=instData[instData['pert_iname']!=drug].index
#     iidsTr=[i for i in iids if i in inst.index]
#     if len(iidsTr)<50:
#         print("Training set too small ({0}), skipping".format(len(iidsTr)))
#         continue
#     instTr=inst.loc[iidsTr]
#     X_train = instTr[instTr.columns[1:]].to_numpy()
#     y_train = instTr['viability'].to_numpy()
#     y_train = np.where(y_train < 0.3, 1, 0)
#
#     iids=instData[instData['pert_iname'] == drug].index
#     iidsTe=[i for i in iids if i in inst.index]
#     if len(iidsTe)<50:
#         print("Test set too small ({0}), skipping".format(len(iidsTe)))
#         continue
#     instTe=inst.loc[iidsTe]
#     X_test = instTe[instTe.columns[1:]].to_numpy()
#     y_test = instTe['viability'].to_numpy()
#     y_test = np.where(y_test < 0.3, 1, 0)
#
#     clf = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
#     if (sum(y_pred) == 0):
#         print("SVM Accuracy of death: 1", " sum(y_test)=", sum(y_test))
#     else:
#         print("SVM Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
#     print("SVM Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
#
#     XGB = XGBClassifier(use_label_encoder=False)
#     XGB.fit(X_train, y_train)
#     y_pred = XGB.predict(X_test)
#     print("XGB Accuracy:",metrics.accuracy_score(y_test, y_pred))
#     if(sum(y_pred)==0):
#         print("XGB Accuracy of death: 1" , " sum(y_test)=",sum(y_test))
#     else:
#         print("XGB Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
#     print("XGB Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
#
#
#     y_pred0 = clf0.predict(X_test)
#     print("SVM Accuracy by base predictor:",metrics.accuracy_score(y_test, y_pred0))
#     if (sum(y_pred0) == 0):
#         print("SVM Accuracy of death: 1", " sum(y_test)=", sum(y_test))
#     else:
#         print("SVM Accuracy of death by base predictor:", 1.0 * sum([y_pred0[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred0))
#     print("SVM Recall by base predictor:",metrics.recall_score(y_test, y_pred0,average='micro'))
#
#     y_pred0 = XGB0.predict(X_test)
#     print("XGB Accuracy by base predictor:", metrics.accuracy_score(y_test, y_pred0))
#     if (sum(y_pred0) == 0):
#         print("XGB Accuracy of death: 1", " sum(y_test)=", sum(y_test))
#     else:
#         print("XGB Accuracy of death by base predictor:",
#               1.0 * sum([y_pred0[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred0))
#     print("XGB Recall by base predictor:", metrics.recall_score(y_test, y_pred0, average='micro'))

# if(0):
    # print("\n\n\n")
    # for cell in cells:
    #     print("Training only on cel-line {0}".format(cell))
    #
    #     iids=instData[instData['cell_id'] == cell].index
    #     iidsTe=[i for i in iids if i in inst.index]
    #     if len(iidsTe)<50:
    #         print("Test set too small ({0}), skipping".format(len(iidsTe)))
    #         continue
    #     instTe=inst.loc[iidsTe]
    #     data = instTe[instTe.columns[1:]].to_numpy()
    #     target = instTe['viability'].to_numpy()
    #     target1 = np.where(target < 0.3, 1, 0)
    #
    #     X_train, X_test, y_train, y_test = train_test_split(data, target1, test_size=0.3, random_state=109)
    #     clf = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #     if (sum(y_pred) == 0):
    #         print("Accuracy of death: 1", " sum(y_test)=", sum(y_test))
    #     else:
    #         print("Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
    #     print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))
    #
    #     XGB = XGBClassifier(use_label_encoder=False)
    #     XGB.fit(X_train, y_train)
    #     y_pred = XGB.predict(X_test)
    #     print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #     if(sum(y_pred)==0):
    #         print("Accuracy of death: 1" , " sum(y_test)=",sum(y_test))
    #     else:
    #         print("Accuracy of death:", 1.0 * sum([y_pred[i] * y_test[i] for i in range(len(y_test))]) / sum(y_pred))
    #     print("Recall:",metrics.recall_score(y_test, y_pred,average='micro'))


# print("Saving SVM model...")
# X_train = data
# y_train = target1
# clf0 = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
# clf0.fit(X_train, y_train)
# # outfilename = 'E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_SVM_LiveDead_deltaDMSO_43pthway_v1.sav'
#
# # outfilename = 'E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_SVM_LiveDead_deltaDMSO_183pthway_v2Roni.sav'
# # outfilename = 'E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_SVM_LiveDead_deltaDMSO_183pthway_v2Roni.sav'
# # pickle.dump(clf0, open(outfilename, 'wb'))
#
# # outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni.sav'
#
# XGB0 = XGBClassifier(use_label_encoder=False)
# XGB0.fit(X_train, y_train)
# # outfilename = 'E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_XGB_LiveDead_deltaDMSO_183pthway_v2Roni.sav'
# # pickle.dump(XGB0, open(outfilename, 'wb'))
#
# # outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB_LiveDead_deltaDMSO_138pthway_DEAD_v2Roni.sav'
# outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB_LiveDead_deltaDMSO_138pthway_ALIVE_v2Roni.sav'
# pickle.dump(XGB0, open(outfilename, 'wb'))

print("Done.")
