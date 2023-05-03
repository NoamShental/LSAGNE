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
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import copy

if os.name == 'nt':
    ppref = 'E:/'
else:
    ppref = '/mnt/e/'

inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v3aRoni.txt', sep="\t", index_col=0)
instData = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2Roni.txt', sep="\t", index_col=0)

testSet = [["MCF7","estriol",-1,-1,-1,-1],["MCF7","wortmannin",-1,-1,-1,-1],["MCF7","raloxifene",-1,-1,-1,-1],
           ["MCF7","vorinostat",-1,-1,-1,-1],["MCF7","tamoxifen",-1,-1,-1,-1],["MCF7","sirolimus",-1,-1,-1,-1],
           ["MCF7","trichostatin-a",-1,-1,-1,-1],
           ["HEPG2", "estriol",-1,-1,-1,-1], ["HEPG2", "wortmannin",-1,-1,-1,-1],["HEPG2", "raloxifene",-1,-1,-1,-1],
           ["HEPG2", "vorinostat",-1,-1,-1,-1], ["HEPG2", "tamoxifen",-1,-1,-1,-1],["HEPG2", "sirolimus",-1,-1,-1,-1],
           ["HEPG2", "trichostatin-a",-1,-1,-1,-1]]

# testSet = [["MCF7","estriol",-1,-1,-1,-1],["MCF7","wortmannin",-1,-1,-1,-1],["MCF7","raloxifene",-1,-1,-1,-1],
#            ["MCF7","vorinostat",-1,-1,-1,-1],["MCF7","tamoxifen",-1,-1,-1,-1],["MCF7","sirolimus",-1,-1,-1,-1],
#            ["MCF7","trichostatin-a",-1,-1,-1,-1]]

# testSet = [["MCF7","raloxifene",-1,-1,-1,-1],["MCF7","sirolimus",-1,-1,-1,-1]]

# inst = inst.sample(frac=1)      #shuffle

instData=instData[instData['cell_id'].isin(['MCF7','HEPG2'])]
# instData=instData[instData['cell_id'].isin(['MCF7'])]
instData=instData[(instData['pert_dose'] >= 5)|(instData['pert_dose'] < 0)]
inst = inst.loc[instData.index]

#take half of the tested data
trainids = []
testids = []
impids=[]
for i in range(len(testSet)):
    cell = testSet[i][0]    # "MCF7"
    ddrug = testSet[i][1]    # 'raloxifene'
    instTe=instData[instData['cell_id'] == cell]
    instTe=instTe[instTe['pert_iname'] == ddrug]
    instTe=instTe[instTe['pert_dose'] >= 5]
    iids=instTe.index
    iidsTe=[i for i in iids if i in inst.index]
    ll=len(iidsTe)
    print(ll)
    ll = 5*int(ll / 6)
    trainids.extend(iidsTe[:ll])
    testids.extend(iidsTe[ll:])
    impids.extend(iidsTe)

restids=[i for i in inst.index if (i not in impids)]
trainids0 = trainids
trainids.extend(restids)

instTe = inst.loc[trainids]
instDe = instData.loc[trainids]
X_train = instTe[inst.columns[1:]].to_numpy()
y_train = instDe['viability'].to_numpy()
y_train[y_train == 1.5] = 1
y_train[y_train == 1.95] = 1.3
y_train[y_train > 1.3] = 1.4
bins0=[0,0.3,0.8,10]
y_trainBinned = np.digitize(y_train,bins0)-1

current_params = {
    'max_depth': 6,
    'colsample_bytree': 0.5,
    'colsample_bylevel': 0.5,
    'subsample': 0.6,
    'learning_rate': 0.01,
}

X_valid0 = []
y_valid0 = []
for i in range(len(testSet)):
    cell = testSet[i][0]    # "MCF7"
    ddrug = testSet[i][1]    # 'raloxifene'
    instTe=instData[instData['cell_id'] == cell]
    instTe=instTe[instTe['pert_iname'] == ddrug]
    instTe=instTe[instTe['pert_dose'] >= 5]
    iids=instTe.index
    iidsTe=[i for i in iids if i in inst.index]

    ll=len(iidsTe)
    ll = 5*int(ll / 6)
    testids2=iidsTe[ll:]

    instTe=inst.loc[testids2]
    instDe = instData.loc[testids2]
    X_valid2 = instTe[instTe.columns[1:]].to_numpy()
    y_valid2 = instDe['viability'].to_numpy()
    y_valid2[y_valid2 == 1.5] = 1
    y_valid2[y_valid2 == 1.95] = 1.3
    y_valid2[y_valid2 > 1.3] = 1.4
    X_valid0.extend(X_valid2)
    y_valid0.extend(y_valid2)


eval_set = [(np.array(X_valid0), np.array(y_valid0))]
y_valid0Binned = np.digitize(y_valid0,bins0)-1
eval_setBinned = [(np.array(X_valid0), np.array(y_valid0Binned))]

# xg_reg = XGBRegressor(objective='reg:squarederror',
#     n_estimators=1000,
#     random_state=RANDOMSTATE,
#     verbosity=1,
#     n_jobs=-1,
#     booster='gbtree',
#     **current_params
# )
#
# xg_reg.fit(X_train,y_train, early_stopping_rounds=10, eval_set=eval_set, verbose=True)

XGB0 = XGBClassifier(use_label_encoder=False,max_depth = 5)
XGB0.fit(X_train, y_trainBinned, early_stopping_rounds=10, eval_set=eval_setBinned, verbose=True)
# XGB0.fit(X_train, y_trainBinned, verbose=True)

SVM0 = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
SVM0.fit(X_train, y_trainBinned)

testSetBinned=copy.deepcopy(testSet)
testSetBinnedSVM=copy.deepcopy(testSet)
for i in range(len(testSet)):
    cell = testSet[i][0]    # "MCF7"
    ddrug = testSet[i][1]    # 'raloxifene'
    print(ddrug)
    instTe=instData[instData['cell_id'] == cell]
    instTe=instTe[instTe['pert_iname'] == ddrug]
    instTe=instTe[instTe['pert_dose'] >= 5]
    iids=instTe.index
    iidsTe=[i for i in iids if i in inst.index]

    ll=len(iidsTe)
    ll = 5*int(ll / 6)
    if(ll==0):
        continue
    trainids2=iidsTe[:ll]
    testids2=iidsTe[ll:]

    instTe=inst.loc[trainids2]
    instDe = instData.loc[trainids2]
    X_valid1 = instTe[instTe.columns[1:]].to_numpy()
    y_valid1 = instDe['viability'].to_numpy()
    y_valid1[y_valid1 == 1.5] = 1
    y_valid1[y_valid1 == 1.95] = 1.3
    y_valid1[y_valid1 > 1.3] = 1.4
    y_valid1Binned = np.digitize(y_valid1,bins0)-1
    # preds = xg_reg.predict(X_valid1)
    # testSet[i][2] = 1-np.mean(y_valid1)
    # testSet[i][3] = 1-np.mean(preds)
    predsB = XGB0.predict(X_valid1)
    testSetBinned[i][2] = np.mean(y_valid1Binned)
    testSetBinned[i][3] = np.mean(predsB)
    predsS = SVM0.predict(X_valid1)
    testSetBinnedSVM[i][2] = np.mean(y_valid1Binned)
    testSetBinnedSVM[i][3] = np.mean(predsS)

    instTe=inst.loc[testids2]
    instDe = instData.loc[testids2]
    X_valid2 = instTe[instTe.columns[1:]].to_numpy()
    y_valid2 = instDe['viability'].to_numpy()
    y_valid2[y_valid2 == 1.5] = 1
    y_valid2[y_valid2 == 1.95] = 1.3
    y_valid2[y_valid2 > 1.3] = 1.4
    y_valid2Binned = np.digitize(y_valid2, bins0)-1
    # preds = xg_reg.predict(X_valid2)
    # testSet[i][4] = 1-np.mean(y_valid2)
    # testSet[i][5] = 1-np.mean(preds)
    predsB = XGB0.predict(X_valid2)
    testSetBinned[i][4] = np.mean(y_valid2Binned)
    testSetBinned[i][5] = np.mean(predsB)
    predsS = SVM0.predict(X_valid2)
    testSetBinnedSVM[i][4] = np.mean(y_valid2Binned)
    testSetBinnedSVM[i][5] = np.mean(predsS)

# result = pd.DataFrame(testSet)
# result.columns=['Cell-line','Drug','train','pred-train','test','pred-test']
# print("XGB Regressor")
# print(result)

resultB = pd.DataFrame(testSetBinned)
resultB.columns=['Cell-line','Drug','train','pred-train','test','pred-test']
print("XGB Classifier, 3 classes")
print(resultB)

resultS = pd.DataFrame(testSetBinnedSVM)
resultS.columns=['Cell-line','Drug','train','pred-train','test','pred-test']
print("SVM Classifier, 3 classes")
print(resultS)

print("Done. Retraining on all data and saving")

instTe = inst
instDe = instData
X_train = instTe[instTe.columns[1:]].to_numpy()
y_train = instDe['viability'].to_numpy()
y_train[y_train == 1.5] = 1
y_train[y_train == 1.95] = 1.3
y_train[y_train > 1.3] = 1.4
y_trainBinned = np.digitize(y_train, bins0)-1

SVM = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
SVM.fit(X_train, y_trainBinned)
outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_SVM3_LiveDead_deltaDMSO_138pthway_ALIVE_v2Roni.sav'
pickle.dump(SVM, open(outfilename, 'wb'))


XGB = XGBClassifier(use_label_encoder=False,max_depth = 5)
XGB.fit(X_train, y_trainBinned, early_stopping_rounds=5, eval_set=eval_setBinned, verbose=False)
outfilename = 'E:/Noam/CMAP/allsamp/to_Michael/BASHAN_XGB3_LiveDead_deltaDMSO_138pthway_ALIVE_v2Roni.sav'
XGB.save_model(outfilename)

print('Done')