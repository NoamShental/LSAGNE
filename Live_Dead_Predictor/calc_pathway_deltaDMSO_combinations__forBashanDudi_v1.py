####################################################################################################################
# This is a program to predict cell death based on gene expression
# Here I'm loading all of the expression data from files, but you can get it from anywhere (output of LSAGNE,etc)
# This also needs a corresponding DMSO24 gene expression because we use it as baseline
# For the association between a sample and its DMSO, I am using one of the "inst" files.
# I am grouping the genes into pathways, and so I need a pathway file.
# Finally it uses a stored SVM model, so we need that too.
# This works on the expanded CMAP genes - i.e. 12327. Won't work on 977 landmark genes.
####################################################################################################################
import pandas as pd
import statistics
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import numpy as np
import pickle

if os.name == 'nt':
    ppref = 'E:/'
else:
    ppref = '/mnt/e/'
print("Loading drug-combo expression data (12k genes) ...")     #replace with your own datat here
exp_data = pd.read_hdf(ppref + 'Noam/CMAP/drug_combinations/to_Michael_9_March/to_Michael/drug_combinations_0_58_HCC515 lung carcinoma_wortmannin.hdf')
exp_data.columns = [str(i).strip() for i in exp_data.columns]


print("Loading DMSO expression data  (12k genes) ...")
DMSO_data = pd.read_hdf(ppref + 'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.h5')
DMSO_data.columns = [str(i).strip() for i in DMSO_data.columns]

print("Loading annotations data to know the correspondance ...")
inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2.txt', sep="\t", index_col=0)


print("Loading pathway data ...")
CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/GO_CellCycleStressDeath_gid_Biomartx12327.txt',sep="~|\t",index_col=0,header=None,engine='python')
# Select the magical 43 pathways
reliable_pathways = ["GO:0000070","GO:0000076","GO:0000084","GO:0000085","GO:0000087","GO:0000727",
                          "GO:0000729","GO:0002438","GO:0002526","GO:0002675","GO:0006957","GO:0007051",
                          "GO:0007052","GO:0007077","GO:0007094","GO:0007140","GO:0008628","GO:0008631",
                          "GO:0009414","GO:0009635","GO:0010544","GO:0019731","GO:0031571","GO:0034121",
                          "GO:0034145","GO:0042262","GO:0042832","GO:0043068","GO:0043620","GO:0045786",
                          "GO:0045787","GO:0045954","GO:0046599","GO:0060561","GO:0090201","GO:0090331",
                          "GO:1900744","GO:1902237","GO:1903608","GO:1903753","GO:1990253","GO:2000107",
                          "GO:2001236"]
reliable_CellCyclePathways = CellCyclePathways.loc[reliable_pathways]

print("Loading SVM model ...")
clf0 = pickle.load(open(ppref+'Noam/CMAP/allsamp/to_Michael/SVM_LiveDead_deltaDMSO_43pthway_v1.sav', 'rb'))

#calculate the average values for every selected pathway for DMSO in each cell line
print("Calculating baseline values for pathways in DMSO...")
dmso_data={}
inst_dmso=inst[inst["pert_iname"]=='DMSO']
dmso_grouped=inst_dmso.groupby(['cell_id'])
for group_name, cell_group in dmso_grouped:
    dmso_data[group_name]={}
    print(group_name)
    print(cell_group.shape)
    tdata1=DMSO_data.loc[cell_group.index]
    for pathway in list(reliable_CellCyclePathways.index):
        genes1 = [str(i).strip() for i in reliable_CellCyclePathways.loc[pathway][2].split(",")]
        cell_pathway1 = tdata1[genes1]
        if(len(genes1)!=len(cell_pathway1.columns)):
            print("Something wrong")
        if(len(list(set(genes1)))!=len(genes1)):
            print("Something wrong")
        mmean1 = statistics.mean(list(cell_pathway1.mean(axis=1)))
        dmso_data[group_name][pathway]=mmean1

#calculate difference between pathway in DMSO in same cell line and drug measurement
print("Calculating adjusted pathways in all samples...")
ccnt=0
X_test=[]
for index1, row1 in exp_data.iterrows():
    ccnt+=1
    if ((ccnt % 100) == 0):
        print(ccnt)
    tdata2 = row1
    rdata=[]
    for pathway1 in list(reliable_CellCyclePathways.index):
        genes2 = [str(i).strip() for i in reliable_CellCyclePathways.loc[pathway1][2].split(",")]
        cell_pathway2 = tdata2.loc[genes2]
        if(len(genes2)!=cell_pathway2.shape[0]):
            print("Something wrong")
        if(len(list(set(genes2)))!=len(genes2)):
            print("Something wrong")
        mmean2 = cell_pathway2.mean()
        rdata.append(dmso_data[index1[0].split(' ')[0]][pathway1]-mmean2)
    X_test.append(rdata)

#calculate death
print("Calculating Death in all adjusted samples...")
X_test = np.array(X_test)
y_pred = clf0.predict(X_test)

print("Done.")
