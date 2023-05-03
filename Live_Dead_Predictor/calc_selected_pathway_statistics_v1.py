import pandas as pd
import statistics
import os
from io import StringIO
import numpy as np
import sys

if os.name == 'nt':
    ppref = 'E:/'
else:
    ppref = '/mnt/e/'
data = pd.read_hdf(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.h5')
inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2.txt', sep="\t", index_col=0)

CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/GO_CellCycle_gid_Biomartx12327.txt',sep="~|\t",index_col=0,header=None)
reliable_CellCyclePathways = CellCyclePathways
CellCyclePathways = pd.read_csv(ppref+'Noam/CMAP/GO_Stress_gid_Biomartx12327.txt', sep="~|\t", index_col=0,header=None)
reliable_CellCyclePathways=pd.concat([reliable_CellCyclePathways,CellCyclePathways],axis=0)
CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/GO_Death_gid_Biomartx12327.txt',sep="~|\t",index_col=0,header=None)
reliable_CellCyclePathways=pd.concat([reliable_CellCyclePathways,CellCyclePathways],axis=0)

CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/All_GO_12327_groupped_6genesPathway_v2.txt',sep="\t",index_col=0,header=None)
reliable_CellCyclePathways=pd.concat([reliable_CellCyclePathways,CellCyclePathways],axis=0)
reliable_CellCyclePathways = reliable_CellCyclePathways.loc[~reliable_CellCyclePathways.index.duplicated(keep='first')]

reliable_pathways = []

# #These are the pathways that show some correlation with viability according to CMAP+CTRPv2
# reliable_pathways.extend(["GO:0000070","GO:0000076","GO:0000084","GO:0000085","GO:0000087","GO:0000727",
#                           "GO:0000729","GO:0002438","GO:0002526","GO:0002675","GO:0006957","GO:0007051",
#                           "GO:0007052","GO:0007077","GO:0007094","GO:0007140","GO:0008628","GO:0008631",
#                           "GO:0009414","GO:0009635","GO:0010544","GO:0019731","GO:0031571","GO:0034121",
#                           "GO:0034145","GO:0042262","GO:0042832","GO:0043068","GO:0043620","GO:0045786",
#                           "GO:0045787","GO:0045954","GO:0046599","GO:0060561","GO:0090201","GO:0090331",
#                           "GO:1900744","GO:1903608","GO:1903753","GO:1990253","GO:2000107",
#                           "GO:2001236","GO:0006921","GO:0010941","GO:0034393","GO:0012501",
#                           "GO:0070244","GO:0043154","GO:0016265","GO:2001241","GO:1902176","GO:0043523",
#                           "GO:1902230","GO:0006978","GO:1900246","GO:0060339","GO:0045824","GO:0006970",
#                           "GO:0034341","GO:0033555","GO:0150076","GO:2000121","GO:1902237","GO:0070417",
#                           "GO:0043152","GO:0072378","GO:0010458","GO:0000090","GO:0007130","GO:0071157",
#                           "GO:0007141","GO:0045840","GO:0040020","GO:0007131","GO:0090307"])
reliable_pathways.extend(["GO:0000070","GO:0000076","GO:0000084","GO:0000085","GO:0000087","GO:0000727","GO:0000729",
                          "GO:0002438","GO:0002526","GO:0002675","GO:0006957","GO:0007051","GO:0007052","GO:0007077",
                          "GO:0007094","GO:0007140","GO:0008628","GO:0008631","GO:0009414","GO:0009635","GO:0010544",
                          "GO:0019731","GO:0031571","GO:0034121","GO:0034145","GO:0042262","GO:0042832","GO:0043068",
                          "GO:0043620","GO:0045786","GO:0045787","GO:0045954","GO:0046599","GO:0060561","GO:0090201",
                          "GO:0090331","GO:1900744","GO:1902237","GO:1903608","GO:1903753","GO:1990253","GO:2000107",
                          "GO:2001236","GO:0006921","GO:0010941","GO:0034393","GO:0012501","GO:0070244","GO:0043154",
                          "GO:0016265","GO:2001241","GO:1902176","GO:0043523","GO:1902230","GO:0006978","GO:1900246",
                          "GO:0060339","GO:0045824","GO:0006970","GO:0034341","GO:0033555","GO:0150076","GO:2000121",
                          "GO:0070417","GO:0043152","GO:0010458","GO:0000090","GO:0007130","GO:0071157",
                          "GO:0007141","GO:0045840","GO:0040020","GO:0007131","GO:0090307","GO:0002504","GO:0010757",
                          "GO:0010891","GO:0030023","GO:0043394","GO:0045110","GO:0048495","GO:0072602",
                          "GO:0098992","GO:0001561","GO:0001758","GO:0001887","GO:0006572",
                          "GO:0019210","GO:0042167","GO:0047023","GO:0090131","GO:0001527","GO:0003337","GO:0030020",
                          "GO:0030195","GO:0048251","GO:0060346","GO:1900028","GO:1905907",
                          "GO:0006069","GO:0019439","GO:0031581","GO:0034137","GO:0048248","GO:0052650",
                          "GO:0085029","GO:0090277","GO:0009308","GO:0030299","GO:0042178",
                          "GO:0051258","GO:0060228","GO:0070508","GO:0003073","GO:0005504","GO:0006068",
                          "GO:0010886","GO:0047676","GO:0050544","GO:0051346","GO:0070633","GO:0071682","GO:2000146",
                          "GO:0005351","GO:0016628","GO:0019640","GO:0030284","GO:0031406","GO:0060687",
                          "GO:0060706","GO:0072178","GO:1904036","GO:0060351","GO:0090027","GO:2000096","GO:0035791",
                          "GO:0045541","GO:0048261"])

reliable_CellCyclePathways = reliable_CellCyclePathways.loc[reliable_pathways]

M = [['' for i in range(reliable_CellCyclePathways.shape[0])] for j in range(reliable_CellCyclePathways.shape[0])]
Mi = np.zeros([reliable_CellCyclePathways.shape[0],reliable_CellCyclePathways.shape[0]])
badlist=[]
for i in range(reliable_CellCyclePathways.shape[0]):
    for j in range(reliable_CellCyclePathways.shape[0]):
        Di=set([int(k) for k in reliable_CellCyclePathways.iloc[i][2].split(',')])
        Dj=set([int(k) for k in reliable_CellCyclePathways.iloc[j][2].split(',')])
        M[i][j]='{0}/{1}'.format(len([i for i in Di if i in Dj]),len(Di))
        Mi[i,j]=100.0*len([i for i in Di if i in Dj])/len(Di)
        if (min(len(Di),len(Dj))<4):
            badlist.append(
                (reliable_CellCyclePathways.iloc[i].name, i, j, reliable_CellCyclePathways.iloc[j].name, len(Dj)))
        if (Mi[i,j]>25):
            if i==j:
                continue
            badlist.append((reliable_CellCyclePathways.iloc[i].name,i,j,reliable_CellCyclePathways.iloc[j].name,len(Dj)))
np.set_printoptions(threshold=sys.maxsize)
# display(Mi)

for i in badlist:
    print('{0} [{1},{2} {3} {4} {5}-{6}]'.format(i[0],i[1],i[2],M[i[1]][i[2]],Mi[i[1],i[2]],i[3],i[4]))