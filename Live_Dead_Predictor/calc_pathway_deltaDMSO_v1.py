import pandas as pd
import statistics
import os
from io import StringIO


if os.name == 'nt':
    ppref = 'E:/'
else:
    ppref = '/mnt/e/'

data1 = pd.read_hdf(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2Roni.h5')
# data1 = pd.read_hdf(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.h5')
# data1 = pd.read_hdf('E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.h5')
data1.columns = [str(i).strip() for i in data1.columns]

inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2Roni.txt', sep="\t", index_col=0)
# inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2.txt', sep="\t", index_col=0)

# inst = pd.read_csv('E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.txt', sep="\t", index_col=0)

# inst.drop(inst[inst['pert_iname']=='isonicotinohydroxamic-acid'].index, inplace=True)     #these are bad measurements
# inst.drop(inst[inst['pert_iname']=='raloxifene'].index, inplace=True)     #these are bad measurements

CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/GO_CellCycle_gid_Biomartx12327.txt',sep="~|\t",index_col=0,header=None,engine='python')
reliable_CellCyclePathways = CellCyclePathways
CellCyclePathways = pd.read_csv(ppref+'Noam/CMAP/GO_Stress_gid_Biomartx12327.txt', sep="~|\t", index_col=0,header=None,engine='python')
reliable_CellCyclePathways=pd.concat([reliable_CellCyclePathways,CellCyclePathways],axis=0)
CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/GO_Death_gid_Biomartx12327.txt',sep="~|\t",index_col=0,header=None,engine='python')
reliable_CellCyclePathways=pd.concat([reliable_CellCyclePathways,CellCyclePathways],axis=0)

CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/All_GO_12327_groupped_6genesPathway_v2.txt',sep="\t",index_col=0,header=None)
reliable_CellCyclePathways=pd.concat([reliable_CellCyclePathways,CellCyclePathways],axis=0)
reliable_CellCyclePathways = reliable_CellCyclePathways.loc[~reliable_CellCyclePathways.index.duplicated(keep='first')]

reliable_pathways = []
# #43
# reliable_pathways.extend(["GO:0000070","GO:0000076","GO:0000084","GO:0000085","GO:0000087","GO:0000727",
#                           "GO:0000729","GO:0002438","GO:0002526","GO:0002675","GO:0006957","GO:0007051",
#                           "GO:0007052","GO:0007077","GO:0007094","GO:0007140","GO:0008628","GO:0008631",
#                           "GO:0009414","GO:0009635","GO:0010544","GO:0019731","GO:0031571","GO:0034121",
#                           "GO:0034145","GO:0042262","GO:0042832","GO:0043068","GO:0043620","GO:0045786",
#                           "GO:0045787","GO:0045954","GO:0046599","GO:0060561","GO:0090201","GO:0090331",
#                           "GO:1900744","GO:1902237","GO:1903608","GO:1903753","GO:1990253","GO:2000107",
#                           "GO:2001236"])

#75
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

#138
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

#27
# reliable_pathways.extend(["GO:0000070","GO:0000076","GO:0000087","GO:0000090","GO:0000720","GO:0002903",
#                           "GO:0007052","GO:0034121","GO:0034134","GO:0042262","GO:0051321","GO:0090201",
#                           "GO:1902177","GO:1902237","GO:1902262","GO:2000107","GO:2000121","GO:2001236",
#                           "GO:0030284","GO:0031581","GO:0034363","GO:0042613","GO:0047023","GO:0048261",
#                           "GO:0048407","GO:0051346","GO:0072178"])

#18
# reliable_pathways.extend(["GO:0000070","GO:0000076","GO:0000087","GO:0000090","GO:0000720","GO:0002903",
#                           "GO:0007052","GO:0034121","GO:0034134","GO:0042262","GO:0051321","GO:0090201",
#                           "GO:1902177","GO:1902237","GO:1902262","GO:2000107","GO:2000121","GO:2001236"])

#14
# reliable_pathways.extend(["GO:0000070","GO:0000076","GO:0000084","GO:0000087","GO:0002438","GO:0006957",
#                           "GO:0007052","GO:0007084","GO:0007094","GO:0008628","GO:0045954","GO:0070231",
#                           "GO:1900744","GO:2000271"])
#
reliable_CellCyclePathways = reliable_CellCyclePathways.loc[reliable_pathways]

#calculate the average values for every selected pathway for DMSO in each cell line
print("Calculating baseline values for pathways in DMSO...")
dmso_data={}
inst_dmso=inst[inst["pert_iname"]=='DMSO']
dmso_grouped=inst_dmso.groupby(['cell_id'])
for group_name, cell_group in dmso_grouped:
    dmso_data[group_name]={}
    print(group_name)
    print(cell_group.shape)
    # tdata=data.loc[cell_group.index]
    tdata1=data1.loc[cell_group.index]
    for pathway in list(reliable_CellCyclePathways.index):
        genes1 = [str(i).strip() for i in reliable_CellCyclePathways.loc[pathway][2].split(",")]
        # genes = [int(i) for i in reliable_CellCyclePathways.loc[pathway][2].split(",")]
        # cell_pathway = tdata[genes]
        cell_pathway1 = tdata1[genes1]
        # if(len(genes)!=len(cell_pathway.columns)):
        #     print("Something wrong")
        # if(len(list(set(genes)))!=len(genes)):
        #     print("Something wrong")
        if(len(genes1)!=len(cell_pathway1.columns)):
            print("Something wrong")
        if(len(list(set(genes1)))!=len(genes1)):
            print("Something wrong")
        # mmean = statistics.mean(list(cell_pathway.mean(axis=1)))
        mmean1 = statistics.mean(list(cell_pathway1.mean(axis=1)))
        # if (mmean!=mmean1):
        #     print("Something wrong")
        dmso_data[group_name][pathway]=mmean1

#now create training set with the pathways as features and viability as the outcome
#use difference between pathway in DMSO in same cell line and drug measurement
print("Calculating adjusted pathways in all samples...")
# inst_drugs=inst[inst["pert_iname"] != 'DMSO']           #its unclear whether this is a mistake
inst_drugs=inst
headers = "inst_id\tviability"
for pathway in list(reliable_CellCyclePathways.index):
    headers += "\t{0}".format(pathway)

llst = [headers+'\n']
ccnt=0
for index1, row1 in inst_drugs.iterrows():
    ccnt+=1
    if ((ccnt % 100) == 0):
        print(ccnt)
    # tdata1 = data.loc[index1]
    tdata1a = data1.loc[index1]
    viab = row1['viability']
    sstr = "{0}\t{1}".format(index1,viab)
    for pathway1 in list(reliable_CellCyclePathways.index):
        genes1a = [str(i).strip() for i in reliable_CellCyclePathways.loc[pathway1][2].split(",")]
        # genes = [int(i) for i in reliable_CellCyclePathways.loc[pathway1][2].split(",")]
        # cell_pathway = tdata1.loc[genes]
        cell_pathway1a = tdata1a.loc[genes1a]
        # if(len(genes)!=cell_pathway.shape[0]):
        #     aa=1
        # if(len(list(set(genes)))!=len(genes)):
        #     aa=1
        if(len(genes1a)!=cell_pathway1a.shape[0]):
            print("Something wrong")
        if(len(list(set(genes1a)))!=len(genes1a)):
            print("Something wrong")
        # mmean1 = cell_pathway.mean()
        mmean1a = cell_pathway1a.mean()
        # if (mmean1!=mmean1a):
        #     print("Something wrong")
        sstr+="\t{0}".format(dmso_data[row1['cell_id']][pathway1]-mmean1a)
    llst.append(sstr)

rres = pd.read_csv(StringIO('\n'.join(llst)), sep='\t',index_col=0)
rres.to_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v3aRoni.txt', sep="\t")
# rres.to_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v3a.txt', sep="\t")
# rres.to_csv('E:/Noam/CMAP/drug_combinations/DUDI/AlphaDrugCombinations-MCF7_wortmannin/DUDI_decoded_12K_all9drugs_138Pathways_deltaDMSO_livedead_v1.txt', sep="\t")

print("Done.")
