import pandas as pd
import statistics
import os
from io import StringIO


if os.name == 'nt':
    ppref = 'E:/'
else:
    ppref = '/mnt/e/'
data = pd.read_hdf(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.h5')
inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2.txt', sep="\t", index_col=0)
CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/GO_CellCycle_gid_Biomartx12327.txt',sep="~|\t",index_col=0,header=None)
# CellCyclePathways = pd.read_csv(ppref+'Noam/CMAP/GO_Stress_gid_Biomartx12327.txt', sep="~|\t", index_col=0,header=None)
# CellCyclePathways=pd.read_csv(ppref+'Noam/CMAP/GO_Death_gid_Biomartx12327.txt',sep="~|\t",index_col=0,header=None)

reliable_pathways = []
# #These are the cell cycle genes that go up with viability measure according to CMAP+CTRPv2
# reliable_pathways.extend(["GO:0000087","GO:0000070","GO:0000084","GO:0000085","GO:0007051",
#                           "GO:0000076","GO:0007084","GO:0007091",
#                           "GO:0007094","GO:0007052","GO:0007077","GO:0090267",
#                           "GO:0051436","GO:0051439","GO:0000278","GO:0000132","GO:0051437","GO:0000082",
#                           "GO:0007144","GO:0007067","GO:0007093","GO:0046599","GO:0000083"])

# # These are the death genes that go down opposite to viability measure according to CMAP+CTRPv2
# reliable_pathways.extend(["GO:0043068","GO:0097527","GO:0070233","GO:2000352","GO:0008628","GO:0070244",
#                           "GO:0070231","GO:1901215","GO:2001240","GO:2001054","GO:1902176","GO:1900118",
#                           "GO:0070265","GO:0043525","GO:2000353","GO:0016265","GO:1902262","GO:0043524",
#                           "GO:0043154"])

#These are the stress genes that go down opposite to viability measure according to CMAP+CTRPv2
# reliable_pathways.extend(["GO:0038124","GO:0002523","GO:0048143","GO:0006957","GO:1900744","GO:1903753",
#                           "GO:0090331","GO:0009635","GO:0001774","GO:0061428","GO:0019732","GO:0031102",
#                           "GO:0050832","GO:0002526","GO:0007256","GO:0034405","GO:0009414","GO:1900069",
#                           "GO:0071447","GO:0001867","GO:0009651","GO:0043152","GO:0006968","GO:0002729",
#                           "GO:0010544","GO:1990253","GO:0046328","GO:0002544","GO:0002438","GO:0045953",
#                           "GO:0002674","GO:0006953","GO:0034141","GO:0051918","GO:0019731","GO:0072378",
#                           "GO:1900015","GO:0061042","GO:0045088","GO:0150076","GO:0009267","GO:0045954",
#                           "GO:0006972","GO:0002221","GO:0006952","GO:0042270","GO:0060266","GO:0071499",
#                           "GO:0150077","GO:0002224","GO:0030168","GO:0000303","GO:0006954","GO:0050729",
#                           "GO:0071456","GO:2000780","GO:0009409","GO:0033555","GO:0071455","GO:0007597",
#                           "GO:0031103","GO:0002931","GO:0042832","GO:1904294","GO:0034145","GO:0035313"])

# reliable_CellCyclePathways = CellCyclePathways.loc[reliable_pathways]
reliable_CellCyclePathways = CellCyclePathways

cells = list(set(list(inst["cell_id"])))


def calcPathwayPerSample(pathway, sample):
    return


res = {}
llst = []
for celline in cells:
    print(celline)
    cl = inst[inst["cell_id"] == celline]
    drugs = list(set(list(cl["pert_iname"]+"|"+cl["pert_dose"].astype(str))))
    res[celline] = {}
    for drug in drugs:
        print(drug)
        res[celline][drug] = {}
        cl_drug = cl[(cl["pert_iname"] == drug.split('|')[0]) & (cl["pert_dose"] == float(drug.split('|')[1]))]
        cl_drug_vals = data.loc[cl_drug.index]
        for pathway in list(reliable_CellCyclePathways.index):
            genes = [int(i) for i in reliable_CellCyclePathways.loc[pathway][2].split(",")]
            cl_drug_vals_pathway = cl_drug_vals[genes]
            mmean = statistics.mean(list(cl_drug_vals_pathway.mean(axis=1)))
            try:
                mstd = statistics.stdev(list(cl_drug_vals_pathway.mean(axis=1)))
            except:
                mstd = 0
            mviab = statistics.mean(list(cl_drug_vals["viability"]))
            res[celline][drug][pathway] = [mmean, mstd]
            sstr = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(celline, drug.split('|')[0], drug.split('|')[1], pathway,
                                                         reliable_CellCyclePathways.loc[pathway][1], mmean, mstd, mviab)
            llst.append(sstr)
            # print(sstr)

rres = pd.read_csv(StringIO('\t'.join(["Celline", "Drug", "Dose", "PathwayID", "PathwayDesc", "Mean", "Std", "viability"]) + '\n' + '\n'.join(llst)), sep='\t')
rres=rres.sort_values(['Celline', 'PathwayID','Mean'], ascending=[True, True, False])    #sort correctly
rres.drop(rres[rres['Drug']=='isonicotinohydroxamic-acid'].index, inplace=True)     #these are bad measurements
rres.drop(rres[rres['Drug']=='raloxifene'].index, inplace=True)     #these are bad measurements
rres_grouped=rres.groupby(['Celline', 'PathwayID'])
tlst=[]
tlst2=[]
for group_name, DrugCell_group in rres_grouped:
    # print(group_name)
    t1=DrugCell_group[['Mean','viability']].corr()['Mean'][1]
    t2=100-100*DrugCell_group[['Mean']].min()[0]/DrugCell_group[['Mean']].max()[0]
    # print(t1)
    tlst.extend([t1 for i in range(DrugCell_group.shape[0])])
    tlst2.extend([t2 for i in range(DrugCell_group.shape[0])])
rres['Corr_MeanViability_CellinePathway']=tlst
rres['FC_Min_Max']=tlst2
# rres = pd.DataFrame(llst, ["Celline", "Drug", "PathwayID", "PathwayDesc", "Mean", "Std"])
# rres.to_csv(ppref+'Noam/CMAP/BASHAN_PATHWAYtest_res_v1.txt',sep="\t")
rres.to_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_CellCycle_res_v22.txt', sep="\t")

print("Done.")
