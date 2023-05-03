import pandas as pd
import statistics
import os
from io import StringIO


if os.name == 'nt':
    ppref = 'E:/'
else:
    ppref = '/mnt/e/'
data = pd.read_hdf(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.h5')
data.columns = [str(i).strip() for i in data.columns]
inst = pd.read_csv(ppref+'Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_v2.txt', sep="\t", index_col=0)
reliable_CellCyclePathways = pd.read_csv(ppref+'Noam/CMAP/All_GO_12327_groupped_6genesPathway_v2.txt',sep="\t",index_col=0,header=None)
# reliable_CellCyclePathways = reliable_CellCyclePathways.head(20)      # for testing purposes
cells = list(set(list(inst["cell_id"])))

llst = []
for celline in cells:
    print(celline)
    cl_drug = inst[(inst["cell_id"] == celline) & (inst["pert_iname"] == 'DMSO')]
    cl_drug_vals = data.loc[cl_drug.index]
    for pathway in list(reliable_CellCyclePathways.index):
        genes = [str(i).strip() for i in reliable_CellCyclePathways.loc[pathway][2].split(",")]
        cl_drug_vals_pathway = cl_drug_vals[genes]
        mmean = statistics.mean(list(cl_drug_vals_pathway.mean(axis=1)))
        try:
            mstd = statistics.stdev(list(cl_drug_vals_pathway.mean(axis=1)))
        except:
            mstd = 0
        mviab = statistics.mean(list(cl_drug_vals["viability"]))
        sstr = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(celline, 'DMSO', len(genes), pathway,
                                                     reliable_CellCyclePathways.loc[pathway][1], mmean, mstd, mviab)
        llst.append(sstr)

rres = pd.read_csv(StringIO('\t'.join(["Celline", "Drug", "#genes", "PathwayID", "PathwayDesc", "Mean", "Std", "viability"]) + '\n' + '\n'.join(llst)), sep='\t')
rres=rres.sort_values(['PathwayID','Celline'], ascending=[True, True])    #sort correctly
rres_grouped=rres.groupby(['PathwayID'])

#now go over each pathway and note if its high in specific cell-lines
scells=sorted(cells)
Dcells = {scells[i]:i for i in range(len(scells))}
tlst=[]
for group_name, Cell_group in rres_grouped:
    mx=Cell_group[['Mean']].max()[0]
    mn=Cell_group[['Mean']].min()[0]
    tarr=['']*(len(scells)+2)
    if (100-100.0*mn/mx)>15:
        tcnt=0;ccnt=0
        idxmx=0;maxmn=mn
        for index1, row1 in Cell_group.iterrows():
            if (100-100.0*row1['Mean']/mx)<15:
                tarr[ccnt]=0
                tcnt+=1
            else:
                if row1['Mean']>maxmn:
                    maxmn=row1['Mean']
            if row1['Mean']==mx:
                idxmx=ccnt
            ccnt+=1
        tarr[idxmx]=tcnt
        tarr[-2]=(100-100.0*maxmn/mx)
        tarr[-1]=sum([1 if i!='' else 0 for i in tarr[:-2]])
    tlst.extend([tarr for i in range(Cell_group.shape[0])])

for icell in range(len(scells)):
    rres[scells[icell]]=[tlst[i][icell] for i in range(len(tlst))]

rres["MinDiff"]=[tlst[i][-2] for i in range(len(tlst))]
rres["#lines"]=[tlst[i][-1] for i in range(len(tlst))]

rres.to_csv(ppref+'Noam/CMAP/allsamp/to_Michael/AllGO_9cellines_DMSO_viability_v21.txt', sep="\t")

print("Done.")
