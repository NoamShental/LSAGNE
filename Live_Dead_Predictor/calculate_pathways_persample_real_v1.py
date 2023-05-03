#read data
import pandas as pd
import copy

print("Read Data")
# data=pd.read_hdf('/mnt/e/Noam/CMAP/livedead_12k_ld.h5')
data=pd.read_hdf('/mnt/e/Noam/CMAP/allsamp/to_Michael/decoded_12K_all9drugs_DMSO_livedead_CTD_PRISM_GDSC2_ld_v2.h5')
print("Done. Extracting unique gene ids")
eids = list(data.columns)
eidd = {i for i in eids}
# viabilitycolumn = 'livedead'
viabilitycolumn = 'viability'


def permute_test(ndata, nname, cutoff=0.2,permute_num=1000):
    a=[]
    nmax=ndata[viabilitycolumn].max()
    tdata=ndata[ndata[viabilitycolumn]<cutoff]
    z=tdata["mean"].mean()
    tdata2=ndata[ndata[viabilitycolumn]>(nmax-cutoff)]
    ztop=tdata2["mean"].mean()
    for i in range(permute_num):
        td=ndata.sample(n=tdata.shape[0])
        a.append(td["mean"].mean())
    lessZ=[j for j in a if j < z]
    numVals_lessZ = len(lessZ)
    if numVals_lessZ>(0.95*permute_num):
        # print("pval: {0} Induced {3:0.00}%; Bottom(0.2) {1} vs Top(0.2) {2}".format(1-1.0*numVals_lessZ/1000,z,ztop,(z-ztop)/ztop*100))
        print("{0}\t{4}\t{3}\t{1}\t{2}".format(1-1.0*numVals_lessZ/1000,z,ztop,(z-ztop)/ztop*100,nname))
        return (1-1.0*numVals_lessZ/1000)
    else:
        if numVals_lessZ<(0.05*permute_num):
            # print("pval: {0} Repressed {3:0.00}%; Bottom(0.2) {1} vs Top(0.2) {2}".format(1.0*numVals_lessZ/1000,z,ztop,(ztop-z)/z*100))
            print("{0}\t{4}\t{3}\t{1}\t{2}".format(1.0 * numVals_lessZ / 1000, z, ztop,(ztop - z) / z * 100,nname))
        return (1.0*numVals_lessZ/1000)


finnames=["/mnt/e/Noam/CMAP/GO_CellCycle_gid_Biomartx12327.txt","/mnt/e/Noam/CMAP/GO_Death_gid_Biomartx12327.txt",
            "/mnt/e/Noam/CMAP/GO_Stress_gid_Biomartx12327.txt"]

print("pval\tGO\tinduced\tlow\thigh")
for finname in finnames:
    # print(finname)
    ccnt=0
    with open(finname,'r') as fin:
        for l in fin:
            line=l.split('\t')
            ccnt+=1
            genes=line[1].strip().split(",")
            genes1=[]
            for i in genes:
                if len(i)>0:
                    genes1.append(int(i))
            genes=copy.deepcopy(genes1)
            genes.append(viabilitycolumn)
            # print("Processing {0}".format(line[0].split("~")[0]))
            #print("Processing {0} ({1})".format(line[0].split("~")[0],genes))
            ndata=data[genes]
            ndata["mean"]=ndata[genes1].mean(axis=1)
            ndata["median"]=ndata[genes1].median(axis=1)
            ndata["std"]=ndata[genes1].std(axis=1)
            ndata=ndata[ndata.index.str.contains("24H")]
            crr=ndata[viabilitycolumn].corr(ndata['mean'])
            t=permute_test(ndata,line[0].split("~")[0]+"\t"+finname.split('_')[1]+"\t"+line[0].split("~")[1].split('\t')[0])
            #print("Corr: {0}".format(crr))
            # foutname='/mnt/e/Noam/CMAP/pathways/'+line[0].split("~")[0].replace(":","_")+'_CMAP_12k.txt'
            foutname='/mnt/e/Noam/CMAP/allsamp/to_Michael/pathways/'+line[0].split("~")[0].replace(":","_")+'_CMAP_12k.txt'
            # ndata.to_csv(foutname)



