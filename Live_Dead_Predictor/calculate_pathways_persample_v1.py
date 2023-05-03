#read data
import pandas as pd
print("Read Data")
data=pd.read_hdf('/mnt/e/Noam/CMAP/12327_genes.h5')
print("Done. Extracting unique gene ids")
eids=list(data.columns)

# finname="E:/Noam/CMAP/12327_entrezIDs.txt"
# eids=[]
# with open(finname,'r') as fin:
    # line=fin.readline()
    # eids=line.split(',')

eidd={i for i in eids}

finname="/mnt/e/Noam/CMAP/GO_Stress_gid_Biomart.txt"
foutname="/mnt/e/Noam/CMAP/GO_Stress_gid_Biomartx12327.txt"
fout=open(foutname,'w')
nPth={}
ccnt=0
with open(finname,'r') as fin:
    for l in fin:
        line=l.split('\t')
        ccnt+=1
        if line[0] in nPth:
            print("Duplicate pathway found")
            continue
        oldGenes=line[1].strip().split(",")
        oldGenes1=[]
        for i in oldGenes:
            if len(i)>0:
                oldGenes1.append(int(i))
        oldGenes=oldGenes1
        newGenes=[i for i in oldGenes if i in eidd]
        if len(newGenes)<4:
            print("Skipping {0}. Not enough genes remain in the pathway (was {1}, remained {2})".format(line[0],len(oldGenes),len(newGenes)))
            continue
        if 1.0*len(newGenes)/len(oldGenes)<0.6:
            print("Skipping {0}. Relatively too many genes lost ({3})from the pathway (was {1}, remains {2})".format(line[0],
            len(oldGenes),len(newGenes),1.0*len(newGenes)/len(oldGenes)))
            continue
        nPth[line[0]]=newGenes
        fout.write("{0}\t{1}\n".format(line[0],",".join([str(i) for i in newGenes])))

print("{0} pathways out of {1} remained".format(len(nPth),ccnt))
fout.close()

