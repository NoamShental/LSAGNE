#read data
import argparse
import lzma
import os
import pickle
import warnings
from dataclasses import dataclass
from os.path import exists
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn import preprocessing

from src.os_utilities import create_dir_if_not_exists
from src.perturbation import Perturbation
from src.tissue import Tissue

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'invalid value encountered in sqrt')

# import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AugmentationDbCreationParameters:
    raw_data_augmentation_dir: Path
    raw_cmap_dir: Path
    min_drug_samples_per_cellline: int
    min_cellines_perdrug: int
    min_genes_per_go: int
    max_genes_per_go: int
    drug_batch_size: int
    use_compression: bool
    calc_beta: bool
    output_dir: Path
    use_drugs: list[Perturbation]

    @classmethod
    def create_using_args(cls):
        parser = argparse.ArgumentParser(description="Generating ...")

        parser.add_argument(
            "--min-drug-samples-per-cellline",
            type=int,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--min-cellines-perdrug",
            type=int,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--min-genes-per-go",
            type=int,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--max-genes-per-go",
            type=int,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--drug-batch-size",
            type=int,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--use-compression",
            action="store_true",
            help="Works but slows down considerably"
        )
        parser.add_argument(
            "--calc-beta",
            action="store_true",
            help="Works but slows down considerably, I see no real benefit, everything is approx normal"
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            required=True,
            help="The output folder"
        )
        parser.add_argument(
            "--raw-cmap-dir",
            type=Path,
            required=True,
            help="The raw CMAP folder"
        )
        parser.add_argument(
            "--raw-data-augmentation-dir",
            type=Path,
            required=True,
            help="The raw data augmentation folder"
        )
        parser.add_argument(
            "--use-drugs",
            type=Perturbation,
            nargs='+',
            default=None,
            help=""
        )
        # parser.add_argument(
        #     "--use-cell-lines",
        #     type=Tissue,
        #     nargs='+',
        #     default=None,
        #     help=""
        # )
        args = parser.parse_args()
        return cls(
            raw_data_augmentation_dir=args.raw_data_augmentation_dir,
            raw_cmap_dir=args.raw_cmap_dir,
            min_drug_samples_per_cellline=args.min_drug_samples_per_cellline,
            min_cellines_perdrug=args.min_cellines_perdrug,
            min_genes_per_go=args.min_genes_per_go,
            max_genes_per_go=args.max_genes_per_go,
            drug_batch_size=args.drug_batch_size,
            use_compression=args.use_compression,
            calc_beta=args.calc_beta,
            output_dir=args.output_dir,
            use_drugs=args.use_drugs
        )

def generate_augmentation_db(
    args: AugmentationDbCreationParameters
):
    # definitions

    #these are required
    CMAP_ANNOT_FILE = args.raw_data_augmentation_dir / 'GSE92742_Broad_LINCS_inst_info_filt_DMSO-trt_6h-24h_DoseOver2.txt'
    CMAP_GENEXP_FILE = args.raw_cmap_dir / 'data.h5'
    ALL_GO_FILE = args.raw_data_augmentation_dir / 'All_GO_v2.txt'

    #each of next two can be empty if don't want to restrict
    # USE_DRUGS_FILE = None
    # USE_CELLINES_FILE = None
    USE_DRUGS_FILE = args.raw_data_augmentation_dir / 'use_drugs.txt'
    USE_CELLINES_FILE = args.raw_data_augmentation_dir / 'use_cell_lines.txt'

    MIN_DRUG_SAMPLES_PER_CELLLINE = args.min_drug_samples_per_cellline
    MIN_CELLINES_PERDRUG = args.min_cellines_perdrug
    MIN_GENES_PER_GO = args.min_genes_per_go
    MAX_GENES_PER_GO = args.max_genes_per_go
    DRUG_BATCH_SIZE = args.drug_batch_size
    USE_COMPRESSION = args.use_compression
    CALC_BETA = args.calc_beta

    # output
    create_dir_if_not_exists(args.output_dir)
    OUT_GO_FILE = args.output_dir / 'GO_Over3genesLess50_from977_v22.txt'
    OUT_DRUGS_CELLS_FILE = args.output_dir / 'drugsOver6reps_cellinesOver4_v22.txt'
    OUT_DATA_PREFIX = os.path.join(args.output_dir, 'data_augment_cmap_db_{}_v22.pkl')

    print("Read CMAP annotation data")
    cmap_annot=pd.read_csv(CMAP_ANNOT_FILE, sep="\t",index_col=0)

    print("Read gene expression data")
    data = pd.read_hdf(CMAP_GENEXP_FILE).T

    print("Reduce annotation data to current CMAP genex")
    cmap_annot=cmap_annot[cmap_annot.index.isin(data.columns)]

    if args.use_drugs:
        target_drugs = args.use_drugs
    elif USE_DRUGS_FILE:
        target_drugs = pd.read_csv(USE_DRUGS_FILE, sep="\t", index_col=None, header= None)[0].to_list()
    else:
        raise RuntimeError('No drugs defined')
    print(f'Filtering for drugs ({target_drugs})...')
    cmap_annot = cmap_annot[cmap_annot["pert_iname"].isin(target_drugs)]

    if USE_CELLINES_FILE:
        target_cells = pd.read_csv(USE_CELLINES_FILE, sep="\t", index_col=None, header= None)[0].to_list()
        print(f'Filtering for cell-lines ({target_cells})...')
        cmap_annot = cmap_annot[cmap_annot["cell_id"].isin(target_cells)]

    if 'DMSO' in target_drugs[0]:
        cmap_annot.loc[(cmap_annot['pert_time'] == 24) & (cmap_annot['pert_iname'] == 'DMSO'), 'pert_iname'] = 'time 24h'

    # filter to drugs that have at least 6 repeats on at least 4 cell-lines
    cmap_annot_cell_drug = cmap_annot.groupby(["pert_iname","cell_id"]).size().reset_index(name='counts')
    cmap_annot_cell_drug = cmap_annot_cell_drug[cmap_annot_cell_drug["counts"]>=MIN_DRUG_SAMPLES_PER_CELLLINE]
    cmap_annot_drug = cmap_annot_cell_drug.groupby(["pert_iname"])['cell_id'].apply(','.join).reset_index()
    cmap_annot_drug["cell_lines"] = [len(x.split(',')) for x in cmap_annot_drug["cell_id"]]
    cmap_annot_drug = cmap_annot_drug[cmap_annot_drug["cell_lines"]>=MIN_CELLINES_PERDRUG]
    print("Found {} drugs with {} or more repeats in {} or more cell-lines".format(cmap_annot_drug.shape[0],
                                                                                   MIN_DRUG_SAMPLES_PER_CELLLINE,
                                                                                   MIN_CELLINES_PERDRUG))
    cln = list(set(','.join(cmap_annot_drug["cell_id"]).split(',')))
    cmap_annot_f = cmap_annot[cmap_annot["pert_iname"].isin(cmap_annot_drug["pert_iname"])]
    cmap_annot_f = cmap_annot_f[cmap_annot_f["cell_id"].isin(cln)]

    data=data[cmap_annot_f.index].T
    data = data.astype(np.float64)
    # Michael - fill all outliers above 95% quantile or below 5% with corresponding quantile values
    data = data.apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))
    # Scale the data to 0-1.
    scaler = preprocessing.MinMaxScaler()
    # self.scaler = preprocessing.StandardScaler()
    scaled_data_np = scaler.fit_transform(data)
    data = pd.DataFrame(scaled_data_np, columns=data.columns, index=data.index).T


    print("Loaded {} samples".format(data.shape[1]))
    print("Done.")

    print("Reading GO pathways")
    GO=pd.read_csv(ALL_GO_FILE, sep="\t",index_col=False, dtype=str)
    GO = GO[GO["GO term accession"].notnull()]
    cmapgenes = list(data.index)
    GO = GO[GO["NCBI gene ID"].isin(cmapgenes)]
    GO.drop_duplicates(keep = False, inplace = True)
    GO = GO.groupby(["GO term accession"])['NCBI gene ID'].apply(','.join).reset_index()
    GO["count"] = [len(set(x.split(','))) for x in GO["NCBI gene ID"]]
    GO = GO[GO["count"]>=MIN_GENES_PER_GO]; GO = GO[GO["count"] <= MAX_GENES_PER_GO];
    GO.columns = ["GO","Genes","Count"]
    GO2 = GO.copy(deep=True)

    print("Found {} pathways with at least {} genes and less than {} from the 977 CMAP".format(
        GO2.shape[0],MIN_GENES_PER_GO,MAX_GENES_PER_GO))

    cmap_annot_drug.to_csv(OUT_DRUGS_CELLS_FILE,sep="\t")
    GO.to_csv(OUT_GO_FILE,sep="\t")

    #skip precomputed parts
    ccnt0 = 0
    for ccnt0 in range(0,cmap_annot_drug.shape[0],DRUG_BATCH_SIZE):
        if (exists(OUT_DATA_PREFIX.format((ccnt0) // DRUG_BATCH_SIZE))):
            print("Found precomputed chunk {} , skipping...".format(ccnt0 // DRUG_BATCH_SIZE))
            continue
        else:
            break
    ccnt0 = ccnt0 - DRUG_BATCH_SIZE

    ccnt = 0
    res ={};res["GO_names"] = GO2["GO"]; res["gene_names"] = cmapgenes
    for index, row in cmap_annot_drug.iterrows():
        if ccnt <= ccnt0:
            ccnt = ccnt + 1
            continue
        if((ccnt % DRUG_BATCH_SIZE)==0):
            if ccnt > 0:
                print("Saving chunk {} / {} ".format(ccnt, cmap_annot_drug.shape[0]))
                if USE_COMPRESSION:
                    with lzma.open(OUT_DATA_PREFIX.format(ccnt // DRUG_BATCH_SIZE), "wb") as f:
                        pickle.dump(res, f)
                else:
                    with open(OUT_DATA_PREFIX.format(ccnt // DRUG_BATCH_SIZE), "wb") as f:
                        pickle.dump(res, f)
                res = {}; res["GO_names"] = GO2["GO"]; res["gene_names"] = cmapgenes
        pert = row['pert_iname']
        print(pert)
        ccnt = ccnt+1
        cellines = row['cell_id'].split(',')
        for cell in cellines:
            samples = cmap_annot_f[(cmap_annot_f["pert_iname"]==pert) & (cmap_annot_f["cell_id"]==cell)]
            print(cell,samples.shape[0])
            res[(pert,cell)] = {}
            tdata = data[samples.index].copy(deep=True)
            tdata2 = tdata.copy(deep=True).transpose()

            # pathways
            GO3 = GO2.copy(deep=True)
            GO3["mean"] = 0.0;GO3["std"] = 0.0;
            for index1, row1 in GO3.iterrows():
                genes = row1["Genes"].split(',')
                cdata = tdata.loc[genes].copy(deep=True).transpose()
                cdata["mean"] = cdata.mean(axis=1)
                GO3._set_value(index1, "mean", np.mean(cdata["mean"]))
                GO3._set_value(index1, "std", np.std(cdata["mean"]))

                if CALC_BETA:
                    params = beta.fit(cdata["mean"])
                    #test we're ok
                    # rv2 = beta.rvs(params[0], params[1], loc=params[2], scale=params[3], size=len(cdata["mean"]))
                    # rv3 = norm.rvs(np.mean(cdata["mean"]), np.std(cdata["mean"]), size=len(cdata["mean"]))
                    # plt.hist(rv2, color="red", alpha=0.2, label='beta'); plt.hist(cdata["mean"], color="blue", alpha=0.2, label='hist'); plt.hist(rv3, color="green", alpha=0.2, label='norm'); plt.show()

                    GO3._set_value(index1, "Beta_a", params[0])
                    GO3._set_value(index1, "Beta_b", params[1])
                    GO3._set_value(index1, "Beta_loc", params[1])
                    GO3._set_value(index1, "Beta_scale", params[1])
                else:
                    GO3._set_value(index1, "Beta_a", 0)
                    GO3._set_value(index1, "Beta_b", 0)
                    GO3._set_value(index1, "Beta_loc", 0)
                    GO3._set_value(index1, "Beta_scale", 0)

                tdata2[row1["GO"]] = cdata["mean"]
            tdata2b = tdata2[tdata2.columns[977:]]
            corm = tdata2b.corr()
            res[(pert, cell)]["GO_vals"] = GO3[["mean", "std", "Beta_a", "Beta_b", "Beta_loc", "Beta_scale"]].to_numpy()
            res[(pert, cell)]["GO_corr"] = corm.to_numpy()

            #single genes
            if CALC_BETA:
                a=[];b=[];floc=[];scale=[]
                for i in tdata.index:
                    params = beta.fit(tdata.loc[i])
                    a.append(params[0]);b.append(params[1]);floc.append(params[2]);scale.append(params[3])
                    # rv2 = beta.rvs(params[0], params[1], loc=params[2], scale=params[3], size=len(tdata.loc[i]))
                    # rv3 = norm.rvs(np.mean(tdata.loc[i]), np.std(tdata.loc[i]), size=len(tdata.loc[i]))
                    # plt.hist(rv2, color="red", alpha=0.2, label='beta'); plt.hist(tdata.loc[i], color="blue", alpha=0.2, label='hist'); plt.hist(rv3, color="green", alpha=0.2, label='norm'); plt.show()
            else:
                a = [0]*tdata.shape[0]; b = [0]*tdata.shape[0]; floc = [0]*tdata.shape[0]; scale = [0]*tdata.shape[0]

            tdata["mean"] = tdata.mean(axis=1)
            tdata["std"] = tdata.std(axis=1)
            tdata["Beta_a"] = a; tdata["Beta_b"] = b; tdata["Beta_loc"] = floc;tdata["Beta_scale"] = scale;
            tdata2a = tdata2[tdata2.columns[:977]]
            corm = tdata2a.corr()
            res[(pert, cell)]["Gene_vals"] = tdata[["mean","std", "Beta_a", "Beta_b", "Beta_loc", "Beta_scale"]].to_numpy()
            res[(pert, cell)]["Gene_corr"] = corm.to_numpy()
            print(res[(pert, cell)]["Gene_vals"].shape)
            print(res[(pert, cell)]["GO_vals"].shape)

    #save last chunk
    print("Saving chunk {} / {} ".format(ccnt, cmap_annot_drug.shape[0]))
    if USE_COMPRESSION:
        with lzma.open(OUT_DATA_PREFIX.format((ccnt+DRUG_BATCH_SIZE) // DRUG_BATCH_SIZE), "wb") as f:
            pickle.dump(res, f)
    else:
        with open(OUT_DATA_PREFIX.format((ccnt + DRUG_BATCH_SIZE) // DRUG_BATCH_SIZE), "wb") as f:
            pickle.dump(res, f)


if __name__ == '__main__':
    args = AugmentationDbCreationParameters.create_using_args()
    generate_augmentation_db(args)
