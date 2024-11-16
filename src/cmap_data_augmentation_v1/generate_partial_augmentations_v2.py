import logging
import time
import os.path
from logging import Logger
import sys

import pandas as pd
# import swifter
import numpy as np
import pickle
from os.path import exists

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../..')
from src.configuration import config
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

# input
GO_FILE = os.path.join(config.organized_data_augmentation_folder, 'GO_Over3genesLess50_from977_v22.txt')
DRUGS_CELLS_FILE = os.path.join(config.organized_data_augmentation_folder, 'drugsOver6reps_cellinesOver4_v22.txt')
DATA_PREFIX = os.path.join(config.organized_data_augmentation_folder, 'data_augment_cmap_db_{}_v22.pkl')
DRUG_BATCH_SIZE = 15
MAX_CORRPATHWAYS = 10
MAX_CORRGENES = 10
GENE_VALS_EPSILON = 0.001
GO_VALS_EPSILON = 0.001


class CmapAugmentation:
    def __init__(self, logger: Logger, sample_genexp_names=None, usevariance="perDrugMax", specificdrug=None):
        """
        Ctor
        :param logger: Logger
        :param usevariance: can be "perDrugMax", "perDrugCelline", "perDrugAvg"
        """
        self.logger = logger
        # read data
        self.cmap_annot_drug = pd.read_csv(DRUGS_CELLS_FILE, sep="\t", index_col=0)
        self.GO = pd.read_csv(GO_FILE, sep="\t", index_col=1)
        self.GO = self.GO[["Genes", "Count"]]
        self.GO["Genes"] = self.GO["Genes"].str.split(',')
        self.usevariance = usevariance
        self.db = self._prepare_db_for_all_clouds(self.cmap_annot_drug, specificdrug)
        if sample_genexp_names is None:
            sample_genexp_names = self.db["gene_names"]
        self.sample_genexp_names_lookup = pd.Series(np.arange(len(sample_genexp_names)), index=sample_genexp_names)
        self.GOnames_lookup = np.array(self.db["GO_names"])
        self.cmapgenes_lookup = np.array(self.db["gene_names"])
        self.len_GOnames = len(self.db["GO_names"])
        self.len_cmapgenes = len(self.db["gene_names"])

    def _load_db_for_drug(self, cmap_annot_drug, drug):
        """
        Loads data augmentation CMAP DB from the disk for the given drug.
        :param cmap_annot_drug: drug/cell-line annotation file
        :param drug: drug
        :return: Data augmentation CMAP DB
        """
        idx = list(np.where(cmap_annot_drug["pert_iname"] == drug)[0])
        if len(idx) != 1:
            raise AssertionError("More than one hit or no hits in annotation file - exiting ...")
        idx = idx[0]
        iidx = (idx + DRUG_BATCH_SIZE) // DRUG_BATCH_SIZE
        if not exists(DATA_PREFIX.format(iidx)):
            raise AssertionError("Can't find DB file {}. Exiting...".format(DATA_PREFIX.format(iidx)))
        with open(DATA_PREFIX.format(iidx), "rb") as f:
            return pickle.load(f), idx

    def _prepare_db_for_cloud(
            self,
            cmap_annot_drug,
            drug,
            celline,
            usevariance=None
    ):
        """
        Create DB for current cloud
        :param cmap_annot_drug: drug/cell-line annotation file
        :param drug:
        :param celline: group the samples belongs to
        :return: New DB
        """
        if usevariance is None:
            usevariance = self.usevariance
        db, idx = self._load_db_for_drug(cmap_annot_drug, drug)

        GOnames = db["GO_names"]
        cmapgenes = db["gene_names"]

        cloud_ref = (drug, celline)

        if cloud_ref not in db.keys():
            raise AssertionError("No drug-cell combination in database - exiting...")

        # modify db if needed
        if ((usevariance == "perDrugMax") | (usevariance == "perDrugAvg")):
            self.logger.debug("Generalizing drug effects DB...")
            res_db = {
                cloud_ref: db[cloud_ref].copy(),
                "GO_names": GOnames,
                "gene_names": cmapgenes
            }
            res_db[cloud_ref]["GO_vals"] = pd.DataFrame(res_db[cloud_ref]["GO_vals"][:, 0:2],
                                                        columns=["mean", "std_{}".format(celline)], index=GOnames)
            res_db[cloud_ref]["GO_corr"] = pd.DataFrame(res_db[cloud_ref]["GO_corr"], columns=GOnames,
                                                        index=GOnames)
            res_db[cloud_ref]["Gene_vals"] = pd.DataFrame(res_db[cloud_ref]["Gene_vals"][:, 0:2],
                                                          columns=["mean", "std_{}".format(celline)],
                                                          index=cmapgenes)
            res_db[cloud_ref]["Gene_corr"] = pd.DataFrame(res_db[cloud_ref]["Gene_corr"], columns=cmapgenes,
                                                          index=cmapgenes)

            allcellines = list(set(cmap_annot_drug.iloc[idx]["cell_id"].split(',')) - set([celline]))
            for cl in allcellines:
                GO_vals = pd.DataFrame(db[(drug, cl)]["GO_vals"][:, 0:2], columns=["mean", "std"], index=GOnames)
                Gene_vals = pd.DataFrame(db[(drug, cl)]["Gene_vals"][:, 0:2], columns=["mean", "std"], index=cmapgenes)
                res_db[cloud_ref]["GO_vals"]["std_{}".format(cl)] = GO_vals["std"]
                res_db[cloud_ref]["Gene_vals"]["std_{}".format(cl)] = Gene_vals["std"]
            if (usevariance == "perDrugMax"):
                res_db[cloud_ref]["GO_vals"]["std"] = res_db[cloud_ref]["GO_vals"][
                    res_db[cloud_ref]["GO_vals"].columns[1:]].max(axis=1)
            elif (usevariance == "perDrugAvg"):
                res_db[cloud_ref]["GO_vals"]["std"] = res_db[cloud_ref]["GO_vals"][
                    res_db[cloud_ref]["GO_vals"].columns[1:]].mean(axis=1)
            else:
                raise AssertionError("Unknown usevariance: {}".format(self.usevariance))
            res_db[cloud_ref]["GO_vals"] = res_db[cloud_ref]["GO_vals"][["mean", "std"]]

            if (usevariance == "perDrugMax"):
                res_db[cloud_ref]["Gene_vals"]["std"] = res_db[cloud_ref]["Gene_vals"][
                    res_db[cloud_ref]["Gene_vals"].columns[1:]].max(axis=1)
            elif (usevariance == "perDrugAvg"):
                res_db[cloud_ref]["Gene_vals"]["std"] = res_db[cloud_ref]["Gene_vals"][
                    res_db[cloud_ref]["Gene_vals"].columns[1:]].max(axis=1)
            else:
                raise AssertionError("Unknown usevariance: {}".format(self.usevariance))
            res_db[cloud_ref]["Gene_vals"] = res_db[cloud_ref]["Gene_vals"][["mean", "std"]]
        elif (usevariance == "perDrugCelline"):
            res_db = {
                cloud_ref: db[cloud_ref].copy(),
                "GO_names": GOnames,
                "gene_names": cmapgenes
            }
            res_db[cloud_ref]["GO_vals"] = pd.DataFrame(res_db[cloud_ref]["GO_vals"][:, 0:2],
                                                        columns=["mean", "std"], index=GOnames)
            res_db[cloud_ref]["GO_corr"] = pd.DataFrame(res_db[cloud_ref]["GO_corr"], columns=GOnames,
                                                        index=GOnames)
            res_db[cloud_ref]["Gene_vals"] = pd.DataFrame(res_db[cloud_ref]["Gene_vals"][:, 0:2],
                                                          columns=["mean", "std"],
                                                          index=cmapgenes)
            res_db[cloud_ref]["Gene_corr"] = pd.DataFrame(res_db[cloud_ref]["Gene_corr"], columns=cmapgenes,
                                                          index=cmapgenes)
        else:
            raise AssertionError("Unknown usevariance: {}".format(self.usevariance))
        return res_db

    def _prepare_db_for_all_clouds(self, cmap_annot_drug, specificdrug=None):
        """
        Creates DB for all clouds
        :param cmap_annot_drug: drug/cell-line annotation file
        :return: DB for all clouds
        """
        db = {}
        for i, row in cmap_annot_drug.iterrows():
            drug = row['pert_iname']
            if specificdrug != None:
                if drug != specificdrug:
                    continue
            if drug in ['DMSO', 'time 24h']:
                usevariance = "perDrugCelline"
            else:
                usevariance = self.usevariance
            print(f'Drug {drug} - {usevariance}')
            cellines = row['cell_id'].split(',')
            for celline in cellines:
                self.logger.info(f'Augmentor preparing db for ({drug}, {celline})...')
                db_for_current_cloud = self._prepare_db_for_cloud(cmap_annot_drug, drug, celline, usevariance)
                db.update(db_for_current_cloud)
        return db

    def augment_sample(
            self,
            N_PATHWAYS=5,
            N_CORRPATHWAYS=3,
            PROBA_PATHWAY=1,
            N_GENES=5,
            N_CORRGENES=5,
            PROBA_GENE=1
    ):
        """
        Create augmentation for single sample
        :param genexp: Sample from raw CMAP (before standardization) to augment
        :param drug: the drug
        :param celline: group the samples belongs to
        :param N_PATHWAYS: Number of pathways to attempts to modify each with probability PROBA_PATHWAY
        :param N_CORRPATHWAYS: Number of correlated pathways to modify
        :param PROBA_PATHWAY: Probability to modify pathway (NVIDIA showed this approach to be better for few-shot)
        :param N_GENES: Number of genes to attempt to modify each with probability PROBA_GENE
        :param N_CORRGENES: Number of correlated genes to modify
        :param PROBA_GENE:
        :return:
        """
        # st = time.time()
        res = self.AVG_Genexp
        # pick N_PATHWAYS pathways based on their std (not same ones) with probability PROBA_PATHWAY.
        N_PATHWAYS_proba = sum(np.random.choice([0, 1], N_PATHWAYS, replace=True, p=[1 - PROBA_PATHWAY, PROBA_PATHWAY]))
        pidx = np.random.choice(self.len_GOnames, N_PATHWAYS_proba, replace=False, p=self.pp)
        # pick N_CORRPATHWAYS correlated pathways based on correlation to original pathway
        corrPathways = self.corrPathways[self.db["GO_names"].iloc[pidx]].head(N_CORRPATHWAYS + 1)
        # Calculate the foldchange for each selected pathway - it will be the same for correlated pathways
        targetVals = np.random.normal(self.GO_vals.iloc[pidx]["mean"], self.GO_vals.iloc[pidx]["std"], len(pidx))
        # FCs = targetVals / self.GO_vals.iloc[pidx]["mean"]
        FCs = targetVals / np.maximum(GO_VALS_EPSILON, self.GO_vals.iloc[pidx]["mean"])
        FCs[FCs < 0.01] = 0.01;
        FCs[FCs > 10] = 10
        FCs = FCs.repeat(N_CORRPATHWAYS + 1)
        # now combine the pathways and their correlated pathways into one list
        combined_values = corrPathways.values.T.flatten().tolist()
        FCs.index = combined_values
        FCs = FCs[~FCs.index.duplicated(keep='first')]
        mask = np.isin(self.GOnames_lookup, FCs.index)
        pidx = np.where(mask)[0]
        # now extract genes for each pathway
        FCs = FCs.repeat(self.GO.loc[self.GO_vals.iloc[pidx].index]["Count"])
        FCs.index = self.GO.loc[self.GO_vals.iloc[pidx].index]["Genes"].explode()
        FCs = FCs[~FCs.index.duplicated(keep='first')]
        res.loc[FCs.index] *= FCs
        pidx0 = pidx
        # et = time.time()
        # elapsed_time = et - st
        # print(f'Done augmenting pathways, Elapsed {elapsed_time} seconds. ')
        # st = time.time()
        # pick N_GENES genes based on their std (not same ones) with probability PROBA_GENE.
        N_GENES_proba = sum(np.random.choice([0, 1], N_GENES, replace=True, p=[1 - PROBA_GENE, PROBA_GENE]))
        pidx = np.random.choice(self.len_cmapgenes, N_GENES_proba, replace=False, p=self.pg)
        # pick N_CORRGENES correlated genes based on correlation to original genes
        corrGenes = self.corrGenes[self.cmapgenes_lookup[pidx]].head(N_CORRGENES + 1)
        # Calculate the foldchange for each selected gene - it will be the same for correlated genes
        targetVals = np.random.normal(self.Gene_vals.iloc[pidx]["mean"], self.Gene_vals.iloc[pidx]["std"], len(pidx))
        # FCs = targetVals / self.Gene_vals.iloc[pidx]["mean"]
        FCs = targetVals / np.maximum(GENE_VALS_EPSILON, self.Gene_vals.iloc[pidx]["mean"])
        FCs[FCs < 0.01] = 0.01;
        FCs[FCs > 10] = 10
        FCs = FCs.repeat(N_CORRGENES + 1)
        # now combine the genes and their correlated genes into one list
        combined_values = corrGenes.values.T.flatten().tolist()
        FCs.index = combined_values
        FCs = FCs[~FCs.index.duplicated(keep='first')]
        # mask = np.isin(self.cmapgenes_lookup, FCs.index)
        # pidx = np.where(mask)[0]
        res.loc[FCs.index] *= FCs
        # et = time.time()
        # elapsed_time = et - st
        # print(f'Done augmenting genes, Elapsed {elapsed_time} seconds. ')

        pidx2 = np.concatenate((pidx0, pidx))
        pidx2 = np.unique(pidx2)
        ret = [pidx2, res.iloc[pidx2]]

        # return ret
        return 1
        # return res

    def multisample_augment(
            self,
            drug,
            celline,
            NumberOfSamplesToGenerate,  # How many samples from this cloud to make augmentations for
            resDict
    ):
        if (drug, celline) not in self.db:
            return resDict

        GOnames = self.db["GO_names"]
        cmapgenes = self.db["gene_names"]
        self.GO_vals = pd.DataFrame(self.db[(drug, celline)]["GO_vals"], columns=["mean", "std"], index=GOnames)
        self.GO_corr = pd.DataFrame(self.db[(drug, celline)]["GO_corr"], columns=GOnames, index=GOnames)

        self.Gene_vals = pd.DataFrame(self.db[(drug, celline)]["Gene_vals"], columns=["mean", "std"], index=cmapgenes)
        self.Gene_corr = pd.DataFrame(self.db[(drug, celline)]["Gene_corr"], columns=cmapgenes, index=cmapgenes)

        self.AVG_Genexp = self.Gene_vals['mean']

        p = self.GO_vals["std"] / sum(self.GO_vals["std"])  # probabilities based on std
        p = np.array(p)
        self.pp = p / p.sum()

        p = self.Gene_vals["std"] / np.sum(self.Gene_vals["std"])  # probabilities based on std
        p = np.array(p)
        self.pg = p / p.sum()

        if (celline, drug) not in resDict:
            resDict[(celline, drug)] = []

        pidx = list(range(self.len_GOnames))
        self.corrPathways = abs(self.GO_corr[self.GO_vals.iloc[pidx].index]).apply(
            lambda col: col.nlargest(MAX_CORRPATHWAYS + 1).index.tolist())
        pidx = list(range(self.len_cmapgenes))
        self.corrGenes = abs(self.Gene_corr[self.cmapgenes_lookup[pidx]]).apply(
            lambda col: col.nlargest(MAX_CORRGENES + 1).index.tolist())

        # parallel using swifter/dask - slower
        # # Create a DataFrame with NumberOfSamplesToGenerate rows
        # df = pd.DataFrame(index=range(NumberOfSamplesToGenerate))
        # # Apply the augment_sample function to each row in the DataFrame using swifter
        # augmented_samples = df.swifter.apply(lambda x: self.augment_sample(), axis=1)
        # # Append the results to the resDict
        # resDict[(celline, drug)].extend(augmented_samples.tolist())

        # parallel joblib - fastest so far
        def augment():
            return self.augment_sample()

        resDict[(celline, drug)].extend(
            Parallel(n_jobs=-1)(delayed(augment)() for _ in range(NumberOfSamplesToGenerate)))

        # straightforward - second place so far
        # for i in range(NumberOfSamplesToGenerate):
        # if ((i%100)==0):
        # print(i)
        # resDict[(celline, drug)].append(self.augment_sample2())

        # Parallel(n_jobs=num_cores)(delayed(self.augment_sample)() for i in range(NumberOfSamplesToGenerate))

        return resDict


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python generate_partial_augmentations_v1.py CELL_LINE DRUG NUM_OF_AUGMENTATIONS OUTFILE')
        print(
            'Example:\n python generate_partial_augmentations_v1.py A375 geldanamycin 1000 C:/Augment_A375_geldanamycin_1000.pkl')
        exit()

    celline = sys.argv[1]
    drug = sys.argv[2]
    NumberOfSamplesToGenerate = int(sys.argv[3])
    outfile = sys.argv[4]

    print(f'Running {celline} - {drug} on {num_cores}')
    # drug = "geldanamycin"
    # celline = "A375"

    print(f'Loading augmentation library...')
    st = time.time()
    augmentor = CmapAugmentation(logging, sample_genexp_names=None, usevariance="perDrugMax", specificdrug=drug)
    et = time.time()
    elapsed_time = et - st
    print(f'Done, Elapsed {elapsed_time} seconds')

    # a_sample = augmentor.augment_sample(
    #     drug=drug,
    #     celline=celline
    # )
    #
    resDict = {}
    print(f'Augmenting {NumberOfSamplesToGenerate} samples...')
    st = time.time()
    resDict = augmentor.multisample_augment(drug, celline,
                                            NumberOfSamplesToGenerate=NumberOfSamplesToGenerate,
                                            resDict=resDict)
    et = time.time()
    elapsed_time = et - st
    print(f'Done, Elapsed {elapsed_time} seconds. Saving...')
    pickle.dump(resDict, open(outfile, 'wb'))
    print("Done.")

