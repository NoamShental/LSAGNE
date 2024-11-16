import logging
import os.path
import pickle
import sys
from logging import Logger
from os.path import exists

import numpy as np
import pandas as pd

from src.configuration import config

#input
GO_FILE = os.path.join(config.organized_data_augmentation_folder, 'GO_Over3genesLess50_from977_v2.txt')
DRUGS_CELLS_FILE = os.path.join(config.organized_data_augmentation_folder, 'drugsOver6reps_cellinesOver4_v2.txt')
DATA_PREFIX = os.path.join(config.organized_data_augmentation_folder, 'data_augment_cmap_db_{}_v2.pkl')
CMAP_ANNOT_FILE = os.path.join(config.raw_data_augmentation_folder, 'GSE92742_Broad_LINCS_inst_info_filt_DMSO-trt_6h-24h_DoseOver2.txt')
# CMAP_GENEXP_FILE = os.path.join(config.raw_cmap_folder, 'GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x977.h5')
CMAP_GENEXP_FILE = os.path.join(config.organized_cmap_folder, 'data.h5')
DRUG_BATCH_SIZE = 10
USE_COMPRESSION = False  # Works but slows down considerably


class CmapAugmentationMock:
    def __init__(self, logger: Logger, sample_genexp_names, usevariance="perDrugMax"):
        """
        Ctor
        :param logger: Logger
        :param usevariance: can be "perDrugMax", "perDrugCelline", "perDrugAvg"
        """
        self.logger = logger
        self.sample_genexp_names = sample_genexp_names

    def augment_sample(
            self,
            genexp,
            drug,
            celline_code,
            N_PATHWAYS=5,
            N_CORRPATHWAYS=3,
            PROBA_PATHWAY=0.7,
            N_GENES=5,
            N_CORRGENES=5,
            PROBA_GENE=0.7
    ):
        """
        Create augmentation for single sample
        :param genexp: Sample from raw CMAP (before standardization) to augment
        :param drug: the drug
        :param celline_code: group the samples belongs to
        :param N_PATHWAYS: Number of pathways to attempts to modify each with probability PROBA_PATHWAY
        :param N_CORRPATHWAYS: Number of correlated pathways to modify
        :param PROBA_PATHWAY: Probability to modify pathway (NVIDIA showed this approach to be better for few-shot)
        :param N_GENES: Number of genes to attempt to modify each with probability PROBA_GENE
        :param N_CORRGENES: Number of correlated genes to modify
        :param PROBA_GENE:
        :return:
        """
        genexp_df = pd.DataFrame(genexp, columns=self.sample_genexp_names)
        noise = np.random.normal(0, 0, genexp.shape)
        return genexp + noise


class CmapAugmentation:
    def __init__(self, logger: Logger, sample_genexp_names, usevariance="perDrugMax"):
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
        self.usevariance = usevariance
        self.db = self._prepare_db_for_all_clouds(self.cmap_annot_drug)
        self.sample_genexp_names_lookup = pd.Series(np.arange(len(sample_genexp_names)), index=sample_genexp_names)

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
            celline_code,
    ):
        """
        Create DB for current cloud
        :param cmap_annot_drug: drug/cell-line annotation file
        :param drug:
        :param celline_code: group the samples belongs to
        :return: New DB
        """

        db, idx = self._load_db_for_drug(cmap_annot_drug, drug)

        GOnames = db["GO_names"]
        cmapgenes = db["gene_names"]

        cloud_ref = (drug, celline_code)

        if cloud_ref not in db.keys():
            raise AssertionError("No drug-cell combination in database - exiting...")

        # modify db if needed
        if ((self.usevariance == "perDrugMax") | (self.usevariance == "perDrugAvg")):
            self.logger.debug("Generalizing drug effects DB...")
            res_db = {
                cloud_ref: db[cloud_ref].copy(),
                "GO_names": GOnames,
                "gene_names": cmapgenes
            }
            res_db[cloud_ref]["GO_vals"] = pd.DataFrame(res_db[cloud_ref]["GO_vals"],
                                                           columns=["mean", "std_{}".format(celline_code)], index=GOnames)
            res_db[cloud_ref]["GO_corr"] = pd.DataFrame(res_db[cloud_ref]["GO_corr"], columns=GOnames,
                                                           index=GOnames)
            res_db[cloud_ref]["Gene_vals"] = pd.DataFrame(res_db[cloud_ref]["Gene_vals"],
                                                             columns=["mean", "std_{}".format(celline_code)],
                                                             index=cmapgenes)
            res_db[cloud_ref]["Gene_corr"] = pd.DataFrame(res_db[cloud_ref]["Gene_corr"], columns=cmapgenes,
                                                             index=cmapgenes)

            allcellines = list(set(cmap_annot_drug.iloc[idx]["cell_id"].split(',')) - set([celline_code]))
            for cl in allcellines:
                GO_vals = pd.DataFrame(db[(drug, cl)]["GO_vals"], columns=["mean", "std"], index=GOnames)
                Gene_vals = pd.DataFrame(db[(drug, cl)]["Gene_vals"], columns=["mean", "std"], index=cmapgenes)
                res_db[cloud_ref]["GO_vals"]["std_{}".format(cl)] = GO_vals["std"]
                res_db[cloud_ref]["Gene_vals"]["std_{}".format(cl)] = Gene_vals["std"]
            if (self.usevariance == "perDrugMax"):
                res_db[cloud_ref]["GO_vals"]["std"] = res_db[cloud_ref]["GO_vals"][
                    res_db[cloud_ref]["GO_vals"].columns[1:]].max(axis=1)
            elif (self.usevariance == "perDrugAvg"):
                res_db[cloud_ref]["GO_vals"]["std"] = res_db[cloud_ref]["GO_vals"][
                    res_db[cloud_ref]["GO_vals"].columns[1:]].mean(axis=1)
            else:
                raise AssertionError("Unknown usevariance: {}".format(self.usevariance))
            res_db[cloud_ref]["GO_vals"] = res_db[cloud_ref]["GO_vals"][["mean", "std"]]

            if (self.usevariance == "perDrugMax"):
                res_db[cloud_ref]["Gene_vals"]["std"] = res_db[cloud_ref]["Gene_vals"][
                    res_db[cloud_ref]["Gene_vals"].columns[1:]].max(axis=1)
            elif (self.usevariance == "perDrugAvg"):
                res_db[cloud_ref]["Gene_vals"]["std"] = res_db[cloud_ref]["Gene_vals"][
                    res_db[cloud_ref]["Gene_vals"].columns[1:]].max(axis=1)
            else:
                raise AssertionError("Unknown usevariance: {}".format(self.usevariance))
            res_db[cloud_ref]["Gene_vals"] = res_db[cloud_ref]["Gene_vals"][["mean", "std"]]
        elif (self.usevariance == "perDrugCelline"):
            res_db = db
        else:
            raise AssertionError("Unknown usevariance: {}".format(self.usevariance))
        return res_db

    def _prepare_db_for_all_clouds(self, cmap_annot_drug):
        """
        Creates DB for all clouds
        :param cmap_annot_drug: drug/cell-line annotation file
        :return: DB for all clouds
        """
        db = {}
        for i, row in cmap_annot_drug.iterrows():
            drug = row['pert_iname']
            celline_codes = row['cell_id'].split(',')
            for celline_code in celline_codes:
                self.logger.debug(f'Augmentor preparing db for ({drug}, {celline_code})...')
                db_for_current_cloud = self._prepare_db_for_cloud(cmap_annot_drug, drug, celline_code)
                db.update(db_for_current_cloud)
        return db

    def augment_samples(
            self,
            genexps,
            drug,
            celline_code,
            N_PATHWAYS=5,
            N_CORRPATHWAYS=3,
            PROBA_PATHWAY=0.7,
            N_GENES=5,
            N_CORRGENES=5,
            PROBA_GENE=0.7
    ):
        """
        Create augmentation for single sample
        :param genexp: Sample from raw CMAP (before standardization) to augment
        :param drug: the drug
        :param celline_code: group the samples belongs to
        :param N_PATHWAYS: Number of pathways to attempts to modify each with probability PROBA_PATHWAY
        :param N_CORRPATHWAYS: Number of correlated pathways to modify
        :param PROBA_PATHWAY: Probability to modify pathway (NVIDIA showed this approach to be better for few-shot)
        :param N_GENES: Number of genes to attempt to modify each with probability PROBA_GENE
        :param N_CORRGENES: Number of correlated genes to modify
        :param PROBA_GENE:
        :return:
        """
        augmented_samples_list = []
        for genexp in genexps:
            augmented_sample = self.augment_sample(
                genexp=genexp,
                drug=drug,
                celline_code=celline_code,
                N_PATHWAYS=N_PATHWAYS,
                N_CORRPATHWAYS=N_CORRPATHWAYS,
                PROBA_PATHWAY=PROBA_PATHWAY,
                N_GENES=N_GENES,
                N_CORRGENES=N_CORRGENES,
                PROBA_GENE=PROBA_GENE
            )
            augmented_samples_list.append(augmented_sample)
        return np.vstack(augmented_samples_list)

    def augment_sample(
            self,
            genexp,
            drug,
            celline_code,
            N_PATHWAYS=5,
            N_CORRPATHWAYS=3,
            PROBA_PATHWAY=0.7,
            N_GENES=5,
            N_CORRGENES=5,
            PROBA_GENE=0.7
    ):
        """
        Create augmentation for single sample
        :param genexp: Sample from raw CMAP (before standardization) to augment
        :param drug: the drug
        :param celline_code: group the samples belongs to
        :param N_PATHWAYS: Number of pathways to attempts to modify each with probability PROBA_PATHWAY
        :param N_CORRPATHWAYS: Number of correlated pathways to modify
        :param PROBA_PATHWAY: Probability to modify pathway (NVIDIA showed this approach to be better for few-shot)
        :param N_GENES: Number of genes to attempt to modify each with probability PROBA_GENE
        :param N_CORRGENES: Number of correlated genes to modify
        :param PROBA_GENE:
        :return:
        """
        GOnames = self.db["GO_names"]
        cmapgenes = self.db["gene_names"]
        GO_vals = pd.DataFrame(self.db[(drug, celline_code)]["GO_vals"], columns=["mean", "std"], index=GOnames)
        GO_corr = pd.DataFrame(self.db[(drug, celline_code)]["GO_corr"], columns=GOnames, index=GOnames)

        Gene_vals = pd.DataFrame(self.db[(drug, celline_code)]["Gene_vals"], columns=["mean", "std"], index=cmapgenes)
        Gene_corr = pd.DataFrame(self.db[(drug, celline_code)]["Gene_corr"], columns=cmapgenes, index=cmapgenes)

        res = genexp.copy()
        # pick N_PATHWAYS pathways based on their std (not same ones) with probability PROBA_PATHWAY.
        N_PATHWAYS_proba = sum(np.random.choice([0, 1], N_PATHWAYS, replace=True, p=[1 - PROBA_PATHWAY, PROBA_PATHWAY]))
        p = GO_vals["std"] / sum(GO_vals["std"])  # probabilities based on std
        p = np.array(p)
        p /= p.sum()
        pidx = np.random.choice(len(GOnames), N_PATHWAYS_proba, replace=False, p=p)
        # For each pathway
        for i in pidx:
            #   sample value from the pathway distribution
            targetVal = np.random.normal(GO_vals.iloc[i]["mean"], GO_vals.iloc[i]["std"], 1)[0]
            sampleVal = np.mean(genexp[self.sample_genexp_names_lookup[self.GO.iloc[i]["Genes"].split(',')]])
            if (sampleVal == 0):
                continue
            FC = targetVal / sampleVal
            #   multiply the sample pathway genes by the FoldChange from the sample mean, so that the new mean will be the sampled
            res[self.sample_genexp_names_lookup[self.GO.loc[GO_vals.iloc[i].name]["Genes"].split(',')]] *= FC
            #   select N_CORRPATHWAYS correlated pathways based on correlation to original pathway
            corrPathways = abs(GO_corr[[GO_vals.iloc[i].name]]).nlargest(N_CORRPATHWAYS + 1,
                                                                         columns=[GO_vals.iloc[i].name]).index[1:]
            # take their genes but remove duplicates or shared with the original pathway
            corrPathwaysGenes = list(set(','.join(self.GO.loc[corrPathways]["Genes"]).split(',')))
            corrPathwaysGenes = list(set(corrPathwaysGenes) - set(self.GO.loc[GO_vals.iloc[i].name]["Genes"].split(',')))
            #   adjust each non-intersecting gene in those pathways by same amount
            res[self.sample_genexp_names_lookup[corrPathwaysGenes]] *= FC

        # pick N_GENES genes based on their std (not same ones) with probability PROBA_GENE.
        N_GENES_proba = sum(np.random.choice([0, 1], N_GENES, replace=True, p=[1 - PROBA_GENE, PROBA_GENE]))
        p = Gene_vals["std"] / np.sum(Gene_vals["std"])  # probabilities based on std
        p = np.array(p)
        p /= p.sum()
        pidx = np.random.choice(len(cmapgenes), N_GENES_proba, replace=False, p=p)
        # For each gene
        for i in pidx:
            #   sample value from the pathway distribution
            targetVal = np.random.normal(Gene_vals.iloc[i]["mean"], Gene_vals.iloc[i]["std"], 1)[0]
            sampleVal = genexp[self.sample_genexp_names_lookup[cmapgenes[i]]]
            if (sampleVal == 0):
                continue
            FC = targetVal / sampleVal
            #   multiply the sample gene by the FoldChange, so that the new mean will be the sampled
            res[self.sample_genexp_names_lookup[cmapgenes[i]]] *= FC
            #   select N_CORRGENES correlated genes based on correlation to original gene
            corrGenes = abs(Gene_corr[cmapgenes[i]]).nlargest(N_CORRGENES + 1).index[1:]
            # take their genes but remove duplicates or shared with the original pathway
            corrGenes = list(set(corrGenes) - set(cmapgenes[i]))
            #   adjust each non-intersecting gene in those pathways by same amount
            res[self.sample_genexp_names_lookup[corrGenes]] *= FC

        return res


if __name__ == '__main__':
    # # debug1 - provide samples
    # CMAP_GENEXP_TOY_FILE = 'C:/Work/Noam/CMAP/cmap_data_augmentation/GSE92742_Broad_LINCS_Level3_INF_mlr12k_toy_n100x977.h5'
    # data=pd.read_hdf(CMAP_GENEXP_TOY_FILE)
    # data=data.transpose()
    # igenexp = data.iloc[0]
    # drug = "geldanamycin"
    # celline_code = "A375"
    # aug_data = cmap_multisample_augmentation2(cmap_annot_drug, drug, celline_code, data)
    # debug2 - provide samples

    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    drug = "geldanamycin"
    celline = "A375 skin malignant melanoma"
    # celline_code = "A375"
    celline_code = config.tissue_name_to_code[celline]




    print("Read CMAP annotation data")
    cmap_annot = pd.read_csv(CMAP_ANNOT_FILE, sep="\t", index_col=0)
    cmap_annot = cmap_annot[cmap_annot["pert_iname"].isin(list([drug]))]
    cmap_annot = cmap_annot[cmap_annot["cell_id"].isin(list([celline_code]))]
    print("Found {} samples".format(cmap_annot.shape[0]))
    print("Read gene expression data")
    data=pd.read_hdf(CMAP_GENEXP_FILE)
    # data = data.transpose()
    data = data.loc[data.index.isin(cmap_annot.index)]

    samples = data.iloc[0:10].to_numpy()

    augmentor = CmapAugmentation(logger, data.columns)

    augmented_samples = augmentor.augment_samples(
        genexps=samples,
        drug=drug,
        celline_code=celline_code
    )

    # res = cmap_multisample_augmentation(drug, celline_code, numperms_persample = 10, usevariance = "perDrugMax",
    #                                     N_PATHWAYS=5, N_CORRPATHWAYS=3, PROBA_PATHWAY=0.7,
    #                                     N_GENES=5, N_CORRGENES=5, PROBA_GENE=0.7)
    summary = pd.DataFrame(np.abs(augmented_samples - samples).flatten()).describe()
    print(f'=' * 50)
    print(f'Augmentation summary:\n{summary}\n\n')

    print("Done.")