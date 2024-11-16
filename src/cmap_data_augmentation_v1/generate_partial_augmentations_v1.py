from __future__ import annotations
import argparse
import multiprocessing
import pickle
import time
from dataclasses import dataclass
from functools import cached_property
from logging import Logger
from os.path import exists
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from pandas._typing import ArrayLike

from src.cmap_cloud_ref import CmapCloudRef
from src.logger_utils import create_logger
from src.perturbation import Perturbation
from src.tissue import Tissue

num_cores = multiprocessing.cpu_count()


GENE_VALS_EPSILON = 0.001
GO_VALS_EPSILON = 0.001
MAX_CORRPATHWAYS = 10
MAX_CORRGENES = 10

AugmentationVariace = Literal['perDrugMax', 'perDrugCelline', 'perDrugAvg']
Augmentation = tuple[NDArray[bool], NDArray[float]]


@dataclass(frozen=True)
class AugmentationGenerationParameters:
    data_augmentation_db_folder: Path
    drug_batch_size: int
    tissue: Tissue
    perturbation: Perturbation
    num_of_samples: int
    output: Path | None
    n_pathways: int
    n_corrpathways: int
    proba_pathway: float
    n_genes: int
    n_corrgenes: int
    proba_gene: float
    fold_change_factor: float = 1.0
    sample_genexp_names: ArrayLike | None = None
    use_variance: AugmentationVariace = 'perDrugMax'
    random_seed: int | None = None

    @cached_property
    def cloud_ref(self) -> CmapCloudRef:
        return CmapCloudRef(self.tissue, self.perturbation)



    @classmethod
    def create_using_args(cls):
        # srun python generate_partial_augmentations_v1.py TUMOR_CONST PERT_CONST NUM_OF_REPEATS_CONST PYTHON_OUTPUT_FOLDER -job-prefix JOB_PREFIX """
        parser = argparse.ArgumentParser(description="Generating ...")
        parser.add_argument(
            "--data-augmentation-db-folder",
            type=Path,
            required=True,
            help="Path to data augmentation DB folder"
        )
        parser.add_argument(
            "--tissue",
            type=Tissue,
            required=True,
            help="Tissue to augment"
        )
        parser.add_argument(
            "--perturbation",
            type=Perturbation,
            required=True,
            help="Perturbation to augment"
        )
        parser.add_argument(
            "--drug-batch-size",
            type=int,
            required=True,
            help="..."
        )
        parser.add_argument(
            "--num-of-samples",
            type=int,
            required=True,
            help="How much samples are needed"
        )
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="The output file"
        )
        parser.add_argument(
            "--n-pathways",
            type=int,
            required=True,
            help=""
        )
        parser.add_argument(
            "--n-corrpathways",
            type=int,
            required=True,
            help=""
        )
        parser.add_argument(
            "--proba-pathway",
            type=float,
            required=True,
            help=""
        )
        parser.add_argument(
            "--n-genes",
            type=int,
            required=True,
            help=""
        )
        parser.add_argument(
            "--n-corrgenes",
            type=int,
            required=True,
            help=""
        )
        parser.add_argument(
            "--proba-gene",
            type=float,
            required=True,
            help=""
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            required=False,
            default=None,
            help=""
        )
        parser.add_argument(
            "--use-variance",
            choices=get_args(AugmentationVariace),
            required=False,
            help=""
        )
        parser.add_argument(
            "--fold-change-factor",
            type=float,
            required=False,
            default=1.0,
            help=""
        )
        args = parser.parse_args()
        # TODO add rest of the fields
        return cls(
            data_augmentation_db_folder=args.data_augmentation_db_folder,
            drug_batch_size=args.drug_batch_size,
            tissue=args.tissue,
            perturbation=args.perturbation,
            num_of_samples=args.num_of_samples,
            output=args.output,
            n_pathways=args.n_pathways,
            n_corrpathways=args.n_corrpathways,
            proba_pathway=args.proba_pathway,
            n_genes=args.n_genes,
            n_corrgenes=args.n_corrgenes,
            proba_gene=args.proba_gene,
            use_variance=args.use_variance,
            fold_change_factor=args.fold_change_factor
        )


@dataclass(frozen=True)
class CloudData:
    go_mean: NDArray
    go_std: NDArray
    gene_mean: NDArray
    gene_std: NDArray
    go_std_prob: NDArray
    gene_std_prob: NDArray
    corr_pathways: NDArray
    corr_genes: NDArray
    corr_pathways_df: pd.DataFrame
    corr_genes_df: pd.DataFrame
    padded_go_genes: NDArray

    @classmethod
    def create(
            cls,
            cloud_ref: CmapCloudRef,
            db,
            GO,
            EXTRA_GENE,
            go_vec,
            gene_vec
    ) -> CloudData:
        drug = cloud_ref.perturbation
        celline = cloud_ref.tissue_code

        cloud_db = db[(drug, celline)]

        # GO_vals - take the genes of this GO and calculate the mean and std of all the cloud
        GO_vals = cloud_db["GO_vals"]
        GO_mean = GO_vals['mean'].to_numpy()
        # std = if low - then the genes are correlated
        #       if high - they are all over the place - so we can use them in the augmentation process
        GO_std = GO_vals['std'].to_numpy()

        GO_corr = cloud_db["GO_corr"]

        # Gene_vals - the multidimensional normal distribution on the genes in the cloud
        Gene_vals = cloud_db["Gene_vals"]
        Gene_mean = Gene_vals["mean"].to_numpy()
        Gene_std = Gene_vals["std"].to_numpy()
        Gene_corr = cloud_db["Gene_corr"]

        go_std_prob = GO_std / GO_std.sum()  # probabilities based on std
        # pathways prob
        # is noisy - will be chosen frequently, otherwise will be mostly suppressed.
        # pp = p / p.sum()

        gene_std_prob = Gene_std / np.sum(Gene_std)  # probabilities based on std
        # if high - we can use that as augmentation
        # pg = p / p.sum()

        corr_pathways_df = abs(GO_corr[GO_vals.index]).apply(
            lambda col: col.nlargest(MAX_CORRPATHWAYS + 1).index.tolist())
        corr_genes_df = abs(Gene_corr[Gene_corr.index]).apply(
            lambda col: col.nlargest(MAX_CORRGENES + 1).index.tolist())

        _go_genes = [gene_vec(x) for x in GO['Genes'].values.tolist()]
        _go_genes.append([])
        _go_genes_max_size = max(map(len, _go_genes))
        padded_go_genes = np.array(
            [np.pad(x, pad_width=(0, _go_genes_max_size - len(x)), mode='constant', constant_values=EXTRA_GENE) for x in
             _go_genes], dtype=int)

        return cls(
            go_mean=GO_mean,
            go_std=GO_std,
            gene_mean=Gene_mean,
            gene_std=Gene_std,
            go_std_prob=go_std_prob,
            gene_std_prob=gene_std_prob,
            corr_pathways=go_vec(corr_pathways_df),
            corr_genes=gene_vec(corr_genes_df),
            corr_pathways_df=corr_pathways_df,
            corr_genes_df=corr_genes_df,
            padded_go_genes=padded_go_genes
        )


class CmapAugmentation:
    def __init__(self, logger: Logger, parameters: AugmentationGenerationParameters):
        """
        Ctor
        :param logger: Logger
        :param parameters: All parameters for augmentation generation
        """

        go_file = str(parameters.data_augmentation_db_folder / 'GO_Over3genesLess50_from977_v22.txt')
        drugs_cells_file = str(parameters.data_augmentation_db_folder / 'drugsOver6reps_cellinesOver4_v22.txt')
        data_prefix = str(parameters.data_augmentation_db_folder / 'data_augment_cmap_db_{}_v22.pkl')

        self.logger = logger
        # read data
        self.cmap_annot_drug = pd.read_csv(drugs_cells_file, sep="\t", index_col=0)
        self._reduce_cmap_annot_drug(self.cmap_annot_drug, parameters.perturbation, parameters.tissue)
        # Pathway = GO = Gene Ontology
        self.GO = pd.read_csv(go_file, sep="\t", index_col=1)
        self.GO = self.GO[["Genes", "Count"]]
        self.GO["Genes"] = self.GO["Genes"].str.split(',')

        self.db = self._prepare_db_for_all_clouds(self.cmap_annot_drug, parameters.use_variance, parameters.drug_batch_size, data_prefix)
        sample_genexp_names = self.db["gene_names"] if parameters.sample_genexp_names is None else parameters.sample_genexp_names
        self.sample_genexp_names_lookup = pd.Series(np.arange(len(sample_genexp_names)), index=sample_genexp_names)
        # self.GOnames_lookup = np.array(self.db["GO_names"])
        # self.cmapgenes_lookup = np.array(self.db["gene_names"])
        self.len_GO = len(self.db["GO_names"])
        self.len_cmapgenes = len(self.db["gene_names"])
        self.rng = np.random.default_rng(parameters.random_seed)

        self.EXTRA_GO = len(self.GO)
        go_reset = self.GO.reset_index()
        go_mapping = dict(zip(go_reset['GO'], go_reset.index))
        self.go_vec = np.vectorize(lambda go_name: go_mapping[go_name])

        self.EXTRA_GENE = len(self.sample_genexp_names_lookup)
        gene_mapping = dict(zip(self.sample_genexp_names_lookup.index, self.sample_genexp_names_lookup))
        self.gene_vec = np.vectorize(lambda gene_name: gene_mapping[gene_name])

        _go_genes = [self.gene_vec(x) for x in self.GO['Genes'].values.tolist()]
        _go_genes.append([])
        _go_genes_max_size = max(map(len, _go_genes))
        self.padded_go_genes = np.array(
            [np.pad(x, pad_width=(0, _go_genes_max_size - len(x)), mode='constant', constant_values=self.EXTRA_GENE)
             for x in _go_genes], dtype=int)

        self.cloud_ref_to_cloud_data: dict[CmapCloudRef, CloudData] = {}

    @staticmethod
    def _reduce_cmap_annot_drug(cmap_annot_drug: pd.DataFrame, pert: Perturbation, tissue: Tissue):
        """
        This is a hack to delete unnecessary work done while loading DB
        """
        cmap_annot_drug.loc[cmap_annot_drug['pert_iname'] == pert, ['cell_id', 'cell_lines']] = [tissue.tissue_code, 1]

    @staticmethod
    def _load_db_for_drug(
            cmap_annot_drug,
            drug,
            drug_batch_size,
            data_prefix
    ):
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
        iidx = (idx + drug_batch_size) // drug_batch_size
        if not exists(data_prefix.format(iidx)):
            raise AssertionError("Can't find DB file {}. Exiting...".format(data_prefix.format(iidx)))
        with open(data_prefix.format(iidx), "rb") as f:
            return pickle.load(f), idx

    def _prepare_db_for_cloud(
            self,
            cmap_annot_drug,
            drug,
            celline,
            use_variance,
            drug_batch_size,
            data_prefix
    ):
        """
        Create DB for current cloud
        :param cmap_annot_drug: drug/cell-line annotation file
        :param drug:
        :param celline: group the samples belongs to
        :return: New DB
        """
        db, idx = self._load_db_for_drug(cmap_annot_drug, drug, drug_batch_size, data_prefix)

        GOnames = db["GO_names"]
        cmapgenes = db["gene_names"]

        cloud_ref = (drug, celline)

        if cloud_ref not in db.keys():
            raise AssertionError("No drug-cell combination in database - exiting...")

        # modify db if needed
        if (use_variance == "perDrugMax") or (use_variance == "perDrugAvg"):
            self.logger.debug("Generalizing drug effects DB...")
            res_db = {
                cloud_ref: db[cloud_ref].copy(),
                "GO_names": GOnames,
                "gene_names": cmapgenes
            }
            res_db[cloud_ref]["GO_vals"] = pd.DataFrame(res_db[cloud_ref]["GO_vals"][:,0:2],
                                                           columns=["mean", "std_{}".format(celline)], index=GOnames)
            res_db[cloud_ref]["GO_corr"] = pd.DataFrame(res_db[cloud_ref]["GO_corr"], columns=GOnames,
                                                           index=GOnames)
            res_db[cloud_ref]["Gene_vals"] = pd.DataFrame(res_db[cloud_ref]["Gene_vals"][:,0:2],
                                                             columns=["mean", "std_{}".format(celline)],
                                                             index=cmapgenes)
            res_db[cloud_ref]["Gene_corr"] = pd.DataFrame(res_db[cloud_ref]["Gene_corr"], columns=cmapgenes,
                                                             index=cmapgenes)

            allcellines = list(set(cmap_annot_drug.iloc[idx]["cell_id"].split(',')) - {celline})
            for cl in allcellines:
                GO_vals = pd.DataFrame(db[(drug, cl)]["GO_vals"][:,0:2], columns=["mean", "std"], index=GOnames)
                Gene_vals = pd.DataFrame(db[(drug, cl)]["Gene_vals"][:,0:2], columns=["mean", "std"], index=cmapgenes)
                res_db[cloud_ref]["GO_vals"]["std_{}".format(cl)] = GO_vals["std"]
                res_db[cloud_ref]["Gene_vals"]["std_{}".format(cl)] = Gene_vals["std"]
            if (use_variance == "perDrugMax"):
                res_db[cloud_ref]["GO_vals"]["std"] = res_db[cloud_ref]["GO_vals"][
                    res_db[cloud_ref]["GO_vals"].columns[1:]].max(axis=1)
            elif (use_variance == "perDrugAvg"):
                res_db[cloud_ref]["GO_vals"]["std"] = res_db[cloud_ref]["GO_vals"][
                    res_db[cloud_ref]["GO_vals"].columns[1:]].mean(axis=1)
            else:
                raise AssertionError("Unknown usevariance: {}".format(use_variance))
            res_db[cloud_ref]["GO_vals"] = res_db[cloud_ref]["GO_vals"][["mean", "std"]]

            if (use_variance == "perDrugMax"):
                res_db[cloud_ref]["Gene_vals"]["std"] = res_db[cloud_ref]["Gene_vals"][
                    res_db[cloud_ref]["Gene_vals"].columns[1:]].max(axis=1)
            elif (use_variance == "perDrugAvg"):
                res_db[cloud_ref]["Gene_vals"]["std"] = res_db[cloud_ref]["Gene_vals"][
                    res_db[cloud_ref]["Gene_vals"].columns[1:]].max(axis=1)
            else:
                raise AssertionError("Unknown usevariance: {}".format(use_variance))
            res_db[cloud_ref]["Gene_vals"] = res_db[cloud_ref]["Gene_vals"][["mean", "std"]]
        elif (use_variance == "perDrugCelline"):
            res_db = {
                cloud_ref: db[cloud_ref].copy(),
                "GO_names": GOnames,
                "gene_names": cmapgenes
            }
            res_db[cloud_ref]["GO_vals"] = pd.DataFrame(res_db[cloud_ref]["GO_vals"][:,0:2],
                                                           columns=["mean", "std"], index=GOnames)
            res_db[cloud_ref]["GO_corr"] = pd.DataFrame(res_db[cloud_ref]["GO_corr"], columns=GOnames,
                                                           index=GOnames)
            res_db[cloud_ref]["Gene_vals"] = pd.DataFrame(res_db[cloud_ref]["Gene_vals"][:,0:2],
                                                             columns=["mean", "std"],
                                                             index=cmapgenes)
            res_db[cloud_ref]["Gene_corr"] = pd.DataFrame(res_db[cloud_ref]["Gene_corr"], columns=cmapgenes,
                                                             index=cmapgenes)
        else:
            raise AssertionError("Unknown usevariance: {}".format(use_variance))
        return res_db

    def _prepare_db_for_all_clouds(self, cmap_annot_drug, use_variance, drug_batch_size, data_prefix, specificdrug=None):
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
            if drug in ['DMSO','time 24h']:
                current_variance = "perDrugCelline"
            else:
                current_variance = use_variance
            self.logger.info(f'Load DB for drug {drug}, with variance {current_variance}')
            cellines = row['cell_id'].split(',')
            for celline in cellines:
                self.logger.info(f'Preparing DB for ({drug}, {celline})...')
                db_for_current_cloud = self._prepare_db_for_cloud(cmap_annot_drug, drug, celline, current_variance, drug_batch_size, data_prefix)
                db.update(db_for_current_cloud)
        return db

    @staticmethod
    def duplicate_by_row_mask(arr):
        sorted_indices = np.argsort(arr, axis=-1)
        sorted_arr = np.take_along_axis(arr, sorted_indices, -1)
        same_indices = np.where(np.diff(sorted_arr, axis=-1) == 0)
        duplicate_indices = (same_indices[0], same_indices[1] + 1)
        duplicate_mask = np.zeros_like(arr, dtype=bool)
        rows_idx = sorted_indices[duplicate_indices]
        duplicate_mask[(same_indices[0], rows_idx)] = True
        return duplicate_mask

    def multisample_augment(
            self,
            cloud_ref: CmapCloudRef,
            number_of_samples_to_generate,
            N_PATHWAYS,
            N_CORRPATHWAYS,
            PROBA_PATHWAY,
            N_GENES,
            N_CORRGENES,
            PROBA_GENE,
            fc_factor: float
    ):
        if (cloud_ref.perturbation, cloud_ref.tissue.tissue_code) not in self.db:
            raise ValueError(f'Cloud {cloud_ref} does not exist in the DB')

        if cloud_ref in self.cloud_ref_to_cloud_data:
            cloud_data = self.cloud_ref_to_cloud_data[cloud_ref]
        else:
            cloud_data = CloudData.create(cloud_ref, self.db, self.GO, self.EXTRA_GENE, self.go_vec, self.gene_vec)

        # adding extra gene to allow vectorized numpy usage
        res = np.tile(np.append(cloud_data.gene_mean, 0), (number_of_samples_to_generate, 1))
        # pick N_PATHWAYS pathways based on their std (not same ones) with probability PROBA_PATHWAY.
        # N_PATHWAYS_proba = self.rng.binomial(N_PATHWAYS, PROBA_PATHWAY)
        # GO - pathways
        # pp higher = more noise = more prob to be chosen
        go_pidx = np.array([self.rng.choice(self.len_GO, N_PATHWAYS, replace=False, p=cloud_data.go_std_prob) for _ in range(number_of_samples_to_generate)])
        # pick N_CORRPATHWAYS correlated pathways based on correlation to original pathway
        # corr_pathways dims = N * N_PATHWAYS * (N_CORRPATHWAYS + 1)
        # note that for each sample each row represents a GO, the first column is the most correlated GO and so on
        corr_pathways = cloud_data.corr_pathways.T[go_pidx][:, :, :N_CORRPATHWAYS + 1]
        # Calculate the foldchange for each selected pathway - it will be the same for correlated pathways
        GO_vals_pidx_mean = cloud_data.go_mean[go_pidx]
        # Calculate the GO as if it is a normal dist
        targetVals = self.rng.normal(GO_vals_pidx_mean, cloud_data.go_std[go_pidx])
        FCs = targetVals / np.maximum(GO_VALS_EPSILON, GO_vals_pidx_mean)
        FCs = FCs.clip(0.01, 10)
        FCs = FCs.repeat(N_CORRPATHWAYS + 1, axis=-1)
        # now combine the pathways and their correlated pathways into one list
        combined_values = corr_pathways.reshape(number_of_samples_to_generate, -1)
        duplicated_go_mask = self.duplicate_by_row_mask(combined_values)
        combined_values[duplicated_go_mask] = self.EXTRA_GO
        go_genes = cloud_data.padded_go_genes[combined_values]
        go_genes = go_genes.reshape(number_of_samples_to_generate, -1)
        duplicated_gene_mask = self.duplicate_by_row_mask(go_genes)
        go_genes[duplicated_gene_mask] = self.EXTRA_GENE
        FCs = FCs.repeat(cloud_data.padded_go_genes.shape[1], axis=-1)
        res[np.arange(number_of_samples_to_generate)[:, np.newaxis], go_genes] *= FCs * fc_factor

        # pick N_GENES genes based on their std (not same ones) with probability PROBA_GENE.
        # gene_std_prob higher = more noise = more prob to be chosen
        gene_pidx = np.array([self.rng.choice(self.len_cmapgenes, N_GENES, replace=False, p=cloud_data.gene_std_prob) for _ in
                         range(number_of_samples_to_generate)])
        # pick N_CORRGENES correlated pathways based on correlation to original pathway
        # note that for each sample each row represents a gene, the first column is the most correlated gene and so on
        corr_genes = cloud_data.corr_genes.T[gene_pidx][:, :, :N_CORRGENES + 1]
        # Calculate the foldchange for each selected pathway - it will be the same for correlated pathways
        gene_vals_pidx_mean = cloud_data.gene_mean[gene_pidx]
        # Calculate the GO as if it is a normal dist
        sampled_vals = self.rng.normal(gene_vals_pidx_mean, cloud_data.gene_std[gene_pidx])
        FCs = sampled_vals / np.maximum(GO_VALS_EPSILON, gene_vals_pidx_mean)
        FCs = FCs.clip(0.01, 10)
        FCs = FCs.repeat(N_CORRGENES + 1, axis=-1)
        # now combine the pathways and their correlated pathways into one list
        combined_values = corr_genes.reshape(number_of_samples_to_generate, -1)
        duplicated_gene_mask = self.duplicate_by_row_mask(combined_values)
        combined_values[duplicated_gene_mask] = self.EXTRA_GENE
        # go_genes = cloud_data.padded_go_genes[combined_values]
        # go_genes = go_genes.reshape(number_of_samples_to_generate, -1)
        res[np.arange(number_of_samples_to_generate)[:, np.newaxis], combined_values] *= FCs * fc_factor

        # drop the extra gene
        res = res[:, :-1]
        changed_genes_mask = np.abs(res - cloud_data.gene_mean) > 0
        res[~changed_genes_mask] = np.nan

        self.logger.info(f'Done generating {number_of_samples_to_generate:,} samples')

        return changed_genes_mask, res


def generate_augmentations(args: AugmentationGenerationParameters, augmentor: CmapAugmentation | None = None):
    logger = create_logger('aug_gen')
    celline = Tissue(args.tissue).tissue_code
    drug = args.perturbation
    NumberOfSamplesToGenerate = args.num_of_samples
    outfile = args.output

    logger.info(f'Running {celline} - {drug}')

    logger.info(f'Loading augmentation library...')
    st = time.time()
    if not augmentor:
        augmentor = CmapAugmentation(logger, args)
    et = time.time()
    elapsed_time = et - st
    logger.info(f'Done, Elapsed {elapsed_time} seconds')

    resDict = {}
    logger.info(f'Augmenting {NumberOfSamplesToGenerate:,} samples...')
    st = time.time()
    resDict[args.cloud_ref] = augmentor.multisample_augment(
        cloud_ref=args.cloud_ref,
        number_of_samples_to_generate=NumberOfSamplesToGenerate,
        N_PATHWAYS=args.n_pathways,
        N_CORRPATHWAYS=args.n_corrpathways,
        PROBA_PATHWAY=args.proba_pathway,
        N_GENES=args.n_genes,
        N_CORRGENES=args.n_corrgenes,
        PROBA_GENE=args.proba_gene,
        fc_factor=args.fold_change_factor
    )
    et = time.time()
    elapsed_time = et - st
    logger.info(f'Done, Elapsed {elapsed_time} seconds. Saving...')
    if outfile:
        pickle.dump(resDict, open(outfile, 'wb'))
    logger.info("Done.")
    return resDict


if __name__ == '__main__':
    args = AugmentationGenerationParameters.create_using_args()
    generate_augmentations(args)
