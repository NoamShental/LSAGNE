The CMAP data has many drugs with only a few samples per cell-line.
In order to utilize this data in DL, we need to employ techniques from few-shot learning, specifically data augmentation.
The idea for gene expression data-augmentation is that genes tend to change in pathways or groups. 
We learn the range of these changes from the data itself via the well-established GO pathway dataset.

This is an augmentation pipeline for CMAP data. It requires two steps:
1. Build auxilliary database
2. Generate the augmentations

########################################################3
Step 1: Build auxilliary database
Explanation : This program builds a list of 588 GO pathways that are well covered by the CMAP 977 genes.
Then for every drug+cell-line combination in our dataset it builds mean and std for each GO. It also builds GO correlation matrix.
It also does this for each gene.

Main file : generate_augmentation_db_v1.py

Need to provide: 
 1. CMAP_ANNOT_FILE (GSE92742_Broad_LINCS_inst_info_filt_DMSO-trt_6h-24h_DoseOver2.txt) - provided in this archive
 2. ALL_GO_FILE (All_GO_v2.txt) - provided in this archive
 3. CMAP_GENEXP_FILE (GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x977.h5) - very big, not provided, but is our main data file and should be available
Optional to restrict to specific drugs and cell-lines:
 4. drugs file - example provided (use_drugs.txt)
 5. cell-lines file - example provided (use_cell_lines.txt)
 
########################################################3
Step 2: Augment CMAP samples based in pre-built auxilliary database
Explanation : For every sample, it chooses up to N_PATHWAYS GO pathways each with probability PROBA_PATHWAY. For each such GO it samples from the pre-computed
mean and std for this GO. It then modifies the sample genes that correspond to this pathway so that it would equal the sampled value. It then selects N_CORRPATHWAYS GOs that 
are correlated to the selected GO and modifies them accordingly.
It also does the same for genes. You can choose to use std per drug+Cell-line or max per drug (default). 

Main file : augment_data_v1.py

Need to provide: Same as step 1

Main function:
cmap_multisample_augmentation(drug, celline, numperms_persample = 10, usevariance = "perDrugMax",
                                    N_PATHWAYS=5, N_CORRPATHWAYS=3, PROBA_PATHWAY=0.7,
                                    N_GENES=5, N_CORRGENES=5, PROBA_GENE=0.7)

The parameters are explained in the code

