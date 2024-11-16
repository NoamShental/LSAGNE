#!/bin/bash
### Resource allocation
#SBATCH --partition=work
#SBATCH --output="/RG/compbio/michaelel/lsagne-refactor/lsagne/run_sbatch_files/results/base00_raloPDC.out"
#SBATCH --error="/RG/compbio/michaelel/lsagne-refactor/lsagne/run_sbatch_files/results/base00_raloPDC.err"
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=16            # Number of CPU cores per task
#SBATCH --mem=64GB
#SBATCH --job-name="base00_raloPDC"

### Modules
#module load anaconda3

### Runtime
conda activate lsagnev7
export PREFECT__FLOWS__CHECKPOINTING=true
srun python /RG/compbio/michaelel/lsagne-refactor/lsagne/main.py train -tissue-code "A549" -perturbation "raloxifene" -repeat-index-start "1" -num-of-repeats "3" -output-folder "/RG/compbio/michaelel/lsagne-refactor/lsagne/run_sbatch_files/results" -root-organized-folder "/RG/compbio/michaelel/lsagne-refactor/lsagne/organized_data" -organized-folder-name "cmap" -override-run-parameters "{'n_epochs': 5000,'lr':0.0001, 'warmup_reference_points_duration': 400, 'reference_point_reselect_period': 4300, 'embedding_dim': 20,'max_radius':0.1, 'loss_coef': {'clouds_classifier': 5000, 'tissues_classifier': 5000, 'distance_from_cloud_center_6h': 10000, 'distance_from_cloud_center_dmso_24h': 10000, 'distance_from_cloud_center_24h_without_dmso_24h': 10000,'max_radius_limiter':10000, 'vae_kld': 0.01,'vae_mse': 1000, 'treatment_vectors_collinearity_using_batch_treated': 10000.0,'treatment_vectors_collinearity_using_batch_control': 10000.0, 'drug_vectors_collinearity_using_batch_treated': 10000,'drug_vectors_collinearity_using_batch_control': 10000, 'treatment_vectors_different_directions_using_batch': 10000,'drug_and_treatment_vectors_different_directions_using_batch': 10000, 'treatment_and_drug_vectors_distance_p1_p2_loss': 10000, 'treatment_and_drug_vectors_distance_p1_p2_to_treated_loss':10000,'treatment_vectors_magnitude_regulator': 0.0, 'drug_vectors_magnitude_regulator': 0.0}, 'warmup_reference_points_loss_coef': {'clouds_classifier': 2000, 'tissues_classifier': 2000.0, 'distance_from_cloud_center_6h': 0, 'distance_from_cloud_center_dmso_24h': 0, 'distance_from_cloud_center_24h_without_dmso_24h': 0, 'vae_kld': 0.01,'vae_mse': 1000, 'treatment_vectors_collinearity_using_batch_treated': 0.0,'treatment_vectors_collinearity_using_batch_control': 0.0, 'drug_vectors_collinearity_using_batch_treated': 0,'drug_vectors_collinearity_using_batch_control': 0, 'treatment_vectors_different_directions_using_batch': 0,'drug_and_treatment_vectors_different_directions_using_batch': 0,  'treatment_and_drug_vectors_distance_p1_p2_loss': 0.0, 'treatment_and_drug_vectors_distance_p1_p2_to_treated_loss': 0.0,'treatment_vectors_magnitude_regulator': 0.0, 'drug_vectors_magnitude_regulator': 0.0}, 'trim_treated_clouds_ratio_to_keep':0.85, 'trim_untreated_clouds_and_time_24h_ratio_to_keep':0.5, 'clouds_trimming_epochs':[1000],'cross_validation_clouds': ['A375','raloxifene'],'clouds_to_augment': [],'treatment_and_drug_vectors_distance_loss_cdist_usage':true,'vae_decode_inner_dims':[]}" -job-prefix "base00_raloPDC" 
