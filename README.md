# Gene-expression

## How to run the project?
1. Put all raw data (txt, h5, csv etc.) files into `raw_data_folder`.
1. Run `src/datasets_and_organizers/cmap_organizer.py` in order to create an organized CMAP in `organized_data` folder.
1. Set the environment variable `PREFECT__FLOWS__CHECKPOINTING=true` in order to make the caching mechanism working.
1. Run `main.py`.

## Running on SLURM
1. CD to `/RG/compbio/groupData/emil/`.
1. Create a new version using `create-version.sh <version-name>` script. 
   1. Example: `./create-version.sh 0.8.0-small-warmup-no-reset-multi-select-full-data`.
1. Put the code to run in `./code/<version-name>`.
1. CD to the `./code/<version-name>` folder.
1. Load the anaconda module using `module load anaconda3`.
1. Activate the conda environment by using `source activate lsagne-pytorch`.
1. Run `PYTHONPATH=. python ./slurm_runner/sbatch_runner.py train -one-job -num-of-repeats=5`.
1. CD to the `run_sbatch_files` folder.
1. New sbatch files were created in `run_sbatch_files/<version-name>`.
1. Activate the base conda environment by using `source activate base`. It is a very important step, otherwise the next
   step will run in the scope of the current environment and fail.
1. Run Slurm by using the script `./run.sh <version-name>`.
1. To validate that the tasks are running please run `squeue -u $USER`.
1. In order to watch some statement every few seconds please use `watch -n 10 squeue -u $USER`.

## SLURM useful commands
- `srun -n1 -c1 -G1 --qos=gpu -w gpu5 nvidia-smi` 
- `srun -n1 -c1 --qos=gpu -w gpu5 nvidia-smi` 
- `srun -n 1 -c 1 -G 2 --qos=gpu --mem=1G -w gpu6 --pty bash`
- `squeue -u $USER --format=%i,%P,%j,%T`
- ```commandline
   for file in $(squeue -u $USER -h -o "%A" -t RUNNING | xargs -I {} scontrol show job {} -u $USER | sed -n 's/.*StdOut=\(.*\)/\1/p'); do
     cat $file | grep Epoch | tail -n 1
   done
   ```

## Using conda environments
- In order to install the environment from the yml file use the command `conda env create -f environment.yml`.
- In order to update use `conda env update -n <env name> -f environment.yml --prune`.

## Collecting results
```commandline
srun -n 1 -c 4 --mem=1G --pty python /RG/compbio/emil/code/baseline/slurm_runner/collect_slurm_batch_files_singleConf_allClouds_v1.py /RG/compbio/emil/slurm_results/python_results/baseline/basic_runs/ /RG/compbio/emil/slurm_results/python_results/baseline/output_v8_1.csv
```

## Running experiments
```commandline
conda activate /RG/compbio/emil/lsagnev8-test
srun -n 1 -c 32 --mem=16G --export=ALL,PYTHONPATH=/RG/compbio/emil/code/debug_29_4_2024 --pty python /RG/compbio/emil/code/debug_29_4_2024/slurm_runner/v2/generate_slurm_batch_files_singleConf_allClouds_v2.py "/RG/compbio/emil/code/debug_29_4_2024/slurm_runner/v2/running_profiles/baseline.yaml"
```

## Profiling PyTorch
Use documentation in `` and use command 
`tensorboard --logdir="C:\Users\Home\Google Drive\Master Degree\noam-git\Gene-expression\output\basic_dudi_runs_35-0.35.0\6-A375-geldanamycin"`,
it will find the profile file in 
