#!/bin/bash
#SBATCH --job-name=featurize_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

python3 feat_postproc.py

# run the command below in the terminal to submit the job
# sbatch run_feat_postproc_slurm.sh