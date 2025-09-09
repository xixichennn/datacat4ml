#!/bin/bash -l

#SBATCH --job-name=data_categorize
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

python3 cat.py

# run the command below in the terminal to submit the job
# sbatch run_cat_slurm.sh
