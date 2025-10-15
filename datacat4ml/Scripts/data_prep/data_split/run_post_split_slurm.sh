#!/bin/bash
#SBATCH --job-name=post_split_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=6:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Create arrays
rmv_dupMols=(0 1)

# SLURM_ARRAY_TASK_ID will be automatically set by SLURM for each job in the array
rmv_dupMol="${rmv_dupMols[$SLURM_ARRAY_TASK_ID % ${#rmv_dupMols[@]}]}" 

# Submit the job
python3 post_split.py --rmv_dupMol $rmv_dupMol

# run the below command in terminal
# sbatch --array=0-1 run_post_split_slurm.sh