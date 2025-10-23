#!/bin/bash
#SBATCH --job-name=post_split_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Create arrays
rmvDupMols=(0 1)

# SLURM_ARRAY_TASK_ID will be automatically set by SLURM for each job in the array
rmvDupMol="${rmvDupMols[$SLURM_ARRAY_TASK_ID % ${#rmvDupMols[@]}]}" 

# Submit the job
python3 intSplit_post.py --rmvDupMol $rmvDupMol

# run the below command in terminal
# sbatch --array=0-1 run_intSplit_post_slurm.sh