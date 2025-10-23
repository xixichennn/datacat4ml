#!/bin/bash
#SBATCH --job-name=featurize_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

rmvDupMols=(0 1)

# SLURM_ARRAY_TASK_ID
rmvDupMol="${rmvDupMols[$SLURM_ARRAY_TASK_ID % ${#rmvDupMols[@]}]}"

# Run the python script
python3 feat_post.py --rmvDupMol="$rmvDupMol"

# run the command below in the terminal to submit the job
# sbatch --array=0-1 run_feat_post_slurm.sh