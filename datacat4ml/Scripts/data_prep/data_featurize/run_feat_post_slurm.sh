#!/bin/bash
#SBATCH --job-name=featurize_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

rmv_dupMols=(0 1)

# SLURM_ARRAY_TASK_ID
rmv_dupMol="${rmv_dupMols[$SLURM_ARRAY_TASK_ID % ${#rmv_dupMols[@]}]}"

# Run the python script
python3 feat_post.py --rmv_dupMol="$rmv_dupMol"

# run the command below in the terminal to submit the job
# sbatch --array=0-1 run_feat_post_slurm.sh