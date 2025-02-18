#!/bin/bash -l

#SBATCH --job-name=data_curate
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00

tasks=("cls" "reg")

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# SLURM_ARRAY_TASK_ID will be automatically set by Slurm for each job in the array
tasks="${tasks[$SLURM_ARRAY_TASK_ID % ${#tasks[@]}]}"

# Run your Python script with arguments
python3 run_curation_ors.py --task "$tasks"

# run the command below in the terminal to submit the job
# sbatch --array=0-1 run_curation_slurm.sh <num_jobs> with (the total number of combinations you want to process minus 1).
# Here <num_jobs> = 1*2 = 2
