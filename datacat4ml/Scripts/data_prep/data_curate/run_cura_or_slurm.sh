#!/bin/bash -l

#SBATCH --job-name=data_curate
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=2:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

rmvDs=(0 1)

# SLURM_ARRAY_TASK_ID
rmvD="${rmvDs[$SLURM_ARRAY_TASK_ID % ${#rmvDs[@]}]}"

# Run your Python script with arguments
python3 cura_or.py --rmvD="$rmvD"

# run the command below in the terminal to submit the job
# sbatch --array=0-1 run_cura_or_slurm.sh
