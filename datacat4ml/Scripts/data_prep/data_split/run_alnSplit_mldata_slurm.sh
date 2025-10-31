#!/bin/bash
#SBATCH --job-name=alnSplit_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

rmvDs=(0 1)

# SLURM_ARRAY_TASK_ID will be automatically set by SLURM for each job in the array
rmvD="${rmvDs[$SLURM_ARRAY_TASK_ID % ${#rmvDs[@]}]}" 

python3 alnSplit_mldata.py --rmvD $rmvD

# run the below command in terminal
# sbatch --array=0-1 run_alnSplit_mldata_slurm.sh