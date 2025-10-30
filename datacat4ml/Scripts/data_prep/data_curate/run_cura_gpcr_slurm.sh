#!/bin/bash -l

#SBATCH --job-name=data_curate
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=5:00:00
#SBATCH --array=0-29  # 1 tasks * 30 chunks = 30 jobs


job_index=$SLURM_ARRAY_TASK_ID
chunks_per_task=30

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Run with chunk parameters
python3 cura_gpcr.py \
    --job_index "$job_index" \
    --total_jobs "$chunks_per_task"

# run the command below in the terminal to submit the job
# sbatch run_cura_gpcr_slurm.sh
