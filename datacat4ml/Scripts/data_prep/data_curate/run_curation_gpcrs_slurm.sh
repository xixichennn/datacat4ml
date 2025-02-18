#!/bin/bash -l

#SBATCH --job-name=data_curate
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-59  # 2 tasks * 30 chunks = 60 jobs

tasks=("cls" "reg")
total_tasks=${#tasks[@]} # calcute the number of elements in `tasks`. Here, it will be `2`.
chunks_per_task=30  # Split 300 targets into 10-target chunks

# Calculate current task and chunk
task_index=$((SLURM_ARRAY_TASK_ID / chunks_per_task))
chunk_index=$((SLURM_ARRAY_TASK_ID % chunks_per_task))
current_task=${tasks[$task_index]}

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Run with chunk parameters
python3 run_curation_gpcrs.py \
    --task "$current_task" \
    --job_index "$chunk_index" \
    --total_jobs "$chunks_per_task"

# run the command below in the terminal to submit the job
# sbatch run_curation_GPCRs_slurm.sh  
