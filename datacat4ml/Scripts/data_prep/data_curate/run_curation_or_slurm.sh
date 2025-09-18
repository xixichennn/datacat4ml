#!/bin/bash -l

#SBATCH --job-name=data_curate
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=2:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Run your Python script with arguments
python3 run_curation_or.py 

# run the command below in the terminal to submit the job
# sbatch run_curation_or_slurm.sh <num_jobs> with (the total number of combinations you want to process minus 1).
