#!/bin/bash -l

#SBATCH --job-name=tuning4ml_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G

# Create arrays for File_paths, Tasks, Confidence_scores, Use_clusterings, Use_smotes, and Descriptor_cats
filepaths=("/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/cat_datasets")
tasks=("cls" "reg")
use_clusterings=(0 1)
use_smotes=(0 1)
descriptors=("ECFP4")

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate pyg

## SLURM_ARRAY_TASK_ID will be automatically set by Slurm for each job in the array
filepath="${filepaths[$SLURM_ARRAY_TASK_ID % ${#filepaths[@]}]}" # ${#filepaths[@]} is the length of the array
task="${tasks[($SLURM_ARRAY_TASK_ID / ${#filepaths[@]}) % ${#tasks[@]}]}"
use_clustering="${use_clusterings[($SLURM_ARRAY_TASK_ID / (${#filepaths[@]} * ${#tasks[@]})) % ${#use_clusterings[@]}]}"
use_smote="${use_smotes[($SLURM_ARRAY_TASK_ID / (${#filepaths[@]} * ${#tasks[@]} * ${#use_clusterings[@]})) % ${#use_smotes[@]}]}"
descriptor="${descriptors[($SLURM_ARRAY_TASK_ID / (${#filepaths[@]} * ${#tasks[@]} * ${#use_clusterings[@]} * ${#use_smotes[@]})) % ${#descriptors[@]}]}"

# Run your Python script with arguments
python3 tune_cat.py --file_path "$filepath" --task "$task" --use_clustering "$use_clustering" --use_smote "$use_smote" --descriptor "$descriptor"

# run the blow command in terminal to submit the job
# sbatch --array=0-8 tune_cat_slurm.sh # Replace <num_jobs> with the total number of combinations you want to process. 
# Here <num_jobs> = 1 * 2 * 2 * 2 * 1 = 8 