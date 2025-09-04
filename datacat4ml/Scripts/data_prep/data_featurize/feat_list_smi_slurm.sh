#!/bin/bash
#SBATCH --job-name=featurize_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Create arrays for in_dirs, tasks and descriptor_cats
in_dirs=("/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_het_ors"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_cat_ors"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_lhd_ors"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_cat_50_5k_ors")

tasks=("cls" "reg")
#descriptor_cats=("FP" "PHYSICOCHEM" "THREE_D" "TOKENS" "ONEHOT" "GRAPH")
descriptor_cats=("FP" "PHYSICOCHEM" "THREE_D")


# SLURM_ARRAY_TASK_ID will be automatically set by Slurm for each job in the array
in_dir="${in_dirs[$SLURM_ARRAY_TASK_ID % ${#in_dirs[@]}]}"
task="${tasks[($SLURM_ARRAY_TASK_ID / ${#in_dirs[@]}) % ${#tasks[@]}]}"
descriptor_cat="${descriptor_cats[($SLURM_ARRAY_TASK_ID / (${#in_dirs[@]} * ${#tasks[@]})) % ${#descriptor_cats[@]}]}"

# Run your Python script with arguments
python3 feat_list_smi.py --in_dir "$in_dir" --task "$task" --descriptor_cat "$descriptor_cat"

# run the blow command in terminal to submit the job
# sbatch --array=0-23 feat_list_smi_slurm.sh 
# Replace 35 with the total number of combinations you want to process -1. 
# Here <num_jobs> = (4 * 2 * 3) -1 = 23