#!/bin/bash
#SBATCH --job-name=intSplit_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=6:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Create arrays 
in_dirs=("/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_hhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_mhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_lhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_mhd-effect_or"
        )
rmv_stereos=(0 1)
rmv_dupMols=(0 1)

# SLURM_ARRAY_TASK_ID will be automatically set by SLURM for each job in the array
in_dir="${in_dirs[$SLURM_ARRAY_TASK_ID % ${#in_dirs[@]}]}" 
rmv_stereo="${rmv_stereos[($SLURM_ARRAY_TASK_ID / ${#in_dirs[@]}) % ${#rmv_stereos[@]}]}"
rmv_dupMol="${rmv_dupMols[($SLURM_ARRAY_TASK_ID / (${#in_dirs[@]} * ${#rmv_stereos[@]})) % ${#rmv_dupMols[@]}]}"

python3 intSplit_mldata.py --in_dir "$in_dir" --rmv_stereo "$rmv_stereo" --rmv_dupMol "$rmv_dupMol"

#### run the blow command in terminal to submit all the job
# sbatch --array=0-15 run_intSplit_mldata_slurm.sh # 4*2*2 = 16
# Replace 7 with the total number of combinations you want to process -1. 


