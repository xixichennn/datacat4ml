#!/bin/bash
#SBATCH --job-name=intSplit_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Create arrays 
in_dirs=("/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_hhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_mhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_lhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_curate/cura_mhd-effect_or"
        )
rmvDupMols=(0 1)

# SLURM_ARRAY_TASK_ID will be automatically set by SLURM for each job in the array
in_dir="${in_dirs[$SLURM_ARRAY_TASK_ID % ${#in_dirs[@]}]}" 
rmvDupMol="${rmvDupMols[($SLURM_ARRAY_TASK_ID / ${#in_dirs[@]}) % ${#rmvDupMols[@]}]}"

python3 intSplit_mldata.py --in_dir "$in_dir" --rmvDupMol "$rmvDupMol"

#### run the blow command in terminal to submit all the job
# sbatch --array=0-7 run_intSplit_mldata_slurm.sh # 4*2 = 8
# Replace 7 with the total number of combinations you want to process -1.


