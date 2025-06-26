#!/bin/bash -l

#SBATCH --job-name=encode_assay
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00

parquet_path=/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/fsmol_alike/MHDsFold
compounds2smiles=${parquet_path}/compound_smiles.parquet
fp_types=("sprsFP" "morganc+rdkc")
njobs=32

# SLURM_ARRAY_TASK_ID is the index of the current job in the array
fp_type="${fp_types[$SLURM_ARRAY_TASK_ID % ${#fp_types[@]}]}"

# Activate the Python environment
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Run with chunk parameters
python3 encode_compound.py \
    --compounds2smiles ${compounds2smiles} \
    --fp_type ${fp_type} \
    --njobs ${njobs} \

# run the command below in the terminal to submit the job
# sbatch --array=0-1 run_encode_compound_slurm.sh
# Here <num_jobs> = 2