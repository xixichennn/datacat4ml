#!/bin/bash -l

#SBATCH --job-name=encode_assay
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00

# valid GRES specifications: rtx3090, rtx4090, a100, h100
parquet_path=/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/fsmol_alike/MHDsFold

columns_lists=("columns_short" "columns_middle" "columns_long" "columns_full")
encodings=("text" "lsa" "clip")

lsa_path=${parquet_path}/encoded_assay/lsa_models
train_set_size=1 #Yu? chatgpt: is it okay to set this to 1?
gpu=0
batch_size=2048
n_components=355

# SLURM_ARRAY_TASK_ID is the index of the current job in the array
columns_list="${columns_lists[$SLURM_ARRAY_TASK_ID % ${#columns_lists[@]}]}"
encoding="${encodings[($SLURM_ARRAY_TASK_ID / ${#columns_lists[@]}) % ${#encodings[@]}]}"

# Activate the Python environment
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Run with chunk parameters
python3 encode_assay.py \
    --assay_parquet_path ${parquet_path}/assay_info.parquet \
    --columns_list ${columns_list} \
    --encoding ${encoding} \
    --lsa_path ${lsa_path} \
    --train_set_size ${train_set_size} \
    --gpu ${gpu} \
    --batch_size ${batch_size} \
    --n_components ${n_components} 
# run the command below in the terminal to submit the job
# sbatch --array=0-11 run_encode_assay_slurm.sh 
# Here <num_jobs> = 4 * 3 = 12