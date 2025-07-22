#!/bin/bash -l

#SBATCH --job-name=encode_assay
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00

# valid GRES specifications: rtx3090, rtx4090, a100, h100

dataset=/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/fsmol_alike/MHDsFold
assay_modes=("lsa" "clip")
assay_columns_lists=("assay_desc_only" "columns_short" "columns_middle" "columns_long" "columns_full")
#assay_columns_list=assay_desc_only
compound_modes=("morganc+rdkc" "sprsFP")
hyperparams=/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/model_dev/hparams/default.json
split=split
experiment=loss_fun_Con
#loss_funs=("BCE", "CE", "Con")
#loss_fun=Con
loss_fun=CE

# SLURM_ARRAY_TASK_ID is the index of the current job in the array
assay_mode="${assay_modes[$SLURM_ARRAY_TASK_ID % ${#assay_modes[@]}]}"
compound_mode="${compound_modes[($SLURM_ARRAY_TASK_ID / ${#assay_modes[@]}) % ${#compound_modes[@]}]}"
assay_columns_list="${assay_columns_lists[($SLURM_ARRAY_TASK_ID / (${#assay_modes[@]} * ${#compound_modes[@]})) % ${#assay_columns_lists[@]}]}"
#loss_fun="${loss_funs[($SLURM_ARRAY_TASK_ID / (${#assay_modes[@]} * ${#compound_modes[@]} * ${#assay_columns_lists[@]})) % ${#loss_funs[@]}]}"

# load environment variables for configuring wandb
#source ~/.bashrc
# Activate the Python environment
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Run with chunk parameters
python3 train_utils.py \
    --dataset ${dataset} \
    --assay_mode ${assay_mode} \
    --assay_columns_list ${assay_columns_list} \
    --compound_mode ${compound_mode} \
    --hyperparams ${hyperparams} \
    --split ${split} \
    --wandb \
    --experiment ${experiment} \
    --loss_fun ${loss_fun}

# run the command below in the terminal to submit the job
# Here <num_jobs> = 2 * 2 * 5 = 20
# sbatch --array=0-19 run_clamp_slurm.sh
