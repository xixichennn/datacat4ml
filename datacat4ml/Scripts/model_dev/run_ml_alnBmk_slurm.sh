#!/bin/bash
#SBATCH --job-name=bmkML_aln_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=1-00:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Define arrays for parameters
bmk_type="aln"

algos=("RF" 
#"GB" "SVM" "KNN"
)
descriptors=('ECFP4' 
#'ECFP6' 'MACCS' 'RDKITFP' 'PHARM2D' 'ERG'\
             #'PHYSICOCHEM'\
             #'SHAPE3D' 'AUTOCORR3D' 'RDF' 'MORSE' 'WHIM' 'GETAWAY'
             )
rmvSs=(0 1)
aim_spl_combos=('lo_rs-lo' 'lo_cs' 'vs_rs-vs' 'vs_ch')
pls=('HoldoutCV' 'SingleNestedCV' 'NestedCV' 'ConsensusNestedCV')

# SLURM_ARRAY_TASK_ID
algo="${algos[$SLURM_ARRAY_TASK_ID % ${#algos[@]}]}"
descriptor="${descriptors[($SLURM_ARRAY_TASK_ID / ${#algos[@]}) % ${#descriptors[@]}]}"
rmvS="${rmvSs[($SLURM_ARRAY_TASK_ID / (${#algos[@]} * ${#descriptors[@]})) % ${#rmvSs[@]}]}"
aim_spl_combo="${aim_spl_combos[($SLURM_ARRAY_TASK_ID / (${#algos[@]} * ${#descriptors[@]} * ${#rmvSs[@]})) % ${#aim_spl_combos[@]}]}"
pl="${pls[($SLURM_ARRAY_TASK_ID / (${#algos[@]} * ${#descriptors[@]} * ${#rmvSs[@]} * ${#aim_spl_combos[@]})) % ${#pls[@]}]}"

# Run the python script
python3 ml_bmk.py --bmk_type="$bmk_type" \
                    --algo="$algo" \
                    --descriptor="$descriptor" \
                    --rmvS="$rmvS" \
                    --aim_spl_combo="$aim_spl_combo" \
                    --pl="$pl"

# run the command below in the terminal to submit the job
# sbatch --array=0-10 run_ml_alnBmk_slurm.sh # 4*13*2*4*4=1664 => array=0-1663

# test run command:
# sbatch --array=0-31 run_ml_alnBmk_slurm.sh # 1*1*2*4*4=32 => array=0-31


