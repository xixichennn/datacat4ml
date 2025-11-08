#!/bin/bash
#SBATCH --job-name=featurize_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=5:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# Create arrays for in_dirs, tasks and descriptor_cats
in_dirs=("/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/spl_hhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/spl_mhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/spl_lhd_or"\
        "/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/spl_mhd-effect_or"\
        #"/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/spl_hhd_gpcr"\
        #"/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/spl_mhd_gpcr"\
        #"/storage/homefs/yc24j783/datacat4ml/datacat4ml/Data/data_prep/data_split/spl_lhd_gpcr"
        )

descriptors=('ECFP4' 'ECFP6' 'MACCS' 'RDKITFP' 'PHARM2D' 'ERG'\
             'PHYSICOCHEM'\
             'SHAPE3D' 'AUTOCORR3D' 'RDF' 'MORSE' 'WHIM' 'GETAWAY')

rmvDs=(0 1)

# SLURM_ARRAY_TASK_ID will be automatically set by SLURM for each job in the array
in_dir="${in_dirs[$SLURM_ARRAY_TASK_ID % ${#in_dirs[@]}]}" 
descriptor="${descriptors[($SLURM_ARRAY_TASK_ID / ${#in_dirs[@]}) % ${#descriptors[@]}]}"
rmvD="${rmvDs[($SLURM_ARRAY_TASK_ID / (${#in_dirs[@]} * ${#descriptors[@]})) % ${#rmvDs[@]}]}"

python3 feat_smi_list.py --descriptor "$descriptor" --in_dir "$in_dir" --rmvD "$rmvD"

#### run the blow command in terminal to submit all the job
# sbatch --array=0-77 run_feat_slurm.sh 
# Replace 77 with the total number of combinations you want to process -1. 
# Here <num_jobs> = (6 * 13) -1 = 77

#### For jobs that can be done within 6 hours
# sbatch --array=0-44,47-50,53-56,59-62,65-68,71-74,77 run_feat_slurm_T_6h.sh

#### For jobs that need at least 22 hours
# Those jobs include SMILES failed to be embeded by 3D descriptors: hhd_gprc, mhs_gpcr
# sbatch --array=45,46,51,52,57,58,63,64,69,70,75,76 run_feat_slurm_T_22h.sh

### Calc Pharm2D only
# sbatch --array=0-5 run_feat_slurm.sh

### After data_split
# sbatch --array=0-103 run_feat_slurm.sh # 4*13*2 = 104
