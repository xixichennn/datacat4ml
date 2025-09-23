#!/bin/bash
#SBATCH --job-name=tsne_job
#SBATCH --partition=bdw
#SBATCH --ntasks=1
#SBATCH --mem=28G
#SBATCH --time=3:00:00

# Activate the Python environment 
source /storage/homefs/yc24j783/miniconda3/etc/profile.d/conda.sh
conda activate datacat

# parameters
descriptors=('ECFP4' 'ECFP6' 'MACCS' 'RDKITFP' 'PHARM2D' 'ERG'\
             'PHYSICOCHEM'\
             'SHAPE3D' 'AUTOCORR3D' 'RDF' 'MORSE' 'WHIM' 'GETAWAY')
# SLURM_ARRAY_TASK_ID will be automatically set by SLURM for each job in the array
descriptor="${descriptors[$SLURM_ARRAY_TASK_ID % ${#descriptors[@]}]}" 

python3 tsne.py --descriptor "$descriptor"

# run the blow command in terminal to submit all the job
# sbatch --array=0-12 run_tsne_slurm.sh