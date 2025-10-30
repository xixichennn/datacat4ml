# conda env: datacat (Python3.8.20)
import os
import shutil
import argparse

from datacat4ml.const import CURA_HHD_GPCR_DIR, CURA_MHD_GPCR_DIR
from datacat4ml.const import CAT_HHD_GPCR_DIR, CAT_MHD_GPCR_DIR
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.curate_dataset_type import run_curation

def curate_GPCRs(job_index=0, total_jobs=30):
    """Process chunks of GPCR targets in parallel
    
    Params:
    - job_index: int, index of the current job. It is 'chunk_index' in slurm script
    - total_jobs: int, total number of jobs. It is 'chunks_per_task' in slurm script

    return: None

    """
    # Get sorted list of all targets
    hhd_targets = sorted([id for id in os.listdir(CAT_HHD_GPCR_DIR)
                        if os.path.isdir(os.path.join(CAT_HHD_GPCR_DIR, id))])
    
    mhd_targets = sorted([id for id in os.listdir(CAT_MHD_GPCR_DIR) 
                        if os.path.isdir(os.path.join(CAT_MHD_GPCR_DIR, id))])

    # Split into chunks
    def chunk_list(lst, job_idx, total_j):
        chunk = len(lst) // total_j
        start = job_idx * chunk
        end = (job_idx + 1) * chunk if job_idx != total_j - 1 else len(lst)
        return lst[start:end]

    current_hhd = chunk_list(hhd_targets, job_index, total_jobs)
    current_mhd = chunk_list(mhd_targets, job_index, total_jobs)

    print(f"Processing chunk {job_index+1}/{total_jobs} with:")
    print(f"- {len(current_hhd)} hhd targets")
    print(f"- {len(current_mhd)} mhd targets")

    # Process heterogeneous data
    print('\nProcessing hhd_gpcr...')
    _process_hhd(current_hhd)

    # Process categorized data
    print('\nProcessing mhd_gpcr...')
    _process_mhd(current_mhd)

    

def _process_mhd(targets):
    """Process mhd_gpcr"""
    # Binding affinity
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path=CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='bind',assay='RBA', standard_types=["Ki", 'IC50'], ds_type='gpcr', rmvD=0)

    # Agonism
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path= CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='agon', assay='G-GTP', standard_types=["EC50"], ds_type='gpcr', rmvD=0)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path= CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='agon', assay='G-Ca', standard_types=["EC50"], ds_type='gpcr', rmvD=0)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path= CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='agon', assay='G-cAMP', standard_types=["IC50", "EC50"], ds_type='gpcr', rmvD=0)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path= CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='agon', assay='B-arrest', standard_types=["EC50"], ds_type='gpcr', rmvD=0)

    ## Antagonism
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path= CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='antag', assay='G-GTP', standard_types=["IC50", "Ki"], ds_type='gpcr', rmvD=0)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path= CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='antag', assay='G-Ca', standard_types=["IC50"], ds_type='gpcr', rmvD=0)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_GPCR_DIR, output_path= CURA_MHD_GPCR_DIR,
                 targets_list=targets, effect='antag', assay='B-arrest', standard_types=["IC50"], ds_type='gpcr', rmvD=0)

def _process_hhd(targets):
    """\nProcess hhhd_gpcr"""

    run_curation(ds_cat_level='hhd', input_path=CAT_HHD_GPCR_DIR, output_path=CURA_HHD_GPCR_DIR,
                 targets_list=targets, effect=None, assay=None, standard_types=["Ki", "IC50", 'EC50'], ds_type='gpcr', rmvD=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--total_jobs", type=int, default=1)
    args = parser.parse_args()
    
    curate_GPCRs(
        job_index=args.job_index,
        total_jobs=args.total_jobs
    )