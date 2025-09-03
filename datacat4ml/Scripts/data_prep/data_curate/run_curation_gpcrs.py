# conda env: datacat (Python3.8.20)
import os
import shutil
import argparse

from datacat4ml.const import *
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.curate_dataset_type import curate_datasets_and_get_stats

def curate_GPCRs(task='cls', job_index=0, total_jobs=30):
    """Process chunks of GPCR targets in parallel
    
    Params:
    - task: str, 'cls' or 'reg'
    - job_index: int, index of the current job. It is 'chunk_index' in slurm script
    - total_jobs: int, total number of jobs. It is 'chunks_per_task' in slurm script

    return: None

    """
    # Get sorted list of all targets
    het_targets = sorted([id for id in os.listdir(HET_GPCR_DIR)
                        if os.path.isdir(os.path.join(HET_GPCR_DIR, id))])
    
    cat_targets = sorted([id for id in os.listdir(CAT_GPCR_DIR) 
                        if os.path.isdir(os.path.join(CAT_GPCR_DIR, id))])

    # Split into chunks
    def chunk_list(lst, job_idx, total_j):
        chunk = len(lst) // total_j
        start = job_idx * chunk
        end = (job_idx + 1) * chunk if job_idx != total_j - 1 else len(lst)
        return lst[start:end]

    current_het = chunk_list(het_targets, job_index, total_jobs)
    current_cat = chunk_list(cat_targets, job_index, total_jobs)

    print(f"Processing chunk {job_index+1}/{total_jobs} with:")
    print(f"- {len(current_het)} het targets")
    print(f"- {len(current_cat)} cat targets")
    
    # Process heterogeneous data
    print('\nProcessing heterogeneous data...')
    _process_het(current_het, task)

    # Process categorized data
    print('\nProcessing categorized datasets...')
    _process_cat(current_cat, task)
    
    

def _process_cat(targets, task):
    """Process categorized GPCR targets"""
    # Binding affinity
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='bind',assay='RBA', std_types=["Ki", 'IC50'], 
                                input_path=CAT_GPCR_DIR, output_path=CURA_CAT_GPCR_DIR)

    # Agonism
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='agon', assay='G_GTP', std_types=["EC50"], 
                                input_path=CAT_GPCR_DIR, output_path= CURA_CAT_GPCR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='agon', assay='G_Ca', std_types=["EC50"], 
                                input_path=CAT_GPCR_DIR, output_path= CURA_CAT_GPCR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='agon', assay='G_cAMP', std_types=["IC50", "EC50"], 
                                input_path=CAT_GPCR_DIR, output_path= CURA_CAT_GPCR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='agon', assay='B_arrest', std_types=["EC50"], 
                                input_path=CAT_GPCR_DIR, output_path= CURA_CAT_GPCR_DIR)

    ## Antagonism
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='antag', assay='G_GTP', std_types=["IC50", "Ki"], 
                                input_path=CAT_GPCR_DIR, output_path= CURA_CAT_GPCR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='antag', assay='G_Ca', std_types=["IC50"], 
                                input_path=CAT_GPCR_DIR, output_path= CURA_CAT_GPCR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=targets, effect='antag', assay='B_arrest', std_types=["IC50"], 
                                input_path=CAT_GPCR_DIR, output_path= CURA_CAT_GPCR_DIR)   

def _process_het(targets, task):
    """Process heterogeneous GPCR targets"""

    curate_datasets_and_get_stats(
        dataset_type='het', task=task, target_list=targets, effect=None,
        assay=None, std_types=["Ki", "IC50", 'EC50'],
        input_path=HET_GPCR_DIR, output_path=CURA_HET_GPCR_DIR
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cls")
    parser.add_argument("--job_index", type=int, default=0)
    parser.add_argument("--total_jobs", type=int, default=1)
    args = parser.parse_args()
    
    curate_GPCRs(
        task=args.task,
        job_index=args.job_index,
        total_jobs=args.total_jobs
    )