# conda env: pyg (Python3.9.16)
import shutil
import argparse

from datacat4ml.const import *
from datacat4ml.Scripts.data_prep.data_curate.utils.curate_dataset_type import curate_datasets_and_get_stats

def main(task='cls'):

    """
    run the curation process on datasets including:
    - categorize datasets for ORs
    - heterogeneous datasets for ORs
    - heterogeneous datasets for GPCRs

    and get the stats for each dataset type.

    """

    # keep the 'task' in the argument to ensure the stats are generated only once
    print(f'----------->Task is {task}\n')


    ## ==== categorized data for ORs ====
    print('Processing categorized datasets of ORs...')
    #if os.path.exists(os.path.join(CURA_CAT_DATASETS_DIR, task)):
    #    # remove the directory and its contents
    #    shutil.rmtree(os.path.join(CURA_CAT_DATASETS_DIR, task))

    # Binding affinity
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_names, effect='bind', assay='RBA', std_types=["Ki", 'IC50'], 
                                input_path=CAT_DATASETS_DIR, output_path= CURA_CAT_DATASETS_DIR)

    # Agonism
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_names, effect='agon', assay='G_GTP', std_types=["EC50"], 
                                  input_path=CAT_DATASETS_DIR, output_path= CURA_CAT_DATASETS_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_names, effect='agon', assay='G_Ca', std_types=["EC50"], 
                                  input_path=CAT_DATASETS_DIR, output_path= CURA_CAT_DATASETS_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_names, effect='agon', assay='G_cAMP', std_types=["IC50", "EC50"], 
                                  input_path=CAT_DATASETS_DIR, output_path= CURA_CAT_DATASETS_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_names, effect='agon', assay='B_arrest', std_types=["EC50"], 
                                  input_path=CAT_DATASETS_DIR, output_path= CURA_CAT_DATASETS_DIR)
    
    ## Antagonism
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_names, effect='antag', assay='G_GTP', std_types=["IC50", "Ki", "Kb", "Ke"], 
                                  input_path=CAT_DATASETS_DIR, output_path= CURA_CAT_DATASETS_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_names, effect='antag', assay='B_arrest', std_types=["IC50"], 
                                  input_path=CAT_DATASETS_DIR, output_path= CURA_CAT_DATASETS_DIR)
    
    ## ==== het data for ORs ====
    print('Processing heterogeneous data of ORs...')
    if os.path.exists(os.path.join(CURA_HET_DATASETS_DIR, task)):
        # remove the directory and its contents
        shutil.rmtree(os.path.join(CURA_HET_DATASETS_DIR, task))
        
    curate_datasets_and_get_stats(dataset_type='het', task=task, target_list=OR_names, effect=None, assay=None, std_types=["Ki", "IC50", 'EC50'],
                                  input_path=HET_DATASETS_DIR, output_path=CURA_HET_DATASETS_DIR)

    ## ==== het data for GPCRs ====
    print('Processing heterogeneous data of GPCRs...') 
    if os.path.exists(os.path.join(CURA_GPCR_DATASETS_DIR, task)):
        # remove the directory and its contents
        shutil.rmtree(os.path.join(CURA_GPCR_DATASETS_DIR, task))
    GPCR_chembl_ids = [id for id in os.listdir(HET_GPCR_DIR) if os.path.isdir(os.path.join(HET_GPCR_DIR, id))]
    curate_datasets_and_get_stats(dataset_type='het', task=task, target_list=GPCR_chembl_ids, effect=None, assay=None, std_types=["Ki", "IC50", 'EC50'],
                                  input_path=HET_GPCR_DIR, output_path=CURA_GPCR_DATASETS_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curate datasets and get stats')
    parser.add_argument('--task', type=str, default='cls', help='task type: cls or reg')

    args = parser.parse_args()

    main(task=args.task)