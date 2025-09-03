# conda env: datacat (Python3.8.20)
import shutil
import argparse

from datacat4ml.const import *
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.curate_dataset_type import curate_datasets_and_get_stats
def curate_ORs(task='cls'):

    """
    run the curation process on datasets including:
    - categorize datasets for ORs
    - heterogeneous datasets for ORs

    and get the stats for each dataset type.

    param:
    ----------
    task: str: The task to perform. It could be either 'cls' or 'reg'

    """

    # keep the 'task' in the argument to ensure the stats are generated only once
    print(f'----------->Task is {task}\n')

    ## ==== het data for ORs ====
    print('Processing heterogeneous data of ORs...')
    if os.path.exists(os.path.join(CURA_HET_OR_DIR, task)):
        # remove the directory and its contents
        shutil.rmtree(os.path.join(CURA_HET_OR_DIR, task))
        
    curate_datasets_and_get_stats(dataset_type='het', task=task, target_list=OR_chemblids, effect=None, assay=None, std_types=["Ki", "IC50", 'EC50'],
                                input_path=HET_OR_DIR, output_path=CURA_HET_OR_DIR)

    ## ==== categorized data for ORs ====
    print('Processing categorized datasets of ORs...')
    #if os.path.exists(os.path.join(CURA_CAT_OR_DIR, task)):
    #    # remove the directory and its contents
    #    shutil.rmtree(os.path.join(CURA_CAT_OR_DIR, task))

    # Binding affinity
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_chemblids, effect='bind', assay='RBA', std_types=["Ki", 'IC50'], 
                                input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR)

    # Agonism
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_chemblids, effect='agon', assay='G_GTP', std_types=["EC50"], 
                                input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_chemblids, effect='agon', assay='G_Ca', std_types=["EC50"], 
                                input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_chemblids, effect='agon', assay='G_cAMP', std_types=["IC50", "EC50"], 
                                input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_chemblids, effect='agon', assay='B_arrest', std_types=["EC50"], 
                                input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR)
        
    ## Antagonism
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_chemblids, effect='antag', assay='G_GTP', std_types=["IC50", "Ki", "Kb", "Ke"], 
                                input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR)
    curate_datasets_and_get_stats(dataset_type='cat', task=task, target_list=OR_chemblids, effect='antag', assay='B_arrest', std_types=["IC50"], 
                                input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curate datasets and get stats')
    parser.add_argument('--task', type=str, default='cls', help='task type: cls or reg')

    args = parser.parse_args()

    curate_ORs(task=args.task)

