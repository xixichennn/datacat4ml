import logging
logger = logging.getLogger(__name__)
from typing import List

import pandas as pd
from multiprocessing import cpu_count, Pool

from datacat4ml.const import *
from datacat4ml.utils import mkdirs
from datacat4ml.Scripts.data_prep.data_curate.utils.select_assays import select_assays
from datacat4ml.Scripts.data_prep.data_curate.utils.standardize_on_dataset import standardize_withvalue, standardize_novalue
from datacat4ml.Scripts.data_prep.data_curate.utils.apply_thresholds import apply_thresholds


DEFAULT_CLEANING = {
    "hard_only": False,
    "automate_threshold": True,
    "num_workers": cpu_count()
}


def curate_dataset(df: pd.DataFrame, task='reg', target='mor', effect=None, assay=None, std_type='Ki', 
                   output_path=CURA_CAT_OR_DIR) -> pd.DataFrame:
    """
    apply the curation pipeline to a dataframe

    param:
    -----
    target: str, could be target_name 'mor' or 'target_chembl_id' 'CHEMB233'
    output_path: str: The path to save the curated dataset

    return:
    df: pd.DataFrame: The curated dataset
    and save the dataset to the output_path
    """

    # remove index if it was saved with this file (back compatible)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    
    try:
        print(f"Curating dataset")
        if task == 'reg':
            df = select_assays(df, **DEFAULT_CLEANING)
            df = standardize_withvalue(df, **DEFAULT_CLEANING)
            df = apply_thresholds(df, **DEFAULT_CLEANING)
        elif task == 'cls':
            # separate the dataset into two: one with values and the other with values as 'None'
            # df_withvalue
            df_withvalue = df[df['standard_value'] != 'None']
            df_withvalue = select_assays(df_withvalue, **DEFAULT_CLEANING)
            print('start standardizing with value')
            df_withvalue = standardize_withvalue(df_withvalue, **DEFAULT_CLEANING)
            max_num_atom_df_withvalue = df_withvalue["max_num_atoms"].values[0]
            max_mw_df_withvalue = df_withvalue["max_molecular_weight"].values[0]
            print('start applying thresholds')
            df_withvalue = apply_thresholds(df_withvalue, **DEFAULT_CLEANING)
            threshold = float(df_withvalue['threshold'][0])

            # df_novalue
            df_novalue = df[df['standard_value'] == 'None']
            print(f'The length of df_novalue is {len(df_novalue)}')
            if len(df_novalue) > 0:
                df_novalue = standardize_novalue(df_novalue, **DEFAULT_CLEANING)
                # apply the threshold to the df_novalue
                max_num_atom_df_novalue = df_novalue['max_num_atoms'].values[0]
                max_mw_df_novalue = df_novalue['max_molecular_weight'].values[0]
                # set the activity of the mols in the df_novalue to 'inactive
                df_novalue['activity_string'] = 'inactive'
                df_novalue['activity'] = 0.0
                df_novalue['threshold'] = threshold

                df = pd.concat([df_withvalue, df_novalue])
                df["max_num_atoms"] = max(max_num_atom_df_withvalue, max_num_atom_df_novalue)
                df["max_molecular_weight"] = max(max_mw_df_withvalue, max_mw_df_novalue)
            else:
                df = df_withvalue
                df["max_num_atoms"] = max_num_atom_df_withvalue
                df["max_molecular_weight"] = max_mw_df_withvalue
        
        # save the dataset
        if effect is None or assay is None:
            dataset_prefix = f'{target}_{std_type}'
        else:
            dataset_prefix = f'{target}_{effect}_{assay}_{std_type}'
        
        # convert the datatype of effect and assay to string
        effect = str(effect)
        assay = str(assay)

        # add columns to the dataframe
        df['target'] = target
        df['effect'] = effect
        df['assay'] = assay
        df['std_type'] = std_type
        output_full_path = os.path.join(output_path, task, f'{dataset_prefix}_curated.csv')
        mkdirs(os.path.dirname(output_full_path))
        df.to_csv(output_full_path)

        print(f'Done curation.\n')

        return df


    except Exception as e:
        df = pd.DataFrame()
        logger.warning(f"Failed curating the dataset due to {e}")

        return df
    
def check_dataset_aval(dataset_type, target, effect, assay, std_type, input_path):
    """
    Check if the dataframe exists, or if it is empty.
    
    param: 
    ----------
    dataset_type: str: The categorization method. It could be either 'het' or 'cat'
    target: str: The target name (e.g. 'mor') or target_chembl_id (e.g. 'CHEMBL233')
    effect: str: The effect of the dataset. e.g. 'bind', 'antag'
    assay: str: The assay of the dataset. e.g. 'RBA'
    std_type: str: The standard type of the dataset. e.g. 'Ki', 'IC50'
    input_path: str: The path to the input datasets (should be like `CAT_OR_DIR`, `HET_DATASETS_DIR`) 

    
    input_path: str: The path to the input datasets that are to be curated

    return:
    df: pd.DataFrame: read the dataframe if it exists, else return None
    """
    # check if the dataset exists in the input_path
    if dataset_type == 'het':
        df_path = os.path.join(input_path, target, std_type, f'{target}_{std_type}.csv')
        if not os.path.exists(df_path):
            print(f'No dataset for {target}-{std_type}')
            return pd.DataFrame()
    elif dataset_type == 'cat':
        df_path = os.path.join(input_path, target, effect, assay, std_type, f'{target}_{effect}_{assay}_{std_type}Final_df.csv')
        if not os.path.exists(df_path):
            print(f'No dataset for {target}_{effect}_{assay}_{std_type}')
            return pd.DataFrame()
    
    # check if the dataset is empty
    df = pd.read_csv(df_path)
    if df.empty:
        print(f'Empty dataset for {target}-{effect}-{assay}-{std_type}')
        return pd.DataFrame()
    else:
        return df

def curate_datasets_and_get_stats(dataset_type='het', task='cls', target_list: List[str]= OR_names, effect='bind', assay='RBA', std_types=["Ki", 'IC50'], 
                                  input_path=CAT_OR_DIR, output_path= CURA_CAT_OR_DIR):
        """
        curate a series of datasets and get the stats for each dataset

        param:
        ----------
        dataset_type: str: The type of dataset to curate. It could be either 'het' or 'cat'
        task: str: The task to perform. It could be either 'cls' or 'reg'
        target_list: List[str]: The list of targets to process. The element of the list could be either target name (e.g. 'mor') or target_chembl_id (e.g. 'CHEMBL1824')
        effect: str: The effect of the dataset. It could be either 'bind' or 'antag'
        assay: str: The assay of the dataset. It could be either 'RBA' or 'FAC'
        std_types: List[str]: The list of standard types to process. The element of the list could be either 'Ki' or 'IC50'
        input_path: str: The path to the datasets.
        output_path: str: The path to save the curated datasets

        output:
        ----------
        write the stats of the curated datasets to a csv file
        
        """

        for target in target_list:
                
                for std_type in std_types:

                        print(f"Processing {target}_{effect}_{assay}_{std_type}...")

                        # Load the data
                        df = check_dataset_aval(dataset_type, target, effect, assay, std_type, input_path)
                        
                        if len(df) > 0:
                                raw_size = len(df)
                                print(f'The length of the raw dataset is {raw_size}')
                        
                                # run the curation pipeline
                                curated_df = curate_dataset(df, task, target, effect, assay, std_type, output_path)
                        
                                # get the stats and save them in a csv file
                                stats_file = os.path.join(output_path, task, f'{task}_stats.csv')

                                if not os.path.exists(stats_file): # don't use check_file_exists() and then remove the file if it exists
                                        mkdirs(os.path.dirname(stats_file))
                                        with open(stats_file, 'w') as f:
                                                f.write('task, target, effect, assay, standard_type, raw_size, curated_sized, curated_size_delta, threshold, num_active, num_inactives, %_actives\n')
                                if len(curated_df) > 0:
                                        with open(stats_file, 'a') as f:
                                                # Print something for number check
                                                curated_sized = len(curated_df)
                                                curated_size_delta = raw_size - curated_sized
                                                threshold = curated_df['threshold'].unique()[0]
                                                num_active = sum(curated_df['activity'])
                                                if curated_sized == 0:
                                                        percent_a = 0
                                                else:
                                                        percent_a = round(num_active / curated_sized * 100, 2)
                                                
                                                f.write(f'{task}, {target}, {effect}, {assay}, {std_type}, {raw_size}, {curated_sized}, {curated_size_delta}, {threshold}, {num_active}, {curated_sized -num_active}, {percent_a}\n')
                print(f'====================\n')         

