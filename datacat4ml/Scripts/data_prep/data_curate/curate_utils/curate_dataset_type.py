import os
import logging
logger = logging.getLogger(__name__)
from typing import List

import pandas as pd
from multiprocessing import cpu_count, Pool

from datacat4ml.const import CAT_HHD_OR_DIR, CURA_HHD_OR_DIR, CURA_LHD_OR_DIR, CURA_LHD_GPCR_DIR
from datacat4ml.const import OR_chemblids
from datacat4ml.utils import mkdirs
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.select_assays import select_assays
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.standardize_on_dataset import standardize
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.apply_thresholds import apply_thresholds


DEFAULT_CLEANING = {
    "hard_only": False, # keep weak actives/inactives.
    "automate_threshold": True,
    "num_workers": cpu_count()}

def curate(df: pd.DataFrame) -> pd.DataFrame:
    """
    apply the curation pipeline to a dataframe

    param:
    -----
    target: str, should be 'target_chembl_id' 'CHEMB233'

    return:
    -----
    df: pd.DataFrame: The curated dataset and save it to the output_path
    """

    # remove index if it was saved with this file (back compatible)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    try:
        print(f"Curating dataset")
        df = select_assays(df, **DEFAULT_CLEANING)
        df = standardize(df, **DEFAULT_CLEANING)
        df = apply_thresholds(df, **DEFAULT_CLEANING)

        print(f'Done curation.\n')

        return df

    except Exception as e:
        df = pd.DataFrame()
        logger.warning(f"Failed curating the dataset due to {e}")

        return df


def run_curation(ds_level='hhd', input_path=CAT_HHD_OR_DIR, output_path= CURA_HHD_OR_DIR,
                targets_list: List[str]= OR_chemblids, effect='bind', assay='RBA', std_types=["Ki", 'IC50'], 
                ds_type= 'or'):
    """
    curate a series of datasets and get the stats for each dataset

    param:
    ----------
    ds_level: str: The categorization level of dataset to curate. It could only be either 'hhd' or 'mhd', cannot be 'lhd'.
    input_path: str: The path to read the datasets. e.g `CAT_HHD_OR_DIR`, `CAT_MHD_OR_DIR`
    output_path: str: The path to save the curated datasets. e.g `CURA_HHD_OR_DIR`, `CURA_MHD_OR_DIR`
    targets_list: List[str]: The list of targets to process. The element of the list could be target_chembl_id (e.g. 'CHEMBL1824')
    effect: str: The effect of the dataset. It could be either 'bind' or 'antag'
    assay: str: The assay of the dataset. It could be either 'RBA'
    std_types: List[str]: The list of standard types to process. The element of the list could be either 'Ki' or 'IC50'
    ds_type: str: The type of dataset. It could be either 'or' or 'gpcr'
    

    output:
    ----------
    write the stats of the curated datasets to a csv file
    
    """

    for target in targets_list:
        print(f'Processing target {target} ...')
            
        for std_type in std_types:
            print(f'Processing {std_type} ...')

            # ================= curate hhd or mhd ================
            print(f'========================= Curating hhd/mhd dataset===========================')
            if ds_level == 'hhd':
                print(f"Processing hhd: {target}_{std_type}...")
                input_df_path = os.path.join(input_path, target, std_type, f'{target}_{std_type}_hhd_df.csv')

            elif ds_level == 'mhd':
                print(f"Processing mhd: {target}_{effect}_{assay}_{std_type}...")
                input_df_path = os.path.join(input_path, target, effect, assay, std_type, f'{target}_{effect}_{assay}_{std_type}_mhd_df.csv')

            if os.path.exists(input_df_path):
                input_df = pd.read_csv(input_df_path)
                if len(input_df) == 0:
                    print(f'No data points in the dataset at {input_df_path}')
                else: 
                    raw_size = len(input_df)
                    print(f'The length of the raw dataset is {raw_size}')
                    # run the curation pipeline
                    curated_df = curate(input_df)

                    if len(curated_df) == 0:
                        print(f'No data points left after curation for the dataset at {input_df_path}')
                    elif len(curated_df) > 0:
                        curated_df['target'] = target
                        curated_df['effect'] = effect
                        curated_df['assay'] = assay
                        curated_df['std_type'] = std_type

                        if ds_level == 'hhd':
                            if len(curated_df) < 50:
                                filename = f'{target}_{std_type}_hhd_s50_curated.csv'
                            else:
                                filename = f'{target}_{std_type}_hhd_b50_curated.csv'
                        elif ds_level == 'mhd':
                            if len(curated_df) < 50:
                                filename = f'{target}_{effect}_{assay}_{std_type}_mhd_s50_curated.csv'
                            else:
                                filename = f'{target}_{effect}_{assay}_{std_type}_mhd_b50_curated.csv'
                        
                        mkdirs(output_path)
                        curated_df.to_csv(os.path.join(output_path, filename))
                        # get the stats and save them in a csv file
                        stats_file_path = os.path.join(output_path, f'cura_{ds_level}_{ds_type}_stats.csv')

                        if not os.path.exists(stats_file_path): # don't use check_file_exists() and then remove the file if it exists
                            mkdirs(os.path.dirname(stats_file_path))
                            with open(stats_file_path, 'w') as f:
                                f.write('ds_level,target,effect,assay,standard_type,assay_chembl_id,raw_size,curated_size,removed_size,threshold,num_active,num_inactive,%_active\n')
                    
                        with open(stats_file_path, 'a') as f:
                            # Print something for number check
                            curated_size = len(curated_df)
                            removed_size = raw_size - curated_size
                            threshold = curated_df['threshold'].unique()[0]
                            num_active = sum(curated_df['activity'])
                            num_inactive = curated_size - num_active
                            if curated_size == 0:
                                    percent_a = 0
                            else:
                                    percent_a = round(num_active / curated_size * 100, 2)

                            f.write(f"{ds_level},{target},{effect},{assay},{std_type},{None},{raw_size},{curated_size},{removed_size},{threshold},{num_active},{num_inactive},{percent_a}\n")

                    # ================= curate lhd ================
                    print(f'======= Curating lhd dataset=======')
                    lhd_path = os.path.join(os.path.dirname(input_df_path), 'lhd')

                    if os.path.exists(lhd_path):
                        # collect all the assay_chembl_ids in the lhd_path
                        lhd_files = [f for f in os.listdir(lhd_path)]
                        lhd_assay_chembl_ids = list(set([f.split('_')[4] for f in lhd_files]))

                        for assay_chembl_id in lhd_assay_chembl_ids:
                            print(f'Processing lhd: {target}_{effect}_{assay}_{std_type}_{assay_chembl_id}...')
                            lhd_df_path = os.path.join(lhd_path, f'{target}_{effect}_{assay}_{std_type}_{assay_chembl_id}_lhd_df.csv')
                            lhd_df = pd.read_csv(lhd_df_path)
                            if len(lhd_df) == 0:
                                print(f'No data points in the dataset at {lhd_df_path}')
                            else:
                                lhd_raw_size = len(lhd_df)
                                print(f'The length of the raw lhd dataset is {lhd_raw_size}')
                                # run the curation pipeline
                                curated_lhd_df = curate(lhd_df)

                                if len(curated_lhd_df) == 0:
                                    print(f'No data points left after curation for the dataset at {lhd_df_path}')
                                elif len(curated_lhd_df) > 0:
                                    curated_lhd_df['target'] = target
                                    curated_lhd_df['effect'] = effect
                                    curated_lhd_df['assay'] = assay
                                    curated_lhd_df['std_type'] = std_type
                                    if len(curated_lhd_df) < 50:
                                        filename = f'{target}_{effect}_{assay}_{std_type}_{assay_chembl_id}_lhd_s50_curated.csv'
                                    else:
                                        filename = f'{target}_{effect}_{assay}_{std_type}_{assay_chembl_id}_lhd_b50_curated.csv'
                                    if ds_type == 'gpcr':
                                        output_lhd_path = CURA_LHD_GPCR_DIR
                                    elif ds_type == 'or':
                                        output_lhd_path = CURA_LHD_OR_DIR
                                    
                                    mkdirs(output_lhd_path)
                                    curated_lhd_df.to_csv(os.path.join(output_lhd_path, filename))

                                    # get the stats and save them in a csv file
                                    lhd_stats_file_path = os.path.join(output_lhd_path, f'cura_lhd_{ds_type}_stats.csv')

                                    if not os.path.exists(lhd_stats_file_path): # don't use check_file_exists() and then remove the file if it exists
                                        mkdirs(os.path.dirname(lhd_stats_file_path))
                                        with open(lhd_stats_file_path, 'w') as f:
                                            f.write("ds_level,target,effect,assay,standard_type,assay_chembl_id,raw_size,curated_size,removed_size,threshold,num_active,num_inactive,%_active\n")

                                
                                    with open(lhd_stats_file_path, 'a') as f:
                                        # Print something for number check
                                        curated_sized = len(curated_lhd_df)
                                        removed_size = lhd_raw_size - curated_sized
                                        threshold = curated_lhd_df['threshold'].unique()[0]
                                        num_active = sum(curated_lhd_df['activity'])
                                        num_inactive = curated_sized - num_active
                                        if curated_sized == 0:
                                                percent_a = 0
                                        else:
                                                percent_a = round(num_active / curated_sized * 100, 2)

                                        f.write(f"lhd,{target},{effect},{assay},{std_type},{assay_chembl_id},{raw_size},{curated_sized},{removed_size},{threshold},{num_active},{num_inactive},{percent_a}\n")
                    else:
                        print(f'No dataset at {lhd_path}')

            else:
                print(f'No dataset at {input_df_path}')

    print(f'====================\n')

