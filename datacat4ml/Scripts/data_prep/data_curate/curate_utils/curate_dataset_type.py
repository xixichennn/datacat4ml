import os
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict

import pandas as pd
from multiprocessing import cpu_count, Pool

from datacat4ml.const import CURA_DATA_DIR, CAT_HHD_OR_DIR, CURA_HHD_OR_DIR, CURA_MHD_OR_DIR, CURA_LHD_OR_DIR, CURA_HHD_GPCR_DIR, CURA_LHD_GPCR_DIR
from datacat4ml.const import OR_chemblids
from datacat4ml.utils import mkdirs
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.select_assays import select_assays
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.standardize_on_dataset import standardize
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.apply_thresholds import apply_thresholds

from datacat4ml.Scripts.data_prep.data_split.ml_data_split import find_stereochemical_siblings


DEFAULT_CLEANING = {
    "hard_only": False, # keep weak actives/inactives.
    "automate_threshold": True,
    "num_workers": cpu_count()}

def curate(df: pd.DataFrame) -> pd.DataFrame:
    """
    apply the curation pipeline to a dataframe

    param:
    -----
    df: pd.DataFrame: The input dataset to be curated, should be from cat_hhd or cat_mhd

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
        df = apply_thresholds(df, aim='vs', **DEFAULT_CLEANING)
        df = apply_thresholds(df, aim='lo', **DEFAULT_CLEANING)

        print(f'Done curation.\n')

        return df

    except Exception as e:
        df = pd.DataFrame()
        logger.warning(f"Failed curating the dataset due to {e}")

        return df

# =============================================================================
# process columns: rename, add, delete columns 
# =============================================================================
effect_dict = {
    'bind': 'binding affinity',
    'agon': 'agonism',
    'antag': 'antagonism'}

assay_dict = {
    'RBA': 'Receptor binding assay: radioligand binding assay',
    'G-GTP': 'G-protein dependent functional assays: GTPγS binding assay',
    'G-cAMP': 'G-protein dependent functional assays: cAMP accumulation assay',
    'G-Ca': 'G-protein dependent functional assays: IP3/IP1 and calcium accumulation assays',
    'B-arrest': 'Arrestin recruitment assay',
}

def rename_add_delete_cols(df: pd.DataFrame) -> pd.DataFrame:
   
    # add the following columns:
    df['effect_description'] = df['effect'].map(effect_dict)
    df['assay_keywords_description'] = df['assay'].map(assay_dict)

    # rename the following columns:
    df.rename(columns={
        'assay_desc': 'assay_description',
        'assay_type_desc': 'assay_type_description',
        'relationship_type_desc': 'relationship_type_description',
        'confidence_score_desc': 'confidence_score_description',
        'assay_info_hash': 'assay_metadata_hash',
        }, 
        inplace=True)
    
    # delete the following columns
    df.drop(columns=['target', 'std_type', 'max_num_atoms', 'max_molecular_weight',
                    'relationship_type', 'relationship_type_description'], inplace=True)

    return df

#=============================================================================
# delete_failed_smi
#=============================================================================
failed_dict = { 
    # For hhd_gpcr, 10 files, 23 failed times, and 10 unique failed SMILES
    # file name: {row index in the file: SMILES}
    "CHEMBL5850_EC50_hhd_b50_curated.csv":{26: "COc1ccc(-c2ccc(C(=O)Nc3cccc(Cn4ncc(N5C[C@H]6C[C@H]5CN6C)c(Cl)c4=O)c3C)cc2)cn1"},
    "CHEMBL256_EC50_hhd_b50_curated.csv":{108:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL226_IC50_hhd_b50_curated.csv":{132:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         133:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         136:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         138:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         139:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         142:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         146:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21"},
    "CHEMBL226_EC50_hhd_b50_curated.csv":{196:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL256_Ki_hhd_b50_curated.csv":{3439:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL251_Ki_hhd_b50_curated.csv":{4280:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL251_EC50_hhd_b50_curated.csv":{163:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL226_Ki_hhd_b50_curated.csv":{1474:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         1475:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         1478:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1480:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1481:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1484:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1488:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21",
                                         3597:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL255_EC50_hhd_b50_curated.csv":{193:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL3772_EC50_hhd_b50_curated.csv":{127:"O=C(Nc1ccc(-n2c(O)c3c(c2O)[C@H]2C=C[C@H]3C2)c(Cl)c1)c1ccccn1"},
    # For mhd_gpcr 5 files, 18 failed times, and 5 unique failed SMILES
    "CHEMBL256_bind_RBA_Ki_mhd_b50_curated.csv":{3397:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL226_bind_RBA_Ki_mhd_b50_curated.csv":{1428:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                 1429:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                 1432:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1434:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1435:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1438:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1442:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21",
                                                 3309:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL5850_agon_G-Ca_EC50_mhd_s50_curated.csv":{19:"COc1ccc(-c2ccc(C(=O)Nc3cccc(Cn4ncc(N5C[C@H]6C[C@H]5CN6C)c(Cl)c4=O)c3C)cc2)cn1"},
    "CHEMBL226_agon_G-cAMP_IC50_mhd_b50_curated.csv":{20:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                      21:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                      24:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      26:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      27:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      30:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      34:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21"},
    "CHEMBL251_bind_RBA_Ki_mhd_b50_curated.csv":{3859:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"}
    }

#===============================================================================
# find stereochemical siblings and add a column 'stereochemical_siblings'
#===============================================================================
def add_stereoSiblings_col(df: pd.DataFrame) -> pd.DataFrame:

    new_df = df.copy()
    smiles = df['canonical_smiles_by_Std'].tolist()
    pair_smis, pair_idx = find_stereochemical_siblings(smiles)
    # add a column 'stereoSiblings' to df, if the index is in the list, then True, else False
    new_df['stereoSiblings'] = new_df.index.isin(list(set(sum(pair_idx, []))))

    return new_df

#===============================================================================
# run_curation
#===============================================================================
def run_curation(ds_cat_level='hhd', input_path=CAT_HHD_OR_DIR, output_path= CURA_HHD_OR_DIR,
                targets_list: List[str]= OR_chemblids, effect='bind', assay='RBA', std_types=["Ki", 'IC50'], 
                ds_type= 'or'):
    """
    curate a series of datasets and get the stats for each dataset

    param:
    ----------
    ds_cat_level: str: The categorization level of dataset to curate. It could only be either 'hhd' or 'mhd', cannot be 'lhd'.
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
            if ds_cat_level == 'hhd':
                print(f"Processing hhd: {target}_{std_type}...")
                input_df_path = os.path.join(input_path, target, std_type, f'{target}_{std_type}_hhd_df.csv')

            elif ds_cat_level == 'mhd':
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

                        if ds_cat_level == 'hhd':
                            if len(curated_df) < 50:
                                ds_size_level = 's50'
                                filename = f'{target}_{std_type}_hhd_s50_curated.csv'
                            else:
                                ds_size_level = 'b50'
                                filename = f'{target}_{std_type}_hhd_b50_curated.csv'
                        elif ds_cat_level == 'mhd':
                            if len(curated_df) < 50:
                                ds_size_level = 's50'
                                filename = f'{target}_{effect}_{assay}_{std_type}_mhd_s50_curated.csv'
                            else:
                                ds_size_level = 'b50'
                                filename = f'{target}_{effect}_{assay}_{std_type}_mhd_b50_curated.csv'
                        
                        mkdirs(output_path)

                        # delete rows with SMILES strings that will fail to be embeded for 3D descriptor featurization
                        for f in failed_dict.keys():
                            if f == filename:
                                for key, value in failed_dict[f].items():
                                    idx_to_drop = curated_df[curated_df['canonical_smiles_by_Std'] == value].index
                                    print(f'index to drop is {idx_to_drop} for SMILES: {value}')
                                    curated_df = curated_df.drop(index=idx_to_drop)

                        # rename, add, delete columns
                        print(f'=================== \n renaming, adding, deleting columns for {filename} ...\n=====================')
                        curated_df = rename_add_delete_cols(curated_df)

                        # add a column 'stereochemical_siblings'
                        curated_df = add_stereoSiblings_col(curated_df)

                        # save the curated dataset
                        curated_df.to_csv(os.path.join(output_path, filename), index=False)
        
                        # get the stats and save them in a csv file
                        stats_file_path = os.path.join(CURA_DATA_DIR, f'cura_{ds_cat_level}_{ds_type}_stats.csv')

                        if not os.path.exists(stats_file_path): # don't use check_file_exists() and then remove the file if it exists
                            mkdirs(os.path.dirname(stats_file_path))
                            with open(stats_file_path, 'w') as f:
                                f.write('ds_cat_level,ds_type,ds_size_level,target_chembl_id,effect,assay,standard_type,assay_chembl_id,raw_size,curated_size,removed_size,max_num_atoms,max_mw,vs_threshold,vs_num_active,vs_num_inactive,vs_%_active,lo_threshold,lo_num_active,lo_num_inactive,lo_%_active, num_stereoSiblings\n')

                        with open(stats_file_path, 'a') as f:
                            # Print something for number check
                            curated_size = len(curated_df)
                            removed_size = raw_size - curated_size
                            max_num_atoms = curated_df['num_atoms'].max()
                            max_mw = curated_df['molecular_weight'].max()
                            vs_threshold = curated_df['vs_threshold'].unique()[0]
                            vs_num_active = sum(curated_df['vs_activity'])
                            vs_num_inactive = curated_size - vs_num_active

                            lo_threshold = curated_df['lo_threshold'].unique()[0]
                            lo_num_active = sum(curated_df['lo_activity'])
                            lo_num_inactive = curated_size - lo_num_active
                            
                            if curated_size == 0:
                                vs_percent_a = 0
                                lo_percent_a = 0
                            else:
                                vs_percent_a = round(vs_num_active / curated_size * 100, 2)
                                lo_percent_a = round(lo_num_active / curated_size * 100, 2)
                            
                            num_stereoSiblings = sum(curated_df['stereoSiblings'])

                            f.write(f"{ds_cat_level},{ds_type},{ds_size_level},{target},{effect},{assay},{std_type},{None},{raw_size},{curated_size},{removed_size},{max_num_atoms},{max_mw},{vs_threshold},{vs_num_active},{vs_num_inactive},{vs_percent_a},{lo_threshold},{lo_num_active},{lo_num_inactive},{lo_percent_a},{num_stereoSiblings}\n")

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
                                        ds_size_level = 's50'
                                        filename = f'{target}_{effect}_{assay}_{std_type}_{assay_chembl_id}_lhd_s50_curated.csv'
                                    else:
                                        ds_size_level = 'b50'
                                        filename = f'{target}_{effect}_{assay}_{std_type}_{assay_chembl_id}_lhd_b50_curated.csv'
                                    if ds_type == 'gpcr':
                                        output_lhd_path = CURA_LHD_GPCR_DIR
                                    elif ds_type == 'or':
                                        output_lhd_path = CURA_LHD_OR_DIR
                                    
                                    mkdirs(output_lhd_path)
                                    # rename, add, delete columns
                                    print(f'=================== \n renaming, adding, deleting columns for {filename} ...\n=====================')
                                    curated_lhd_df = rename_add_delete_cols(curated_lhd_df)

                                    # add a column 'stereochemical_siblings'
                                    curated_lhd_df = add_stereoSiblings_col(curated_lhd_df)
                                    curated_lhd_df.to_csv(os.path.join(output_lhd_path, filename), index=False)

                                    # get the stats and save them in a csv file
                                    lhd_stats_file_path = os.path.join(CURA_DATA_DIR, f'cura_lhd_{ds_type}_stats.csv')

                                    if not os.path.exists(lhd_stats_file_path): # don't use check_file_exists() and then remove the file if it exists
                                        mkdirs(os.path.dirname(lhd_stats_file_path))
                                        with open(lhd_stats_file_path, 'w') as f:
                                            f.write("ds_cat_level,ds_type,ds_size_level,target_chembl_id,effect,assay,standard_type,assay_chembl_id,raw_size,curated_size,removed_size,max_num_atoms,max_mw,vs_threshold,vs_num_active,vs_num_inactive,vs_%_active,lo_threshold,lo_num_active,lo_num_inactive,lo_%_active,num_stereoSiblings\n")

                                
                                    with open(lhd_stats_file_path, 'a') as f:
                                        # Print something for number check
                                        curated_sized = len(curated_lhd_df)
                                        removed_size = lhd_raw_size - curated_sized
                                        max_num_atoms = curated_lhd_df['num_atoms'].max()
                                        max_mw = curated_lhd_df['molecular_weight'].max()
                                        vs_threshold = curated_lhd_df['vs_threshold'].unique()[0]
                                        vs_num_active = sum(curated_lhd_df['vs_activity'])
                                        vs_num_inactive = curated_sized - vs_num_active
                                        lo_threshold = curated_lhd_df['lo_threshold'].unique()[0]
                                        lo_num_active = sum(curated_lhd_df['lo_activity'])
                                        lo_num_inactive = curated_sized - lo_num_active
                                        if curated_sized == 0:
                                                vs_percent_a = 0
                                                lo_percent_a = 0
                                        else:
                                                vs_percent_a = round(vs_num_active / curated_sized * 100, 2)
                                                lo_percent_a = round(lo_num_active / curated_sized * 100, 2)
                                        
                                        num_stereoSiblings = sum(curated_lhd_df['stereoSiblings'])

                                        f.write(f"lhd,{ds_type},{ds_size_level},{target},{effect},{assay},{std_type},{assay_chembl_id},{raw_size},{curated_sized},{removed_size},{max_num_atoms},{max_mw},{vs_threshold},{vs_num_active},{vs_num_inactive},{vs_percent_a},{lo_threshold},{lo_num_active},{lo_num_inactive},{lo_percent_a},{num_stereoSiblings}\n")
                    else:
                        print(f'No dataset at {lhd_path}')

            else:
                print(f'No dataset at {input_df_path}')

    print(f'====================\n')

#=============================================================================
# mhd_or_effect
#=============================================================================
def group_by_effect(ds_type='or', ds_cat_level='mhd'):
    """
    group dataset by target_chemblid and effect.
    """

    in_path_name = f"CURA_{ds_cat_level.upper()}_{ds_type.upper()}_DIR"
    in_path = globals()[in_path_name]

    curated_path = os.path.join(in_path)
    curated_files = [f for f in os.listdir(curated_path)]

    save_path = os.path.join(CURA_DATA_DIR, 'cura_mhd_or_effect')
    mkdirs(save_path)

    # collect unique target_chembl_id and effect combinations
    target_chembl_id_effects = list(
        {f.split('_')[0] + '_' + f.split('_')[1] for f in curated_files}
    )

    stats_file_path = os.path.join(CURA_DATA_DIR, f'cura_{ds_cat_level}_{ds_type}_effect_stats.csv')
    # open stats file and write header
    with open(stats_file_path, 'w') as f:
        f.write('ds_cat_level,ds_type,ds_size_level,target_chembl_id,effect,assay,standard_type,assay_chembl_id,curated_size,max_num_atoms,max_mw,vs_threshold,vs_num_active,vs_num_inactive,vs_%_active,lo_threshold,lo_num_active,lo_num_inactive,lo_%_active,num_stereoSiblings\n')

        for target_chembl_id_effect in target_chembl_id_effects:
            ds_cat_level = 'mhd_effect'
            ds_type = ds_type
            assay = 'None'
            standard_type = 'None'
            assay_chembl_id = 'None'

            target_chembl_id = target_chembl_id_effect.split('_')[0]
            effect = target_chembl_id_effect.split('_')[1]

            # combine files with the same target_chembl_id and effect
            concat_df = pd.DataFrame()
            for file in curated_files:
                if file.startswith(target_chembl_id_effect):
                    df = pd.read_csv(os.path.join(in_path, file))
                    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
                    concat_df = pd.concat([concat_df, df], ignore_index=True)
                    concat_df.reset_index(drop=True, inplace=True)
            
            # re-apply 'thresholds'
            concat_df.drop(columns=['vs_activity_comment','vs_activity', 'vs_threshold',
                                    'lo_activity_comment','lo_activity', 'lo_threshold',
                                    ], inplace=True)
            concat_df = apply_thresholds(concat_df, aim='vs', **DEFAULT_CLEANING)
            concat_df = apply_thresholds(concat_df, aim='lo', **DEFAULT_CLEANING)
            
            if len(concat_df) < 50:
                ds_size_level = 's50'
            else:
                ds_size_level = 'b50'

            # save combined df
            concat_df.to_csv(os.path.join(save_path, f'{target_chembl_id_effect}_mhd_{ds_size_level}_curated.csv'), index=False)
            print(f'The shape of combined df: {concat_df.shape}')

            # save stats
            curated_size = len(concat_df)
            max_num_atoms = concat_df['num_atoms'].max()
            max_mw= concat_df['molecular_weight'].max()

            vs_threshold = concat_df['vs_threshold'].unique()[0]
            vs_num_active = sum(concat_df['vs_activity'])
            vs_num_inactive = curated_size - vs_num_active

            lo_threshold = concat_df['lo_threshold'].unique()[0]
            lo_num_active = sum(concat_df['lo_activity'])
            lo_num_inactive = curated_size - lo_num_active

            vs_percent_a = round(vs_num_active / curated_size * 100, 2)
            lo_percent_a = round(lo_num_active / curated_size * 100, 2)

            num_stereoSiblings = sum(concat_df['stereoSiblings'])

            f.write(f"{ds_cat_level},{ds_type},{ds_size_level},{target_chembl_id},{effect},{assay},{standard_type},{assay_chembl_id},{curated_size},{max_num_atoms},{max_mw},{vs_threshold},{vs_num_active},{vs_num_inactive},{vs_percent_a},{lo_threshold},{lo_num_active},{lo_num_inactive},{lo_percent_a},{num_stereoSiblings}\n")