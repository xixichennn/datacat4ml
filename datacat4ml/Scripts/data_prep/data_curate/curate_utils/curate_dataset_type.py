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
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.standardize_structures import standardize, remove_dupMol
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.apply_thresholds import apply_thresholds

from datacat4ml.Scripts.data_prep.data_split.intSplit_mldata import find_stereochemical_siblings

#===============================================================================
# delete wrong activity_ids documented by ChEMBL
#===============================================================================
wrong_activity_ids = [
    ##################### file: CHEMBL233_bind_RBA_Ki_CHEMBL3707592_lhd #####################
    #compound_chembl_id: CHEMBL3908275    3
    16348719, 16260203,
    #compound_chembl_id: CHEMBL3948231    3
    16280459, 16285964,
    #compound_chembl_id: CHEMBL3911529    3
    16338796, 16345420,
    #compound_chembl_id: CHEMBL3893577    3
    16329322, 16334546,
    #compound_chembl_id: CHEMBL3954498    3
    16323198, 16325206,
    #compound_chembl_id: CHEMBL3948400    3
    16353251, 16318808,
    #compound_chembl_id: CHEMBL3901699    3
    16311427, 16320230,
    #compound_chembl_id: CHEMBL3664540    3
    16320904, 16329066,
    #compound_chembl_id: CHEMBL3904663    3
    16311300, 16296171,
    #compound_chembl_id: CHEMBL3916122    3
    16343355, 16278872,
    #compound_chembl_id: CHEMBL4107573    3
    17607502, 17607533,
    #compound_chembl_id: CHEMBL3923372    3
    16330728, 16340093,
    #compound_chembl_id: CHEMBL3947184    3
    16262862, 16294741,
    #compound_chembl_id: CHEMBL3907149    3
    16265478, 16268318,
    #compound_chembl_id: CHEMBL3912511    3
    16309495, 16266350,
    #compound_chembl_id: CHEMBL3955920    3
    16329500, 16270670,
    #compound_chembl_id: CHEMBL3919397    3
    16274120, 16305421,
    #compound_chembl_id: CHEMBL3915488    2
    16301326, 
    #compound_chembl_id: CHEMBL3951177    2
    16272693,
    #compound_chembl_id: CHEMBL3946351    2
    16305554, 
    #compound_chembl_id: CHEMBL3900482    2
    16303895,
    #compound_chembl_id: CHEMBL3664541    2
    16346449, 16299045,
    #compound_chembl_id: CHEMBL3944046    2
    16278180,
    #compound_chembl_id: CHEMBL3892737    2
    16276817,
    #compound_chembl_id: CHEMBL3664525    2
    16293587, 
    #compound_chembl_id: CHEMBL3664526    2
    16260396,
    #compound_chembl_id: CHEMBL3982823    2
    16289441,
    #compound_chembl_id: CHEMBL3959791    2
    16356146,
    #compound_chembl_id: CHEMBL3952829    2
    16279037, 
    #compound_chembl_id: CHEMBL3911769    2             
    16289762
]

#=============================================================================
# remove rows with SMILESs that will fail to be embedded for 3D descriptors feature generation
#=============================================================================
failed_dict = { 
    # For hhd_gpcr, 10 files, 23 failed times, and 10 unique failed SMILES
    # file_basename: {row index in the file: SMILES}
    "CHEMBL5850_EC50_hhd":{26: "COc1ccc(-c2ccc(C(=O)Nc3cccc(Cn4ncc(N5C[C@H]6C[C@H]5CN6C)c(Cl)c4=O)c3C)cc2)cn1"},
    "CHEMBL256_EC50_hhd":{108:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL226_IC50_hhd":{132:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         133:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         136:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         138:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         139:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         142:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         146:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21"},
    "CHEMBL226_EC50_hhd":{196:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL256_Ki_hhd":{3439:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL251_Ki_hhd":{4280:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL251_EC50_hhd":{163:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL226_Ki_hhd":{1474:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         1475:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                         1478:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1480:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1481:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1484:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                         1488:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21",
                                         3597:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL255_EC50_hhd":{193:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL3772_EC50_hhd":{127:"O=C(Nc1ccc(-n2c(O)c3c(c2O)[C@H]2C=C[C@H]3C2)c(Cl)c1)c1ccccn1"},
    # For mhd_gpcr 5 files, 18 failed times, and 5 unique failed SMILES
    "CHEMBL256_bind_RBA_Ki_mhd":{3397:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL226_bind_RBA_Ki_mhd":{1428:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                 1429:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                 1432:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1434:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1435:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1438:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                 1442:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21",
                                                 3309:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"},
    "CHEMBL5850_agon_G-Ca_EC50_mhd":{19:"COc1ccc(-c2ccc(C(=O)Nc3cccc(Cn4ncc(N5C[C@H]6C[C@H]5CN6C)c(Cl)c4=O)c3C)cc2)cn1"},
    "CHEMBL226_agon_G-cAMP_IC50_mhd":{20:"CN(C)C(=S)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                      21:"CNC(=O)O[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@@H]5C[C@@H]4C4SC45)ncnc32)[C@H](O)[C@@H]1O",
                                                      24:"CCCNC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      26:"NC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      27:"C[Se]C[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      30:"CCSC[C@H]1O[C@@H](n2cnc3c(N[C@H]4C[C@H]5CC[C@H]4C5)ncnc32)[C@H](O)[C@@H]1O",
                                                      34:"O[C@@H]1[C@H](O)[C@@H](CF)O[C@H]1n1cnc2c(N[C@H]3C[C@H]4CC[C@H]3C4)ncnc21"},
    "CHEMBL251_bind_RBA_Ki_mhd":{3859:"CCn1nnc([C@H]2O[C@@H](n3cnc4c(N[C@H]5C[C@@H]6CC[C@@H]5C6)nc(Cl)nc43)[C@H](O)[C@@H]2O)n1"}
    }

#===============================================================================
# Based on literature checking, remove the wrong activity_ids assigned by 'data_categorize' step
#===============================================================================
ds_dupActids = {
    'CHEMBL233_antag_G-GTP_Ki_None_mhd':[1282351,1292631,1298256], # keep: after checking the orignal literature, keep these activity_ids in this dataset. 
    'CHEMBL233_bind_RBA_Ki_None_mhd':[1282351,1292631,1298256,], # remove.
    'CHEMBL233_bind_RBA_IC50_None_mhd':[1679560,1679561,1679562,1679563,1679564,1679496], #keep
    'CHEMBL233_agon_G-cAMP_IC50_None_mhd':[1679560,1679561,1679562,1679563,1679564,1679496] #remov.
}

ds_dupActids_2_remove = {
    'CHEMBL233_bind_RBA_Ki_None_mhd':[1282351,1292631,1298256], # assay_chembl_id: CHEMBL753396
    'CHEMBL233_agon_G-cAMP_IC50_None_mhd':[1679560,1679561,1679562,1679563,1679564,1679496] # assay_chembl_id: CHEMBL865906
}

#===============================================================================
# curation
#===============================================================================
DEFAULT_CLEANING = {
    "hard_only": False, # keep weak actives/inactives.
    "automate_threshold": True,
    "num_workers": cpu_count()}

def curate(df: pd.DataFrame, rmvD: int = 1) -> pd.DataFrame:
    """
    apply the curation pipeline to a dataframe

    param:
    -----
    df: pd.DataFrame: The input dataset to be curated, should be from cat_hhd or cat_mhd
    rmvD: whether to remove duplicate SMILES with different values. 1 or True means remove, 0 or False means keep.


    return:
    -----
    df: pd.DataFrame: The curated dataset and save it to the output_path
    """
    try:
        print(f"======= Curating dataset=======")
        df = select_assays(df, **DEFAULT_CLEANING)
        df = standardize(df, rmvD, **DEFAULT_CLEANING)
        df = apply_thresholds(df, aim='vs', **DEFAULT_CLEANING)
        df = apply_thresholds(df, aim='lo', **DEFAULT_CLEANING)

        print(f'Done curation.\n')

        return df

    except Exception as e:
        df = pd.DataFrame()
        logger.warning(f"Failed curating the dataset due to {e}")

        return df

#===============================================================================
# find stereochemical siblings and add a column 'stereochemical_siblings'
#===============================================================================
def add_stereoSiblings_col(df: pd.DataFrame) -> pd.DataFrame:

    if len(df) == 0:
        return df
    
    else:
        new_df = df.copy()
        smiles = df['canonical_smiles_by_Std'].tolist()
        pair_smis, pair_idx = find_stereochemical_siblings(smiles)
        # add a column 'stereoSiblings' to df, if the index is in the list, then True, else False
        new_df['stereoSiblings'] = new_df.index.isin(list(set(sum(pair_idx, []))))

        return new_df
    
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
    df.drop(columns=['max_num_atoms', 'max_molecular_weight',
                    'relationship_type', 'relationship_type_description'], inplace=True)

    return df

#===============================================================================
# run_curation
#===============================================================================
def run_curation(ds_cat_level='hhd', input_path=CAT_HHD_OR_DIR, output_path= CURA_HHD_OR_DIR,
                targets_list: List[str]= OR_chemblids, effect='bind', assay='RBA', standard_types=["Ki", 'IC50'], 
                ds_type= 'or', rmvD=1):
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
    standard_types: List[str]: The list of standard types to process. The element of the list could be either 'Ki' or 'IC50'
    ds_type: str: The type of dataset. It could be either 'or' or 'gpcr'
    rmvD: Whether to remove Duplicate SMILES with different values during standardization. 1 or True means remove, 0 or False means keep.
    

    output:
    ----------
    write the stats of the curated datasets to a csv file
    
    """

    for target_chembl_id in targets_list:
        print(f'Processing target_chembl_id {target_chembl_id} ...')

        for standard_type in standard_types:
            print(f'Processing {standard_type} ...')

            # ================= curate hhd or mhd ================
            print(f'========================= Curating hhd/mhd dataset===========================')
            if ds_cat_level == 'hhd':
                print(f"Processing hhd: {target_chembl_id}_{standard_type}...")
                input_df_path = os.path.join(input_path, target_chembl_id, standard_type, f'{target_chembl_id}_{standard_type}_hhd_df.csv')

            elif ds_cat_level == 'mhd':
                print(f"Processing mhd: {target_chembl_id}_{effect}_{assay}_{standard_type}...")
                input_df_path = os.path.join(input_path, target_chembl_id, effect, assay, standard_type, f'{target_chembl_id}_{effect}_{assay}_{standard_type}_mhd_df.csv')

            if os.path.exists(input_df_path):
                input_df = pd.read_csv(input_df_path)
                if len(input_df) == 0:
                    print(f'No data points in the dataset at {input_df_path}')
                else: 
                    raw_size = len(input_df)
                    print(f'The shape of the raw dataset is {input_df.shape}')
                    ########################### preprocess input_df ###########################
                    # fprefix
                    if ds_cat_level == 'hhd':
                        fprefix = f'{target_chembl_id}_None_None_{standard_type}_None_hhd'
                    elif ds_cat_level == 'mhd':
                        fprefix = f'{target_chembl_id}_{effect}_{assay}_{standard_type}_None_mhd'
                    
                    # remove rows with wrong activity_ids documented by ChEMBL
                    input_df = input_df[~input_df['activity_id'].isin(wrong_activity_ids)]

                    # remove rows with SMILES strings that will fail to be embeded for 3D descriptor featurization
                    for f in failed_dict.keys():
                        if f == fprefix:
                            for key, value in failed_dict[f].items():
                                idx_to_drop = input_df[input_df['canonical_smiles'] == value].index
                                print(f'index to drop is {idx_to_drop} for SMILES: {value}')
                                input_df = input_df.drop(index=idx_to_drop)

                    print(f'After dropping failed SMILES, the length of the input dataset is {len(input_df)}')

                    # remove rows with wrong activity_ids assigned by `data_categorize` step
                    if fprefix in ds_dupActids_2_remove.keys():
                        actids_to_remove = ds_dupActids_2_remove[fprefix]
                        idx_to_drop = input_df[input_df['activity_id'].isin(actids_to_remove)].index
                        input_df = input_df.drop(index=idx_to_drop)

                    ########################### run curation ###############################
                    # run the curation pipeline
                    curated_df = curate(input_df, rmvD)
                    print(f'After curation, the shape of the df: {curated_df.shape}')

                    ########################## post-curation process ##############################

                    if len(curated_df) == 0:
                        print(f'No data points left after curation for the dataset at {input_df_path}')
                    elif len(curated_df) > 0:

                        # add a column 'stereochemical_siblings'
                        curated_df = add_stereoSiblings_col(curated_df)
                        curated_df['effect'] = effect
                        curated_df['assay'] = assay

                        # rename, add, delete columns
                        curated_df = rename_add_delete_cols(curated_df)
                        
                        # curated_size = rmvS0_ds_size_level
                        curated_size = len(curated_df)
                        print(f'The final shape of the curated dataset is {curated_df.shape}')

                        if curated_size < 50:
                            rmvS0_ds_size_level = 's50'
                        else:
                            rmvS0_ds_size_level = 'b50'
                        
                        # rmvS1_ds_size_level
                        numStereo = sum(curated_df['stereoSiblings'])
                        rmvS1_ds_size = curated_size - numStereo
                        if rmvS1_ds_size < 50:
                            rmvS1_ds_size_level = 's50'
                        else:
                            rmvS1_ds_size_level = 'b50'

                        # save file
                        fname = fprefix + f'_{rmvS0_ds_size_level}_{rmvS1_ds_size_level}_curated.csv'

                        save_path = os.path.join(output_path, 'rmvD' + str(rmvD))
                        mkdirs(save_path)
                        curated_df.to_csv(os.path.join(save_path, fname), index=False)

                        #################################### stats #######################################
                        stats_file_path = os.path.join(CURA_DATA_DIR, f'cura_{ds_cat_level}_{ds_type}_rmvD{str(rmvD)}_stats.csv')

                        if not os.path.exists(stats_file_path): # don't use check_file_exists() and then remove the file if it exists
                            mkdirs(os.path.dirname(stats_file_path))
                            with open(stats_file_path, 'w') as f:
                                f.write('ds_cat_level,ds_type,' 
                                        'target_chembl_id,effect,assay,standard_type,assay_chembl_id,'
                                        'raw_size,removed_size,rmvS0_ds_size,rmvS0_ds_size_level,numStereo,rmvS1_ds_size,rmvS1_ds_size_level,'
                                        'max_num_atoms,max_mw,'
                                        'vs_threshold,rmvS0_vs_num_active,rmvS0_vs_num_inactive,rmvS0_vs_percent_a, rmvS1_vs_num_active,rmvS1_vs_num_inactive,rmvS1_vs_percent_a,'
                                        'lo_threshold,rmvS0_lo_num_active,rmvS0_lo_num_inactive,rmvS0_lo_percent_a,rmvS1_lo_num_active,rmvS1_lo_num_inactive,rmvS1_lo_percent_a\n'
                                        )

                        with open(stats_file_path, 'a') as f:
                            # Print something for number check
                            removed_size = raw_size - curated_size
                            max_num_atoms = curated_df['num_atoms'].max()
                            max_mw = curated_df['molecular_weight'].max()

                            vs_threshold = curated_df['vs_threshold'].unique()[0]
                            rmvS0_vs_num_active = sum(curated_df['vs_activity'])
                            rmvS0_vs_num_inactive = curated_size - rmvS0_vs_num_active
                            rmvS1_vs_num_active = sum(curated_df[curated_df['stereoSiblings'] == False]['vs_activity'])
                            rmvS1_vs_num_inactive = rmvS1_ds_size - rmvS1_vs_num_active

                            lo_threshold = curated_df['lo_threshold'].unique()[0]
                            rmvS0_lo_num_active = sum(curated_df['lo_activity'])
                            rmvS0_lo_num_inactive = curated_size - rmvS0_lo_num_active
                            rmvS1_lo_num_active = sum(curated_df[curated_df['stereoSiblings'] == False]['lo_activity'])
                            rmvS1_lo_num_inactive = rmvS1_ds_size - rmvS1_lo_num_active
                            
                            if curated_size == 0 or rmvS1_ds_size == 0:
                                rmvS0_vs_percent_a = 0
                                rmvS0_lo_percent_a = 0
                                rmvS1_vs_percent_a = 0
                                rmvS1_lo_percent_a = 0
                            else:
                                rmvS0_vs_percent_a = round(rmvS0_vs_num_active / curated_size * 100, 2)
                                rmvS0_lo_percent_a = round(rmvS0_lo_num_active / curated_size * 100, 2)
                                rmvS1_vs_percent_a = round(rmvS1_vs_num_active / rmvS1_ds_size * 100, 2)
                                rmvS1_lo_percent_a = round(rmvS1_lo_num_active / rmvS1_ds_size * 100, 2)

                            f.write(f"{ds_cat_level},{ds_type},"
                                    f"{target_chembl_id},{effect},{assay},{standard_type},{None},"
                                    f"{raw_size},{removed_size},{curated_size},{rmvS0_ds_size_level},{numStereo},{rmvS1_ds_size},{rmvS1_ds_size_level},"
                                    f"{max_num_atoms},{max_mw},"
                                    f"{vs_threshold},{rmvS0_vs_num_active},{rmvS0_vs_num_inactive},{rmvS0_vs_percent_a},{rmvS1_vs_num_active},{rmvS1_vs_num_inactive},{rmvS1_vs_percent_a},"
                                    f"{lo_threshold},{rmvS0_lo_num_active},{rmvS0_lo_num_inactive},{rmvS0_lo_percent_a},{rmvS1_lo_num_active},{rmvS1_lo_num_inactive},{rmvS1_lo_percent_a}\n")

                    # ================= curate lhd ================
                    print(f'======= Curating lhd dataset=======')
                    lhd_path = os.path.join(os.path.dirname(input_df_path), 'lhd')

                    if os.path.exists(lhd_path):
                        # collect all the assay_chembl_ids in the lhd_path
                        lhd_files = [f for f in os.listdir(lhd_path)]
                        lhd_assay_chembl_ids = list(set([f.split('_')[4] for f in lhd_files]))

                        for assay_chembl_id in lhd_assay_chembl_ids:
                            print(f'Processing lhd: {target_chembl_id}_{effect}_{assay}_{standard_type}_{assay_chembl_id}...')
                            lhd_df_path = os.path.join(lhd_path, f'{target_chembl_id}_{effect}_{assay}_{standard_type}_{assay_chembl_id}_lhd_df.csv')
                            lhd_df = pd.read_csv(lhd_df_path)
                            if len(lhd_df) == 0:
                                print(f'No data points in the dataset at {lhd_df_path}')
                            else:
                                lhd_raw_size = len(lhd_df)
                                print(f'The shape of the raw lhd dataset is {lhd_df.shape}')

                                ############################# preprocess lhd_df ###########################
                                # remove rows with wrong activity_ids documented by ChEMBL
                                lhd_df = lhd_df[~lhd_df['activity_id'].isin(wrong_activity_ids)] # necesary due to lhd_df is read form categorized data, not from the above curated-hhd/mhd data
                                
                                ############################# run curation ###########################
                                # run the curation pipeline
                                curated_lhd_df = curate(lhd_df, rmvD)

                                ############################ post-curation process###############################
                                if len(curated_lhd_df) == 0:
                                    print(f'No data points left after curation for the dataset at {lhd_df_path}')
                                elif len(curated_lhd_df) > 0:
                                    # add a column 'stereochemical_siblings'
                                    curated_lhd_df = add_stereoSiblings_col(curated_lhd_df)

                                    curated_lhd_df['effect'] = effect
                                    curated_lhd_df['assay'] = assay

                                    # rename, add, delete columns
                                    curated_lhd_df = rename_add_delete_cols(curated_lhd_df)

                                    # curated_lhd_size = rmvS0_ds_size_level
                                    curated_lhd_size = len(curated_lhd_df)
                                    print(f'After curation, the shape of the curated lhd dataset is {curated_lhd_df.shape}')

                                    if curated_lhd_size < 50:
                                        rmvS0_ds_size_level = 's50'
                                    else:
                                        rmvS0_ds_size_level = 'b50'

                                    # rmvS1_ds_size_level
                                    numStereo = sum(curated_lhd_df['stereoSiblings'])
                                    rmvS1_ds_size = curated_lhd_size - numStereo
                                    if rmvS1_ds_size < 50:
                                        rmvS1_ds_size_level = 's50'
                                    else:
                                        rmvS1_ds_size_level = 'b50'

                                    # save file
                                    fname = f'{target_chembl_id}_{effect}_{assay}_{standard_type}_{assay_chembl_id}_lhd_{rmvS0_ds_size_level}_{rmvS1_ds_size_level}_curated.csv'

                                    if ds_type == 'gpcr':
                                        output_lhd_path = os.path.join(CURA_LHD_GPCR_DIR, 'rmvD' + str(rmvD))
                                    elif ds_type == 'or':
                                        output_lhd_path = os.path.join(CURA_LHD_OR_DIR, 'rmvD' + str(rmvD))

                                    mkdirs(output_lhd_path)
                                    curated_lhd_df.to_csv(os.path.join(output_lhd_path, fname), index=False)
                                    
                                    #################################### stats #######################################
                                    lhd_stats_file_path = os.path.join(CURA_DATA_DIR, f'cura_lhd_{ds_type}_rmvD{str(rmvD)}_stats.csv')

                                    if not os.path.exists(lhd_stats_file_path): # don't use check_file_exists() and then remove the file if it exists
                                        mkdirs(os.path.dirname(lhd_stats_file_path))
                                        with open(lhd_stats_file_path, 'w') as f:
                                            f.write('ds_cat_level,ds_type,' 
                                                    'target_chembl_id,effect,assay,standard_type,assay_chembl_id,'
                                                    'raw_size,removed_size,rmvS0_ds_size,rmvS0_ds_size_level,numStereo,rmvS1_ds_size,rmvS1_ds_size_level,'
                                                    'max_num_atoms,max_mw,'
                                                    'vs_threshold,rmvS0_vs_num_active,rmvS0_vs_num_inactive,rmvS0_vs_percent_a, rmvS1_vs_num_active,rmvS1_vs_num_inactive,rmvS1_vs_percent_a,'
                                                    'lo_threshold,rmvS0_lo_num_active,rmvS0_lo_num_inactive,rmvS0_lo_percent_a,rmvS1_lo_num_active,rmvS1_lo_num_inactive,rmvS1_lo_percent_a\n'
                                                    )

                                    with open(lhd_stats_file_path, 'a') as f:
                                        # Print something for number check
                                        removed_size = lhd_raw_size - curated_lhd_size
                                        max_num_atoms = curated_lhd_df['num_atoms'].max()
                                        max_mw = curated_lhd_df['molecular_weight'].max()

                                        vs_threshold = curated_lhd_df['vs_threshold'].unique()[0]
                                        rmvS0_vs_num_active = sum(curated_lhd_df['vs_activity'])
                                        rmvS0_vs_num_inactive = curated_lhd_size - rmvS0_vs_num_active
                                        rmvS1_vs_num_active = sum(curated_lhd_df[curated_lhd_df['stereoSiblings'] == False]['vs_activity'])
                                        rmvS1_vs_num_inactive = rmvS1_ds_size - rmvS1_vs_num_active

                                        lo_threshold = curated_lhd_df['lo_threshold'].unique()[0]
                                        rmvS0_lo_num_active = sum(curated_lhd_df['lo_activity'])
                                        rmvS0_lo_num_inactive = curated_lhd_size - rmvS0_lo_num_active
                                        rmvS1_lo_num_active = sum(curated_lhd_df[curated_lhd_df['stereoSiblings'] == False]['lo_activity'])
                                        rmvS1_lo_num_inactive = rmvS1_ds_size - rmvS1_lo_num_active

                                        if curated_lhd_size == 0 or rmvS1_ds_size == 0:
                                                rmvS0_vs_percent_a = 0
                                                rmvS0_lo_percent_a = 0
                                                rmvS1_vs_percent_a = 0
                                                rmvS1_lo_percent_a = 0
                                        else:
                                                rmvS0_vs_percent_a = round(rmvS0_vs_num_active / curated_lhd_size * 100, 2)
                                                rmvS0_lo_percent_a = round(rmvS0_lo_num_active / curated_lhd_size * 100, 2)
                                                rmvS1_vs_percent_a = round(rmvS1_vs_num_active / rmvS1_ds_size * 100, 2)
                                                rmvS1_lo_percent_a = round(rmvS1_lo_num_active / rmvS1_ds_size * 100, 2)

                                        f.write(f"lhd,{ds_type},"
                                                f"{target_chembl_id},{effect},{assay},{standard_type},{assay_chembl_id},"
                                                f"{raw_size},{removed_size},{curated_size},{rmvS0_ds_size_level},{numStereo},{rmvS1_ds_size},{rmvS1_ds_size_level},"
                                                f"{max_num_atoms},{max_mw},"
                                                f"{vs_threshold},{rmvS0_vs_num_active},{rmvS0_vs_num_inactive},{rmvS0_vs_percent_a},{rmvS1_vs_num_active},{rmvS1_vs_num_inactive},{rmvS1_vs_percent_a},"
                                                f"{lo_threshold},{rmvS0_lo_num_active},{rmvS0_lo_num_inactive},{rmvS0_lo_percent_a},{rmvS1_lo_num_active},{rmvS1_lo_num_inactive},{rmvS1_lo_percent_a}\n")
                    else:
                        print(f'No dataset at {lhd_path}')

            else:
                print(f'No dataset at {input_df_path}')

    print(f'====================\n')

#=============================================================================
# mhd-effect_or
#=============================================================================
def group_by_effect(ds_type='or', ds_cat_level='mhd', rmvD=1):
    """
    group dataset by target_chemblid and effect.
    """
    curated_path = os.path.join(CURA_MHD_OR_DIR, 'rmvD0') # start from rmvD0 curated files, because if rmvD=1, the combined files will be removed again later.
    curated_files = [f for f in os.listdir(curated_path)]

    save_path = os.path.join(CURA_DATA_DIR, 'cura_mhd-effect_or', 'rmvD' + str(rmvD))
    mkdirs(save_path)

    # collect unique target_chembl_id and effect combinations
    target_effects = list(
        {f.split('_')[0] + '_' + f.split('_')[1] for f in curated_files}
        )  

    stats_file_path = os.path.join(CURA_DATA_DIR, f'cura_{ds_cat_level}-effect_{ds_type}_rmvD{str(rmvD)}_stats.csv')
                                                  
    # open stats file and write header
    with open(stats_file_path, 'w') as f:
        f.write('ds_cat_level,ds_type,' 
                'target_chembl_id,effect,assay,standard_type,assay_chembl_id,'
                'raw_size,removed_size,rmvS0_ds_size,rmvS0_ds_size_level,numStereo,rmvS1_ds_size,rmvS1_ds_size_level,'
                'max_num_atoms,max_mw,'
                'vs_threshold,rmvS0_vs_num_active,rmvS0_vs_num_inactive,rmvS0_vs_percent_a, rmvS1_vs_num_active,rmvS1_vs_num_inactive,rmvS1_vs_percent_a,'
                'lo_threshold,rmvS0_lo_num_active,rmvS0_lo_num_inactive,rmvS0_lo_percent_a,rmvS1_lo_num_active,rmvS1_lo_num_inactive,rmvS1_lo_percent_a\n'
                )

        for target_effect in target_effects:
            print(f'=========== Processing {target_effect} =============')

            ds_cat_level = 'mhd-effect'
            ds_type = ds_type
            assay = 'None'
            standard_type = 'None'
            assay_chembl_id = 'None'
            raw_size = 0
            removed_size = 0

            target_chembl_id = target_effect.split('_')[0]
            effect = target_effect.split('_')[1]

            # combine files with the same target_chembl_id and effect
            concat_df = pd.DataFrame()
            for file in curated_files:
                if file.startswith(target_effect):
                    df = pd.read_csv(os.path.join(curated_path, file))
                    concat_df = pd.concat([concat_df, df], ignore_index=True)
                    concat_df.reset_index(drop=True, inplace=True)
            print(f'After combining files, the shape of the combined df is {concat_df.shape}')
            
            # remove duplicate SMILES with different values if rmvD is 1 or True
            if rmvD == 1:
                print(f'==> Remove dupMols ...')
                concat_df = remove_dupMol(concat_df)
                print(f'After removing duplicate molecules with different values, the shape of the combined df is {concat_df.shape}')

            # re-apply 'thresholds'
            concat_df.drop(columns=['vs_activity_comment','vs_activity', 'vs_threshold',
                                    'lo_activity_comment','lo_activity', 'lo_threshold',
                                    ], inplace=True)
            concat_df = apply_thresholds(concat_df, aim='vs', **DEFAULT_CLEANING)
            concat_df = apply_thresholds(concat_df, aim='lo', **DEFAULT_CLEANING)

            # curated_size = rmvS0_ds_size_level
            curated_size = len(concat_df)
            if curated_size < 50:
                rmvS0_ds_size_level = 's50'
            else:
                rmvS0_ds_size_level = 'b50'

            # rmvS1_ds_size_level
            numStereo = sum(concat_df['stereoSiblings'])
            rmvS1_ds_size = curated_size - numStereo
            if rmvS1_ds_size < 50:
                rmvS1_ds_size_level = 's50'
            else:
                rmvS1_ds_size_level = 'b50'

            # save combined df
            concat_df.to_csv(os.path.join(save_path, f'{target_effect}_None_None_None_mhd-effect_{rmvS0_ds_size_level}_{rmvS1_ds_size_level}_curated.csv'), index=False)
            print(f'The shape of combined df: {concat_df.shape}')

            # save stats
            max_num_atoms = concat_df['num_atoms'].max()
            max_mw= concat_df['molecular_weight'].max()

            vs_threshold = concat_df['vs_threshold'].unique()[0]
            rmvS0_vs_num_active = sum(concat_df['vs_activity'])
            rmvS0_vs_num_inactive = curated_size - rmvS0_vs_num_active
            rmvS1_vs_num_active = sum(concat_df[concat_df['stereoSiblings'] == False]['vs_activity'])
            rmvS1_vs_num_inactive = rmvS1_ds_size - rmvS1_vs_num_active

            lo_threshold = concat_df['lo_threshold'].unique()[0]
            rmvS0_lo_num_active = sum(concat_df['lo_activity'])
            rmvS0_lo_num_inactive = curated_size - rmvS0_lo_num_active
            rmvS1_lo_num_active = sum(concat_df[concat_df['stereoSiblings'] == False]['lo_activity'])
            rmvS1_lo_num_inactive = rmvS1_ds_size - rmvS1_lo_num_active

            rmvS0_vs_percent_a = round(rmvS0_vs_num_active / curated_size * 100, 2)
            rmvS0_lo_percent_a = round(rmvS0_lo_num_active / curated_size * 100, 2)
            rmvS1_vs_percent_a = round(rmvS1_vs_num_active / rmvS1_ds_size * 100, 2)
            rmvS1_lo_percent_a = round(rmvS1_lo_num_active / rmvS1_ds_size * 100, 2)

            f.write(f"{ds_cat_level},{ds_type},"
                    f"{target_chembl_id},{effect},{assay},{standard_type},{assay_chembl_id},"
                    f"{raw_size},{removed_size},{curated_size},{rmvS0_ds_size_level},{numStereo},{rmvS1_ds_size},{rmvS1_ds_size_level},"
                    f"{max_num_atoms},{max_mw},"
                    f"{vs_threshold},{rmvS0_vs_num_active},{rmvS0_vs_num_inactive},{rmvS0_vs_percent_a},{rmvS1_vs_num_active},{rmvS1_vs_num_inactive},{rmvS1_vs_percent_a},"
                    f"{lo_threshold},{rmvS0_lo_num_active},{rmvS0_lo_num_inactive},{rmvS0_lo_percent_a},{rmvS1_lo_num_active},{rmvS1_lo_num_inactive},{rmvS1_lo_percent_a}\n")