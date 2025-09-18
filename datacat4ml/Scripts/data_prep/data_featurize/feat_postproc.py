"""
This script removes SMILES lines that cannot be embedded into 3D descriptors from the featurized pickle file.
"""

import os
import pandas as pd
from typing import List, Dict
from datacat4ml.const import FEAT_DATA_DIR
from datacat4ml.const import FEAT_HHD_GPCR_DIR, FEAT_MHD_GPCR_DIR, FEAT_LHD_GPCR_DIR, FEAT_HHD_OR_DIR, FEAT_MHD_OR_DIR, FEAT_LHD_OR_DIR

from datacat4ml.utils import mkdirs
from datacat4ml.Scripts.data_prep.data_featurize.feat_smi_list import Cura_Feat_Dic

#=============================================================================
# merge_feat_pkl
#=============================================================================
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

def merge_feat_pkls(in_dir):
    """
    After featurization by different descriptors, 
    get the descriptor column from each pickle file and append it to the original curated dataframe,
    and finally save the merged dataframe to the corresponding "all" subfolder in FEAT_*_DIR.

    Params:
    ------
    in_dir: str
        The input directory contains the original curated files. e.g. FEAT_HHD_GPCR_DIR, FEAT_MHD_OR_DIR

    Returns:
    -------
    None
    """
    descriptors = ['ECFP4', 'ECFP6', 'MACCS', 'RDKITFP', 'PHARM2D', 'ERG', 
                   'PHYSICOCHEM', 
                   'SHAPE3D', 'AUTOCORR3D', 'RDF', 'MORSE', 'WHIM', 'GETAWAY']
    
    # GET the base filenames from the original curated csv files
    files = os.listdir(in_dir) # in_dir should be the subfolder in CURA_DATA_DIR
    original_files = [file for file in files if file.endswith('_curated.csv')]

    base_names = []
    for original_file in original_files:
        base_name = "_".join(os.path.basename(original_file).split("_")[:-1]) # drop '_curated.csv'
        base_names.append(base_name)

    # GET the descriptor column from each pickle file
    for base_name in base_names:
        df = pd.read_csv(os.path.join(in_dir, f"{base_name}_curated.csv"))
        # if df contains column 'Unnamed: 0', drop it
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')

        pkl_dir = Cura_Feat_Dic[in_dir]
        for descriptor in descriptors:
            pkl_file = os.path.join(pkl_dir, f"{base_name}_{descriptor}.pkl")
            pkl_df = pd.read_pickle(pkl_file)
            pkl_df = pkl_df.drop(columns=['Unnamed: 0'], errors='ignore')
            # append the descriptor column to the original dataframe
            df[descriptor] = pkl_df[descriptor].values
        
        ############################ rename, add, delete columns ############################
        print(f'=================== \n renaming, adding, deleting columns for {base_name} ...\n=====================')
        # add the following columns:
        df['effect_description'] = df['effect'].map(effect_dict)
        df['assay_keywords_description'] = df['assay'].map(assay_dict)

        # rename the following columns:
        df.rename(columns={
            'assay_desc': 'assay_description',
            'assay_type_desc': 'assay_type_description',
            'relationship_type_desc': 'relationship_type_description',
            'confidence_score_desc': 'confidence_score_description'}, 
            inplace=True)
        
        # delete the following columns
        df.drop(columns=['target', 'std_type', 'max_num_atoms', 'max_molecular_weight'], inplace=True)
        #####################################################################################

        # save the merged dataframe to out_dir
        out_dir = os.path.join(pkl_dir, "all")
        mkdirs(out_dir)
        merged_pkl_file = os.path.join(out_dir, f"{base_name}_featurized.pkl")
        df.to_pickle(merged_pkl_file)

#=============================================================================
# delete_failed_smi
#=============================================================================
hhd_gpcr_failed = { # 10 files, 23 failed times, and 10 unique failed SMILES
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
    "CHEMBL3772_EC50_hhd_b50_curated.csv":{127:"O=C(Nc1ccc(-n2c(O)c3c(c2O)[C@H]2C=C[C@H]3C2)c(Cl)c1)c1ccccn1"}
    }

mhd_gpcr_failed ={# 5 files, 18 failed times, and 5 unique failed SMILES
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

def delete_failed_rows(failed_dict: Dict[str, Dict[int, str]] = hhd_gpcr_failed,
                       failed_path: str = FEAT_HHD_GPCR_DIR):
    
    """ Delete the rows with failed SMILES in the featurized dataframes
    and replace the original pickle files.
    """
    print(f'Process the failed files in {failed_path} ...')

    failed_dfs = {}
    failed_new_dfs = {}

    for f in failed_dict.keys():
        f_name = f'{f.rsplit("_curated.csv", 1)[0]}_featurized.pkl'
        df = pd.read_pickle(os.path.join(failed_path, 'all', f'{f.rsplit("_curated.csv", 1)[0]}_featurized.pkl'))
        failed_dfs[f_name] = df
        print(f'The shape of {f_name}: {df.shape}')

        for key, value in failed_dict[f].items():
            idx_to_drop = df[df['canonical_smiles_by_Std'] == value].index
            print(f'index to drop is {idx_to_drop} for SMILES: {value}')
            df = df.drop(index=idx_to_drop)
        failed_new_dfs[f_name] = df
        # save the new dataframe
        df.to_pickle(os.path.join(failed_path, 'all', f_name))

        print(f'The shape of {f_name} after dropping rows with failed SMILES: {df.shape}')
        print('=================')
    
    return failed_dfs, failed_new_dfs

#=============================================================================
# get_feat_stats
#=============================================================================
def get_feat_stats(in_path: str = FEAT_HHD_OR_DIR, ds_cat_level: str = 'hhd', ds_type: str = 'or'):
    
    """Get the statistics of the featurized datasets and save them to a csv file."""
    
    print(f'Processing: {in_path}')
    feat_path = os.path.join(in_path, 'all')
    feat_files = [f for f in os.listdir(feat_path)]

    for f in feat_files:
        print(f'f is {f}')

        if ds_cat_level == 'hhd':
            effect, assay, assay_chembl_id = None, None, None
            target, standard_type, ds_cat_level, ds_size_level = f.split('_')[:4]
        elif ds_cat_level == 'mhd':
            assay_chembl_id = None
            target, effect, assay, standard_type, ds_cat_level, ds_size_level = f.split('_')[:6]
        elif ds_cat_level == 'lhd':
            target, effect, assay, standard_type, assay_chembl_id, ds_cat_level, ds_size_level = f.split('_')[:7]

        df = pd.read_pickle(os.path.join(feat_path, f))

        ################################ write the stats origianlly from CURA ################################
        max_num_atoms = df['num_atoms'].max()
        max_mw= df['molecule_weight'].max()

        stats_file_path = os.path.join(FEAT_DATA_DIR, f'feat_{ds_cat_level}_{ds_type}_stats.csv')

        if not os.path.exists(stats_file_path):
            with open(stats_file_path, 'w') as f:
                f.write('ds_cat_level,ds_type,ds_size_level,target,effect,assay,standard_type,assay_chembl_id,feated_size,max_num_atoms,max_mw\n')

        with open(stats_file_path, 'a') as f:
            f.write(f'{ds_cat_level},{ds_type},{ds_size_level},{target},{effect},{assay},{standard_type},{assay_chembl_id},{len(df)},{max_num_atoms},{max_mw}\n')


#=============================================================================
# main
#=============================================================================
if __name__ == "__main__":
    #============ merge_feat_pkls =================
    keys_list = list(Cura_Feat_Dic.keys())

    for in_dir in keys_list:
        print(f'Processing {in_dir}...')
        merge_feat_pkls(in_dir)
    
    #============ delete_failed_smi =================
    hhd_gpcr_failed_dfs, hhd_gpcr_failed_new_dfs = delete_failed_rows(failed_dict=hhd_gpcr_failed,
                                                                   failed_path=FEAT_HHD_GPCR_DIR)
    mhd_gpcr_failed_dfs, mhd_gpcr_failed_new_dfs = delete_failed_rows(failed_dict=mhd_gpcr_failed,
                                                                   failed_path=FEAT_MHD_GPCR_DIR)
    
    #============ get_feat_stats =================
    get_feat_stats(in_path=FEAT_HHD_OR_DIR, ds_cat_level='hhd', ds_type='or')
    get_feat_stats(in_path=FEAT_HHD_GPCR_DIR, ds_cat_level='hhd', ds_type='gpcr')
    get_feat_stats(in_path=FEAT_MHD_OR_DIR, ds_cat_level='mhd', ds_type='or')
    get_feat_stats(in_path=FEAT_MHD_GPCR_DIR, ds_cat_level='mhd', ds_type='gpcr')
    get_feat_stats(in_path=FEAT_LHD_OR_DIR, ds_cat_level='lhd', ds_type='or')
    get_feat_stats(in_path=FEAT_LHD_GPCR_DIR, ds_cat_level='lhd', ds_type='gpcr')