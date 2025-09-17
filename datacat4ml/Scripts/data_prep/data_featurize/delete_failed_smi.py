"""
This script removes SMILES lines that cannot be embedded into 3D descriptors from the featurized pickle file.
"""

import os
import pandas as pd
from typing import List, Dict
from datacat4ml.const import FEAT_HHD_GPCR_DIR, FEAT_MHD_GPCR_DIR

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

#=============================================================================
# Delete the rows with these SMILES from the featurized pickle files
#=============================================================================
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

if __name__ == "__main__":
    hhd_gpcr_failed_dfs, hhd_gpcr_failed_new_dfs = delete_failed_rows(failed_dict=hhd_gpcr_failed,
                                                                   failed_path=FEAT_HHD_GPCR_DIR)
    mhd_gpcr_failed_dfs, mhd_gpcr_failed_new_dfs = delete_failed_rows(failed_dict=mhd_gpcr_failed,
                                                                   failed_path=FEAT_MHD_GPCR_DIR)
    

# run command:
# conda activate datacat4ml
# python3 delete_failed_smi.py