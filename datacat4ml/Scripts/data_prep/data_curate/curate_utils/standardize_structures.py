import math
import pandas as pd
import numpy as np
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

from datacat4ml.Scripts.data_prep.data_curate.curate_utils.standardizer import Standardizer

# ==============================  standardize_smiles ==============================

def standardize_smiles(x: pd.DataFrame, taut_canonicalization: bool = True) -> pd.DataFrame:

    """
    Standardization of a SMILES string.

    Uses the 'Standardizer' to perform sequence of cleaning operations on a SMILES string.
    """
    # reads in a SMILES, and returns it in canonicalized form.
    # can select whether or not to use tautomer canonicalization
    sm = Standardizer(canon_taut=taut_canonicalization)
    df = pd.DataFrame(x)

    def standardize_smile(x: str):
        try:
            mol = Chem.MolFromSmiles(x)
            mol_weight = round(MolWt(mol), 2)  # get molecular weight to do downstream filtering
            num_atoms = mol.GetNumAtoms()
            standardized_mol, _ = sm.standardize_mol(mol)
            return Chem.MolToSmiles(standardized_mol), mol_weight, num_atoms
        except Exception:
            # return a fail as None (downstream filtering)
            return None

    standard = df["canonical_smiles"].apply(lambda row: standardize_smile(row))
     # create new columns in the dataframe
    df["canonical_smiles_by_Std"] = standard.apply(lambda row: row[0])
    df["molecular_weight"] = standard.apply(lambda row: row[1])
    df["num_atoms"] = standard.apply(lambda row: row[2])

    return df

# ======================= convert the standard values to pIC50 ======================================
def log_standard_values(x: pd.Series) -> float:
    """Convert standard value to -log10([C]/nM)"""
    return -1 * math.log10(x["standard_value"] * 10 ** -9) 
    
# ====================== remove duplicate SMILES with different values ==============================
def remove_dupMol(df, std_smiles_col='canonical_smiles_by_Std', pvalue_col='pStandard_value') -> pd.DataFrame:
    
    """
    Remove duplicate molecules with high intra-molecule variability (>1 std),
    keep single-appearance smiles, and replace pStandard_value for multi-appearance
    molecules with the mean pStandard_value of that group.
    """

    #group stats per smiles
    df_group = df.groupby(std_smiles_col)[pvalue_col].agg(['mean', 'std'])

    # singletons: std is NaN (only one activity value)
    single_idx = df_group[df_group['std'].isna()].index
    single_df = df[df[std_smiles_col].isin(single_idx)].copy()
    print(f'single_df.shape: {single_df.shape}')

    # multi-apprearance smiles (std is not NaN)
    multi_idx = df_group[df_group['std'].notna()].index
    multi_df = df[df[std_smiles_col].isin(multi_idx)].copy()
    print(f'multi_df.shape: {multi_df.shape}')

    # remove multi-appearance with high std (>1)
    keep_multi_idx = df_group.loc[multi_idx].loc[lambda x: x['std'] <=1].index
    rmvD_df = multi_df[multi_df[std_smiles_col].isin(keep_multi_idx)].drop_duplicates(subset=std_smiles_col, keep='first').copy()
    print(f'rmvD_df.shape: {rmvD_df.shape}')

    # map the mean pStandard_value to each remaining multi-smiles row
    rmvD_df[pvalue_col] = rmvD_df[std_smiles_col].map(df_group['mean'])

    # combine single and multi-appearance smiles
    final_df = pd.concat([single_df, rmvD_df], axis=0).reset_index(drop=True)
    print(f'final_df.shape: {final_df.shape}')

    return final_df

# ======================= run standardizing pipeling ==============================
def standardize(
    x: pd.DataFrame,
    rmvD: int = 1,
    num_workers: int = 6,
    max_mol_weight: float = 900.0,
    **kwargs,
) -> pd.DataFrame:

    """
    Second stage cleaning; SMILES standardization.
    For rows where the column `pStandard_value` is available:

    1. desalt, canonicalise tautomers in SMILES
    2. remove > 900 Da molecular weight
    3. get log standard values (e.g. pKI)
    4. remove any repeats with conflicting measurements
    (conflicting is standard derivation of pKIs > 1.0)

    parameters:
    x: pd.DataFrame, the dataframe to clean
    num_workers: int, number of workers to use in parallel
    max_mol_weight: float, maximum molecular weight to keep
    rmvD: int, whether to remove duplicate molecules with conflicting values (1: yes, 0: no)

    returns:
    pd.DataFrame, cleaned dataframe

    """
    print(f'==> Standardizing structures ...')
    # first clean the SMILES
    def parallelize_standardization(df):
        data_splits = np.array_split(df, num_workers)
        with Pool(num_workers) as p:
            df = pd.concat(p.map(standardize_smiles, data_splits))
        return df

    df = parallelize_standardization(x)
    print (f'After standardizing the SMILES, the shape of the df: {df.shape}')

    # drop anything that the molecular weight is too high
    df = df.loc[df.molecular_weight <= max_mol_weight]
    print (f'After dropping the mols with MW > {max_mol_weight} , the shape of the df: {df.shape}')

    # get log standard values
    df.loc[(df["standard_units"] == "uM"), "standard_value"] *= 1000
    df.loc[(df["standard_units"] == "uM"), "standard_units"] = "nM" # Convert uM to nM first
    df["pStandard_value"] = df.apply(log_standard_values, axis=1)

    if rmvD == 1:
        print(f'==> Remove duplicate molecules ...')
        df = remove_dupMol(df)
        print(f'After removing the mols with multiple values, the missing values in pStandard_value: {df["pStandard_value"].isna().sum()}, the shape of the df: {df.shape}')

    df["max_num_atoms"] = df.num_atoms.max()
    df["max_molecular_weight"] = df.molecular_weight.max()

    if len(df) > 0:
        return df
    else:
        print(f"Empty dataframe after standardization")
        return pd.DataFrame()