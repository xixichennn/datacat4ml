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
            mol_weight = MolWt(mol)  # get molecular weight to do downstream filtering
            num_atoms = mol.GetNumAtoms()
            standardized_mol, _ = sm.standardize_mol(mol)
            return Chem.MolToSmiles(standardized_mol), mol_weight, num_atoms
        except Exception:
            # return a fail as None (downstream filtering)
            return None

    standard = df["canonical_smiles"].apply(lambda row: standardize_smile(row))
    df["canonical_smiles_by_Std"] = standard.apply(lambda row: row[0])
    df["molecular_weight"] = standard.apply(lambda row: row[1])
    df["num_atoms"] = standard.apply(lambda row: row[2])

    return df

# ======================= convert the standard values to pIC50 ======================================
def log_standard_values(x: pd.Series) -> float:
    """Convert standard value to -log10([C]/nM)"""
    if x["standard_value"] < 1e-13 or np.isnan(x["standard_value"]):
        return float("NaN")
    else:
        return -1 * math.log10(x["standard_value"] * 10 ** -9) 
    
# ====================== remove duplicate SMILES with different values ==============================
def remove_dup_mols(df, std_smiles_col='canonical_smiles_by_Std', pvalue_col='pStandard_value') -> pd.DataFrame:

    """
    Entries with multiple  annotations were included once with the arithmetic mean when the standard deviation of pstardard_value annotations was within 1 log unit;
    Otherwise, the entry was excluded.

    param: df: pd.DataFrame: The dataframe to remove duplicates from
    
    """
    # group by 'canonical_smiles_by_Std' and calculate the mean and standard deviation of 'pstandard_value'
    df_group = df.groupby(std_smiles_col)[pvalue_col].agg(['mean', 'std'])
    # find where the standard deviation is greater than 1, and drop these rows. Then keep the first row of the rows with the same 'canonical_smiles_by_Std'
    df = df[~df[std_smiles_col].isin(df_group[df_group['std'] > 1].index)].drop_duplicates(subset=std_smiles_col, keep='first').copy()
    # map the mean of 'pstandard_value' to the 'molecule_chembl_id' in the original dataframe
    df[pvalue_col] = df[std_smiles_col].map(df_group['mean'])
    # reset the index
    df = df.reset_index(drop=True)

    return df

# ======================= run standardizing pipeling ==============================
def standardize(
    x: pd.DataFrame,
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

    returns:
    pd.DataFrame, cleaned dataframe

    """

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

    # get log standard values -- need to convert uM first
    df.loc[(df["standard_units"] == "uM"), "standard_value"] *= 1000
    df.loc[(df["standard_units"] == "uM"), "standard_units"] = "nM"
    df["pStandard_value"] = df.apply(log_standard_values, axis=1)

    # remove duplicate
    # first need to just keep one of the duplicates if smiles and value are *exactly* the same
    df = df.drop_duplicates(subset=["canonical_smiles_by_Std", "standard_value"], keep="first")
    print (f'After dropping the duplicate combinations of (smiles, value) , the shape of the df:{df.shape}')
    
    # now drop duplicates if the smiles are the same and the values are outside of a threshold
    # Entries with multiple  annotations were included once when the standard deviation of pstardard_value annotations was within 1 log unit;
    # Otherwise, the entry was excluded
    df = remove_dup_mols(df, std_smiles_col='canonical_smiles_by_Std', pvalue_col='pStandard_value')
    print (f'After removing the mols with multiple values, the shape of the df:{df.shape}')

    df["max_num_atoms"] = df.num_atoms.max()
    df["max_molecular_weight"] = df.molecular_weight.max()

    return df

#def standardize_novalue(
#    x: pd.DataFrame,
#    num_workers: int = 6,
#    max_mol_weight: float = 900.0,
#    **kwargs,
#) -> pd.DataFrame:
#
#    """
#    Second stage cleaning; SMILES standardization.
#    For rows where the column `pStandard_value` is None:
#
#    1. desalt, canonicalise tautomers in SMILES
#    2. remove > 900 Da molecular weight
#    3. get log standard values (e.g. pKI)
#    4. remove any repeats with conflicting measurements
#    (conflicting is standard derivation of pKIs > 1.0)
#
#    parameters:
#    x: pd.DataFrame, the dataframe to clean
#    num_workers: int, number of workers to use in parallel
#    max_mol_weight: float, maximum molecular weight to keep
#
#    returns:
#    pd.DataFrame, cleaned dataframe
#
#    """
#
#    # first clean the SMILES
#    def parallelize_standardization(df):
#        data_splits = np.array_split(df, num_workers)
#        with Pool(num_workers) as p:
#            df = pd.concat(p.map(standardize_smiles, data_splits))
#        return df
#
#    df = parallelize_standardization(x)
#    print (f'After standardizing the SMILES, the shape of the df: {df.shape}')
#
#    # drop anything that the molecular weight is too high
#    df = df.loc[df.molecular_weight <= max_mol_weight]
#    print (f'After dropping the mols with MW > {max_mol_weight} , the shape of the df: {df.shape}')
#
#    # get log standard values -- need to convert uM first
#    df["pStandard_value"] = 'None'
#
#    # remove duplicate
#    # first need to just keep one of the duplicates if smiles and value are *exactly* the same
#    df = df.drop_duplicates(subset=["canonical_smiles_by_Std", "standard_value"], keep="first")
#    print (f'After dropping the duplicate combinations of (smiles, value) , the shape of the df:{df.shape}')
#
#    df["max_num_atoms"] = df.num_atoms.max()
#    df["max_molecular_weight"] = df.molecular_weight.max()
#
#    return df