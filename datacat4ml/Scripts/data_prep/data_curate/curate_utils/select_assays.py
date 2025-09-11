## This module is adopted from FS-MOL
## https://github.com/microsoft/FS-Mol.git


import pandas as pd
import numpy as np

standard_unit_set = {"nM", "uM"}

def clean_units(x: pd.Series) -> bool:
    """Remove measurements that have units outside the permitted set"""
    return x["standard_units"] not in standard_unit_set

def clean_values(x: pd.Series) -> bool:
    """Remove where the standard value is None"""
    return np.isnan(x["standard_value"])

def select_assays(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Initial cleaning of all datapoints in an assay/file.

    Removes any points that don't have units in the permitted set:
    standard_unit_set = {"nM", "uM"},
    and converts the standard values to float.
    """

    # first step is to remove anything that doesn't have the approved units
    df = pd.DataFrame(x)
    df.drop(df[df.apply(clean_units, axis=1)].index, inplace=True)
    # drop any rows where the standard value is 'None'
    df.drop(df[df.apply(clean_values, axis=1)].index, inplace=True)
    
    # make sure standard values are floats
    df["standard_value"] = df["standard_value"].astype(float)

    return df