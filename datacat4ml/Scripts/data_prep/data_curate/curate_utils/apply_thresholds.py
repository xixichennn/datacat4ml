## This module is adopted from FS-MOL
## https://github.com/microsoft/FS-Mol.git

import pandas as pd
from typing import Tuple

relation_set_lessthan = {"<", "<="}
relation_set_morethan = {">", ">="}
relation_equals = {"=", "~"}

def get_activity_comment(x: float, threshold: float, buffer: float = 0.5) -> str:

    """
    Apply a threshold to activity measurements.
    This funcion is different from how it is done in FS-MOL, which doesn't concern return inactive when value > threshold + buffer and relation is > or >=.

    For example, IC50/ EC50 measurements.
    

    """

    # for now just use the passed median value to threshold
    # this is calculated from the entire dataframe.

    value = x["pStandard_value"]
    relation = x["standard_relation"]

    if relation in relation_set_morethan:
        return "inactive"
    elif relation in relation_set_lessthan or relation in relation_equals:
        if value >= (threshold + buffer):
            return "active"
        elif value > threshold and value < (threshold + buffer):
            return "weak active"
        elif value > (threshold - buffer) and value <= threshold:
            return "weak inactive"
        elif value <= (threshold - buffer):
            return "inactive"
        else:
            return "unknown"

def autothreshold(x: pd.Series, aim:str) -> Tuple[pd.DataFrame, float]:

    """
    Apply autothesholding procedure to data.

    param 
    -------
    x: pd.Series
        Data series containing activity measurements for a single assay.
    aim: str
        the aim of the model build upon this dataset, e.g. lo(lead optimization), vs(virtual screening)

    1) Find the median for an assay
    2) Use the median as a threshold if it sits within the required range: 
        vs: 4 <= median(pXC) <= 6 
        lo: 5 <= median(pXC) <= 7
    If the median is outside the required range, fix to pXC = 5.0 (10 uM) for vs aim, and pXC = 6.3 (500 nM) for lo aim.
    3) Apply the threshold to the data series.

    For activity measurements, log standard value is used.
    """

    df = pd.DataFrame(x)

    # threshold limits
    threshold_limits = (4, 6) if aim == "vs" else (5, 7)

    # get median and buffer
    median = df["pStandard_value"].median()
    buffer = df["pStandard_value"].std() / 10

    # fix threshold if median is outside limits
    if median < threshold_limits[0] or median > threshold_limits[1]:
        threshold = 5.0 if aim == "vs" else 6.3
    else:
        threshold = median

    col = f"{aim}_activity_comment"
    df[col] = df.apply(get_activity_comment, args=(threshold,), buffer=buffer, axis=1)

    return df, threshold

def fixedthreshold(x: pd.Series, aim: str) -> Tuple[pd.DataFrame, float]:

    """
    Apply fixed threshold to the data.

    param
    -------
    x: pd.Series
        Data series containing activity measurements for a single assay.
    aim: str
        the aim of the model build upon this dataset, e.g. lo(lead optimization), vs(virtual screening)
    
    For lo, pXC = 6.3, or inhibition = 80%
    For vs, pXC = 5.0, or inhibition = 50%

    """

    df = pd.DataFrame(x)
    threshold = 5.0 if aim == "vs" else 6.3

    col = f"{aim}_activity_comment"
    df[col] = df.apply(get_activity_comment, args=(threshold,), axis=1)

    return df, threshold

def apply_thresholds(
    x: pd.DataFrame,
    automate_threshold: bool = True,
    hard_only: bool = False,
    aim: str = "vs",
    **kwargs,
) -> pd.DataFrame:

    """
    Thresholding to obtain binary labels.

    param
    -------
    x: pd.DataFrame
        Dataframe containing activity measurements for a single assay.
    automate_threshold: bool
        Whether to use autothresholding or fixed thresholding.
    hard_only: bool
        Whether to only keep "active" and "inactive" labels, or also keep "weak active" and "weak inactive" labels.
    aim: str
        the aim of the model build upon this dataset, e.g. lo(lead optimization), vs(virtual screening)

    """
    print(f"==>Applying thresholds... ")   

    if len(x) > 0:
        if automate_threshold:
            df, threshold = autothreshold(x, aim=aim)
        else:
            df, threshold = fixedthreshold(x, aim=aim)

        # convert activity classes to binary labels
        # activity column name
        col = f"{aim}_activity_comment"
        thr_col = f"{aim}_threshold"
        bin_col = f"{aim}_activity"

        allowed_labels = ["active", "inactive"]
        if not hard_only:
            allowed_labels += ["weak active", "weak inactive"]
        
        print(f'available labels in {col}: {df[col].unique().tolist()}')

        ## filter labels: remove rows where 'standard_relation' and 'standard_value' cannot give a label. The 'standard_relation' and 'standard_value' can be NaN or other unexpected values.
        #df = df[df[col].isin(allowed_labels)]

        # map to binary labels
        active_labels = {"active", "weak active"}
        df[bin_col] = df[col].apply(lambda v: 1.0 if v in active_labels else 0.0)

        # store threshold
        df[thr_col] = threshold

        print(f"Threshold applied: {threshold}, the shape of the df after applying thresholds: {df.shape}")

        return df
    
    else:
        print(f"Empty dataframe after applying thresholds")
        return pd.DataFrame()