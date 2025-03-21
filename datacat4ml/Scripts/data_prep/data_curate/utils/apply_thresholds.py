## This module is adopted from FS-MOL
## https://github.com/microsoft/FS-Mol.git

import pandas as pd
from typing import Tuple

relation_set_lessthan = {"<", "<="}
relation_set_morethan = {">", ">="}
relation_equals = {"=", "~"}

def activity_threshold(x: float, threshold: float, buffer: float = 0.5) -> str:

    """
    Apply a threshold to activity measurements.

    For example, IC50/ EC50 measurements.
    """

    # for now just use the passed median value to threshold
    # this is calculated from the entire dataframe.

    value = x["pStandard_value"]
    relation = x["standard_relation"]

    # note: standard relations apply to standard value not log standard
    if value >= (threshold + buffer):
        return "active"
    elif value > threshold and value < (threshold + buffer) and relation in relation_set_lessthan:
        return "active"
    elif (
        value > threshold
        and value < (threshold + buffer)
        and relation in relation_set_morethan.union(relation_equals)
    ):
        return "weak active"
    elif (
        value > (threshold - buffer)
        and value <= threshold
        and relation in relation_set_lessthan.union(relation_equals)
    ):
        return "weak inactive"
    elif value > (threshold - buffer) and value <= threshold and relation in relation_set_morethan:
        return "inactive"
    elif value <= (threshold - buffer):
        return "inactive"
    

def autothreshold(x: pd.Series) -> Tuple[pd.DataFrame, float]:

    """
    Apply autothesholding procedure to data:

    1) Find the median for an assay
    2) Use the median as a threshold if it sits within the required range: 4 <= median(pXC) <= 6
    If the median is outside the required range, fix to pXC = 5.0 (i.e standard_value 1 uM)
    3) Apply the threshold to the data series.

    For activity measurements, log standard value is used.
    """

    df = pd.DataFrame(x)
    
    # use as a threshold provided it is in a sensible
    # range. This was chosen as pKI 4-6 in general
    threshold_limits = (4, 6)
    # get median
    median = df["pStandard_value"].median()
    threshold = median
    buffer = df["pStandard_value"].std() / 10

    # fix threshold to 5.0 if median is outside of the limits   
    if median < threshold_limits[0] or median > threshold_limits[1]:
        threshold = 5.0
    else:
        threshold = median

    df["activity_string"] = df.apply(
        activity_threshold, args=(threshold,), buffer=buffer, axis=1
    )
    
    return df, threshold


def fixedthreshold(x: pd.Series) -> Tuple[pd.DataFrame, float]:

    """
    Apply fixed threshold to the data.

    pXC = 5.0, or inhibition = 50%

    """

    df = pd.DataFrame(x)

    threshold = 5.0
    df["activity_string"] = df.apply(activity_threshold, args=(threshold,), axis=1)

    return df, threshold

def apply_thresholds(
    x: pd.DataFrame,
    automate_threshold: bool = True,
    hard_only: bool = False,
    **kwargs,
) -> pd.DataFrame:

    """
    Thresholding to obtain binary labels.

    """
    print(f"Applying thresholds ")   

    if len(x) > 0:
        if automate_threshold:
            df, threshold = autothreshold(x)
        else:
            df, threshold = fixedthreshold(x)

        # convert activity classes to binary labels
        if hard_only:
            df = df[df.activity_string.isin(["active", "inactive"])]
        else:
            df = df[df.activity_string.isin(["active", "inactive", "weak active", "weak inactive"])]

        df.loc[df["activity_string"] == "active", "activity"] = 1.0
        df.loc[df["activity_string"] == "weak active", "activity"] = 1.0
        df.loc[df["activity_string"] == "inactive", "activity"] = 0.0
        df.loc[df["activity_string"] == "weak inactive", "activity"] = 0.0

        df["threshold"] = threshold
        return df
    
    else:
        print(f"Empty dataframe after applying thresholds")