import os
from pathlib import Path
import inspect

import pandas as pd

def mkdirs(path : str):
    """Create a directory if it does not exist."""
    path = path.strip().strip("'").strip('"')
    os.makedirs(path, exist_ok=True)

def get_df_name(df: pd.DataFrame) -> str:
    """
    Get the name of the dataframe
    """
    frame = inspect.currentframe().f_back
    global_vars = frame.f_globals
    for name, obj in global_vars.items():
        if obj is df:
            return name
    return ''
