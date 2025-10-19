import os
import glob
import pandas as pd

from datacat4ml.Scripts.data_prep.data_split.intSplit_mldata import Cura_Spl_Dic
from datacat4ml.const import SPL_DATA_DIR, SPL_HHD_OR_DIR, SPL_MHD_OR_DIR, SPL_MHD_effect_OR_DIR, SPL_LHD_OR_DIR

import argparse

#===============================================================================
# Get statistics of the split datasets
#===============================================================================
def count_splCol(df, patterns):
    """Count the number of split columns in a dataframe based on a specific substring pattern."""
    return {name: sum(df.columns.str.contains(name)) for name in patterns}

def get_spl_stats(in_path: str = SPL_HHD_OR_DIR, ds_cat_level: str = 'hhd', ds_type: str = 'or', rmv_dupMol: int = 0):
    
    """Get the statistics of the split datasets and save them to a csv file."""
    
    print(f'Processing: {in_path}...')
    in_file_dir = os.path.join(in_path, 'rmvDupMol'+str(rmv_dupMol))
    files = [f for f in os.listdir(in_file_dir)]
    print(f'{len(files)} files found.')

    pattern_labels = [
        "rmvStereo0_rs_lo", "rmvStereo0_rs_vs", "rmvStereo0_cs", "rmvStereo0_ch",
        "rmvStereo1_rs_lo", "rmvStereo1_rs_vs", "rmvStereo1_cs", "rmvStereo1_ch"
    ]

    stats = []
    for f in files:
        print(f'f is {f}')
        df = pd.read_csv(os.path.join(in_file_dir, f)).iloc[:, 50:] # get only the split columns
        parts = f.replace('.csv', '').split('_')

        # Create a metadata dictionary
        meta = {}
        meta["ds_size_level"] = parts[6]
        meta["ds_size_level_noStereo"] = parts[7]
        meta["target_chembl_id"] = parts[0]
        meta["effect"] = parts[1]
        meta["assay"] = parts[2]
        meta["standard_type"] = parts[3]
        meta["assay_chembl_id"] = parts[4]

        # count split columns
        counts = count_splCol(df, pattern_labels)
        stats.append({
            "ds_cat_level": ds_cat_level,
            "ds_type": ds_type,
            **meta, **counts
        })
    
    # write stats to a csv file
    stats_file = os.path.join(SPL_DATA_DIR, f'spl_{ds_cat_level}_{ds_type}_rmvDupMol{rmv_dupMol}_stats.csv')
    pd.DataFrame(stats).to_csv(stats_file, index=False)
    print(f"Saved stats to: {stats_file}")

#===============================================================================
# Main
#===============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Post-process split datasets: merge files and get statistics.")
    parser.add_argument("--rmv_dupMol", type=int, choices=[0, 1], default=1, help="Whether to process files with duplicate molecules removed (1) or not (0). Default is 1.")
    args = parser.parse_args()
    
    # get stats
    get_spl_stats(in_path=SPL_HHD_OR_DIR, ds_cat_level='hhd', ds_type='or', rmv_dupMol=args.rmv_dupMol)
    get_spl_stats(in_path=SPL_MHD_OR_DIR, ds_cat_level='mhd', ds_type='or', rmv_dupMol=args.rmv_dupMol)
    get_spl_stats(in_path=SPL_MHD_effect_OR_DIR, ds_cat_level='mhd-effect', ds_type='or', rmv_dupMol=args.rmv_dupMol)
    get_spl_stats(in_path=SPL_LHD_OR_DIR, ds_cat_level='lhd', ds_type='or', rmv_dupMol=args.rmv_dupMol)
