import os
from typing import List
from tqdm import tqdm
import argparse

import pandas as pd

from datacat4ml.const import SPL_DATA_DIR, SPL_LHD_OR_DIR, SPL_MHD_OR_DIR, SPL_MHD_effect_OR_DIR, SPL_HHD_OR_DIR

#===============================================================================
# aligned_split
#===============================================================================

################## get parent_child_pairs ##################
dir_name_dict = {
    'hhd': SPL_HHD_OR_DIR,
    'mhd-effect': SPL_MHD_effect_OR_DIR,
    'mhd': SPL_MHD_OR_DIR,
    'lhd': SPL_LHD_OR_DIR
}

alignment_map = {
    'hhd': ['mhd', 'lhd'],
    'mhd-effect': ['mhd', 'lhd'],
    'mhd': ['lhd']
}

def get_parent_child_pairs(alignment_map):
    """Generate a list of parent-child dataset pairs based on the provided alignment map."""
    parent_child_pairs = []
    for key, values in alignment_map.items():
        for v in values:
            parent_child_pairs.append((key, v))
    print(f'parent_child_pairs is \n{parent_child_pairs}')

    return parent_child_pairs

def meta_cols(f:str):
    """Extract metadata columns from the filename."""
    parts = f.replace('.csv', '').split('_')
    target, effect, assay, standard_type, assay_chembl_id = parts[0], parts[1], parts[2], parts[3], parts[4]
    return target, effect, assay, standard_type, assay_chembl_id

def get_pf_cfs_pairs(rmv_dupMol: int = 1):
    """Get comparison pairs for aligned splits between parent and child datasets.

    params
    ------
    - rmv_dupMol: int, whether duplicate molecules have been removed. It is used to locate the correct directories.

    returns
    -------
    - comparison_pairs: dict
        A dictionary where keys are tuples of (parent, child), and values are dictionaries mapping parent files to lists of matching child files.
    """

    pf_cfs_pairs = {}

    parent_child_pairs = get_parent_child_pairs(alignment_map)
    for pair in parent_child_pairs:

        # get the dir
        parent, child = pair
        #print(f"\nparent is {parent}, child is {child}")
        parent_dir = dir_name_dict[parent]
        #print(f'parent_dir is {parent_dir}')
        child_dir = dir_name_dict[child]
        #print(f'child_dir is {child_dir}')

        pf_cfs_map = {}
        # iterate the files
        for pf in os.listdir(os.path.join(parent_dir, f'rmvDupMol{rmv_dupMol}')):
            #print(f'parent file is {pf}')
            target_p, effect_p, assay_p, standard_type_p, assay_chembl_id_p = meta_cols(f=pf)

            cfs = []
            for cf in os.listdir(os.path.join(child_dir, f'rmvDupMol{rmv_dupMol}')):
                #print(f'child file is {cf}')
                target_c, effect_c, assay_c, standard_type_c, assay_chembl_id_c = meta_cols(f=cf)
                
                # check whether the cols match
                if parent == 'hhd':
                    if target_p == target_c and standard_type_p == standard_type_c:
                        cfs.append(cf)
                if parent == 'mhd-effect':
                    if target_p == target_c and effect_p == effect_c:
                        cfs.append(cf)
                if parent == 'mhd':
                    if target_p == target_c and effect_p == effect_c and assay_p == assay_c and standard_type_p == standard_type_c:
                        cfs.append(cf)
            pf_cfs_map[pf] = cfs

        pf_cfs_pairs[pair] = pf_cfs_map

    return pf_cfs_pairs

def aligned_split(rmv_dupMol: int = 1):
    """
    Perform aligned splits for parent-child file pairs.
    Each parent file will only be written once, including all split columns derived from all its corresponding child files.

    params
    ------
    - rmv_dupMol: int, whether duplicate molecules have been removed. It is used to locate the correct directories.
    """

    pf_cfs_pairs = get_pf_cfs_pairs(rmv_dupMol=rmv_dupMol)

    # collect all derived split columns per parent file across all pairs
    parent_file_map = {} # key = (parent, parent_file), value = dict of DataFrame and added columns

    for (parent, child), pf_cfs_map in pf_cfs_pairs.items():
        pf_path = os.path.join(dir_name_dict[parent], f'rmvDupMol{rmv_dupMol}')
        cf_path = os.path.join(dir_name_dict[child], f'rmvDupMol{rmv_dupMol}')
        print(f"\nProcessing parent='{parent}', child='{child}' ...")

        for pf, cfs in pf_cfs_map.items():
            pf_key = (parent, pf)

            # load parent file if not alreay in memory
            if pf_key not in parent_file_map:
                pf_df = pd.read_csv(os.path.join(pf_path, pf))
                parent_file_map[pf_key] = pf_df
            else:
                pf_df = parent_file_map[pf_key]
            
            # collect new columns for this parent file
            new_cols = {}
            for cf in cfs:
                cf_basename = cf[:-22] # remove the suffix. e.g. '_lhd_b50_s50_split.csv'
                cf_df = pd.read_csv(os.path.join(cf_path, cf))
                cf_df_cols = [c for c in cf_df.columns if c.__contains__('fold')]

                for col in cf_df_cols:
                    test_idx = cf_df.index[cf_df[col] == 'test'].tolist()
                    activity_ids = cf_df.loc[test_idx, 'activity_id'].tolist()
                    col_basename = col.split('.')[1] 
                    parent_col = f'aln.{parent}-{child}.{cf_basename}.{col_basename}'
                    
                    # avoid duplicate columns if the column already exists
                    if parent_col not in pf_df.columns and parent_col not in new_cols:
                        new_cols[parent_col]= ['test' if id in activity_ids else 'train' for id in pf_df['activity_id'].tolist()]
            
            # add all new columns at once to avoid fragmentation
            if new_cols:
                new_df = pd.DataFrame(new_cols)
                pf_df = pd.concat([pf_df, new_df], axis=1)
                parent_file_map[pf_key] = pf_df
    
    # --------- After all pairs are processed, save each parent file once -----------
    for (parent, pf), pf_df in parent_file_map.items():
        pf_path = os.path.join(dir_name_dict[parent], f'rmvDupMol{rmv_dupMol}')
        pf_df.to_csv(os.path.join(pf_path, pf), index=False) # overwrite the original parent file with added columns
        print(f'Saved aligned parent file: {os.path.join(pf_path, pf)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into test/train aligned for ML model training and evaluation")
    parser.add_argument('--rmv_dupMol', type=int, required=True, help='Remove duplicate molecules')

    args = parser.parse_args()

    aligned_split(rmv_dupMol=args.rmv_dupMol)