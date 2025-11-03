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
# in lhd files: 
#aln.child.hhd.lhd ~~ int_col
#aln.child.mhd-effect.lhd ~~ int_col
#aln.child.mhd.lhd ~~ int_col

# in mhd files:
# aln.parent.mhd.lhd
# aln.child.hhd.mhd ~~ int_col
# aln.child.mhd-effect.mhd ~~ int_col

# in mhd-effect files:
# aln.parent.mhd-effect.mhd
# aln.parent.mhd-effect.lhd

# in hhd files:
# aln.parent.hhd.mhd
# aln.parent.hhd.lhd

def get_pd_cd_pairs(alignment_map):
    """Generate a list of pd(parent directory)-cd(child directory) pairs based on the provided alignment map."""
    pd_cd_pairs = []
    for key, values in alignment_map.items():
        for v in values:
            pd_cd_pairs.append((key, v))

    return pd_cd_pairs

def meta_cols(f:str):
    """Extract metadata columns from the filename."""
    parts = f.replace('.csv', '').split('_')
    target_chembl_id, effect, assay, standard_type, assay_chembl_id = parts[0], parts[1], parts[2], parts[3], parts[4]
    return target_chembl_id, effect, assay, standard_type, assay_chembl_id

def get_pfp_cfps_all(rmvD: int = 1):
    """For aligned splits, get the comparison pairs of prefix of parent and child files.

    params
    ------
    - rmvD: int, whether duplicate molecules have been removed. It is used to locate the correct directories.

    returns
    -------
    - comparison_pairs: dict
        A dictionary where keys are tuples of (parent, child), and values are dictionaries mapping parent files to lists of matching child files.
    """

    pfp_cfps_all = {}

    pd_cd_pairs = get_pd_cd_pairs(alignment_map)
    for pair in pd_cd_pairs:
        
        # get the dir
        pd_cat_level, cd_cat_level = pair
        #print(f"\nparent directory is {pd}, child directory is {cd}")
        pd = dir_name_dict[pd_cat_level]
        #print(f'pd is {pd}')
        cd = dir_name_dict[cd_cat_level]
        #print(f'cd is {cd}')

        pfp_cfps_map = {} # pfp = parent file prefix, cfp = child file prefixes
        # iterate the files
        for pf in os.listdir(os.path.join(pd, f'rmvD{rmvD}')):
            #print(f'parent file is {pf}')
            target_p, effect_p, assay_p, standard_type_p, assay_chembl_id_p = meta_cols(f=pf)
            pf_prefix = '_'.join(pf.split('_')[:6])

            cf_prefixes = []
            for cf in os.listdir(os.path.join(cd, f'rmvD{rmvD}')):
                #print(f'child file is {cf}')
                cf_prefix = '_'.join(cf.split('_')[:6])
                target_c, effect_c, assay_c, standard_type_c, assay_chembl_id_c = meta_cols(f=cf)
                
                # check whether the cols match
                if pd_cat_level == 'hhd':
                    if target_p == target_c and standard_type_p == standard_type_c:
                        cf_prefixes.append(cf_prefix)
                if pd_cat_level == 'mhd-effect':
                    if target_p == target_c and effect_p == effect_c:
                        cf_prefixes.append(cf_prefix)
                if pd_cat_level == 'mhd':
                    if target_p == target_c and effect_p == effect_c and assay_p == assay_c and standard_type_p == standard_type_c:
                        cf_prefixes.append(cf_prefix)

            pfp_cfps_map[pf_prefix] = cf_prefixes

        pfp_cfps_all[pair] = pfp_cfps_map

    return pfp_cfps_all

def aligned_split(rmvD: int = 1):
    """
    Perform aligned splits for pf-cf file pairs.
    Each parent file will only be written once, including all split columns derived from all its corresponding child files.

    params
    ------
    - rmvD: int, whether duplicate molecules have been removed. It is used to locate the correct directories.
    """

    pfp_cfps_all = get_pfp_cfps_all(rmvD=rmvD)

    # collect all derived split columns per parent file across all pairs
    pf_storage_dict = {} # key = (pf_cat_level, pf), value = dict of DataFrame with added columns
    #cf_map = {}  # key = (cd_cat_level, cf), value = dict of DataFrame with added columns

    for (pd_cat_level, cd_cat_level), pfp_cfps_map in pfp_cfps_all.items():
        pf_path = os.path.join(dir_name_dict[pd_cat_level], f'rmvD{rmvD}')
        cf_path = os.path.join(dir_name_dict[cd_cat_level], f'rmvD{rmvD}')
        print(f"\nProcessing pd_cat_level:'{pd_cat_level}', cd_cat_level:'{cd_cat_level}' ...")

        for pf_prefix, cf_prefixes in pfp_cfps_map.items():
            pf = [f for f in os.listdir(pf_path) if f.startswith(pf_prefix)][0] # there is only one such file
            pf_key = (pd_cat_level, pf) # e.g. ('hhd', 'CHEMB233_None_None_Ki_None_hhd_b50_b50_split.csv')

            # load parent file if not alreay in memory
            if pf_key not in pf_storage_dict:
                pf_df = pd.read_csv(os.path.join(pf_path, pf))
                pf_storage_dict[pf_key] = pf_df
            else:
                pf_df = pf_storage_dict[pf_key]

            # collect new columns for this parent file
            pf_new_cols = {}
            for cf_prefix in cf_prefixes:
                cf = [f for f in os.listdir(cf_path) if f.startswith(cf_prefix)][0]

                cf_new_cols = {}
                cf_prefix = '_'.join(cf.split('_')[:6]) # get the base name by joining the first 6 identifiers
                cf_df = pd.read_csv(os.path.join(cf_path, cf))
                cf_cols = [c for c in cf_df.columns if c.startswith('rmvS')] # take internal split columns only

                for cf_col in cf_cols:
                    # create new column for the parent file based on the child file's split
                    cf_test_idx = cf_df.index[cf_df[cf_col] == 'test'].tolist()
                    cf_test_activity_ids = cf_df.loc[cf_test_idx, 'activity_id'].tolist()
                    pf_new_col = f'parent.{pf_prefix}.{cf_prefix}.{cf_col}'

                    if pf_new_col.__contains__('rmvS0'):
                        pf_new_cols[pf_new_col]= ['test' if id in cf_test_activity_ids else 'train' for id in pf_df['activity_id'].tolist()]
                    elif pf_new_col.__contains__('rmvS1'):
                        stereo_activity_ids = pf_df.loc[pf_df['stereoSiblings'] == True, 'activity_id'].tolist()
                        pf_new_cols[pf_new_col] = [
                            'test' if (id in cf_test_activity_ids and id not in stereo_activity_ids)
                            else 'train' if id not in stereo_activity_ids
                            else None
                            for id in pf_df['activity_id'].tolist()
                        ]

                    # create new column for the child file based on the assigned split in the parent file
                    cf_new_col = f'child.{pf_prefix}.{cf_prefix}.{cf_col}'
                    pf_test_activity_ids = [id for id, split in zip(pf_df['activity_id'], pf_new_cols[pf_new_col]) if split == 'test']
                    # the difference below originated from the rmvS1 and rmvD1 which may delete some molecules in parent file but still present in child file.
                    diff_test_activity_ids = list(set(cf_test_activity_ids) - set(pf_test_activity_ids))
                    cf_new_cols[cf_new_col] = cf_df[cf_col].tolist() # default copy all values
                    cf_new_cols[cf_new_col] = [
                            None if id in diff_test_activity_ids # assign None if in child's test but not in parent's test
                            else cf_df.loc[cf_df['activity_id'] == id, cf_col].values[0] # keep the original value otherwise
                            for id in cf_df['activity_id'].tolist()
                        ]
                
                if cf_new_cols:
                    # add all new columns at once to avoid fragmentation
                    new_df = pd.DataFrame(cf_new_cols)
                    cf_df = pd.concat([cf_df, new_df], axis=1)
                    # Overwrite the original child file with added columns
                    # the code line below works well because the child files are processed in the order of `alignment_map`. 
                    # Although 'mhd' will be processed twice as child and parent files, 
                    # the 'aln.parent' column for parent mhd files will be created after the 'aln.child' columns for child mhd files have been created, 
                    # therefore no conflict occurs.
                    cf_df.to_csv(os.path.join(cf_path, cf), index=False)
                    print(f'Saved aligned child file: {os.path.join(cf_path, cf)}')
                    

            # add all new columns at once to avoid fragmentation
            if pf_new_cols:
                new_df = pd.DataFrame(pf_new_cols)
                pf_df = pd.concat([pf_df, new_df], axis=1)
                pf_storage_dict[pf_key] = pf_df

    # --------- After all pairs are processed, save each parent file once -----------
    for (pd_cat_level, pf), pf_df in pf_storage_dict.items():
        pf_path = os.path.join(dir_name_dict[pd_cat_level], f'rmvD{rmvD}')
        pf_df.to_csv(os.path.join(pf_path, pf), index=False) # overwrite the original parent file with added columns
        print(f'Saved aligned parent file: {os.path.join(pf_path, pf)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into test/train aligned for ML model training and evaluation")
    parser.add_argument('--rmvD', type=int, required=True, help='Remove duplicate molecules')

    args = parser.parse_args()

    aligned_split(rmvD=args.rmvD)