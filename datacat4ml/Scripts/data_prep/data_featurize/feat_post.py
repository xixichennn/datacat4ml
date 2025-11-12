import os
import pandas as pd
import argparse

from datacat4ml.Scripts.const import FEAT_DATA_DIR, FEAT_HHD_OR_DIR, FEAT_MHD_OR_DIR, FEAT_LHD_OR_DIR, FEAT_MHD_effect_OR_DIR
from datacat4ml.Scripts.const import DESCRIPTORS
from datacat4ml.Scripts.data_prep.data_featurize.feat_smi_list import Spl_Feat_Dic

#=============================================================================
# merge_feat_pkl
#=============================================================================
def merge_feat_pkls(in_dir, rmvD: int = 0):
    """
    After featurization by different descriptors, 
    get the descriptor column from each pickle file and append it to the original curated dataframe,
    and finally save the merged dataframe to the corresponding "all" subfolder in FEAT_*_DIR.

    Params:
    ------
    in_dir: str
        The input directory contains the original split files. e.g. SPL_HHD_OR_DIR
    rmvD: int
        indicate the location of the input directory.

    Returns:
    -------
    None
    """
    # input directory
    in_file_dir = os.path.join(in_dir, f'rmvD{str(rmvD)}')
    print(f'in_file_dir is {in_file_dir}\n')
    files = os.listdir(in_file_dir)

    # output directory
    pkl_dir = os.path.join(Spl_Feat_Dic[in_dir], f'rmvD{str(rmvD)}')
    print(f'pkl_dir is {pkl_dir}\n')


    # GET the descriptor column from each pickle file
    for f in files:
        f_prefix = f.replace('_split.csv', '')
        df = pd.read_csv(os.path.join(in_file_dir, f"{f_prefix}_split.csv"))

        for descriptor in DESCRIPTORS:
            pkl_file = os.path.join(pkl_dir, f"{f_prefix}_{descriptor}.pkl")
            pkl_df = pd.read_pickle(pkl_file)
            # append the descriptor column to the original dataframe
            df[descriptor] = pkl_df[descriptor].values

        # save the merged dataframe to out_dir
        merged_pkl_file = os.path.join(pkl_dir, f"{f_prefix}_featurized.pkl")
        df.to_pickle(merged_pkl_file)

        # remove the temporary descriptor pickle files
        for descriptor in DESCRIPTORS:
            pkl_file = os.path.join(pkl_dir, f"{f_prefix}_{descriptor}.pkl")
            os.remove(pkl_file)

#=============================================================================
# get_feat_stats
#=============================================================================
def get_feat_stats(in_dir: str = FEAT_HHD_OR_DIR, ds_cat_level: str = 'hhd', ds_type: str = 'or', rmvD: int = 0):
    """
    Get the statistics of the featurized datasets and save them to a csv file.
    """

    print(f'Processing: {in_dir}')
    feat_path = os.path.join(in_dir, f'rmvD{str(rmvD)}')
    feat_files = [f for f in os.listdir(feat_path)]

    for f in feat_files:
        print(f'f is {f}')

        target, effect, assay, standard_type, assay_chembl_id, ds_cat_level, ds_size_level, ds_size_level_noStereo = f.split('_')[:8]

        df = pd.read_pickle(os.path.join(feat_path, f))

        ################################ write the stats ################################
        stats_file_path = os.path.join(FEAT_DATA_DIR, f'feat_{ds_cat_level}_{ds_type}_rmvD{rmvD}_stats.csv')

        if not os.path.exists(stats_file_path):
            with open(stats_file_path, 'w') as f:
                f.write('ds_cat_level,ds_type,ds_size_level,ds_size_level_noStereo,target_chembl_id,effect,assay,standard_type,assay_chembl_id,feated_size\n')

        with open(stats_file_path, 'a') as f:
            f.write(f'{ds_cat_level},{ds_type},{ds_size_level},{ds_size_level_noStereo},{target},{effect},{assay},{standard_type},{assay_chembl_id},{len(df)}\n')

#=============================================================================
# main
#=============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge feature pickle files and get feature statistics.')
    parser.add_argument('--rmvD', type=int, help='Indicate the location of the input directory.', default=0)
    args = parser.parse_args()

    #============ merge_feat_pkls =================
    keys_list = list(Spl_Feat_Dic.keys())

    for in_dir in keys_list:
        print(f'Processing {in_dir}...')
        merge_feat_pkls(in_dir, rmvD=args.rmvD)

    #============ get_feat_stats =================
    get_feat_stats(in_dir=FEAT_HHD_OR_DIR, ds_cat_level='hhd', ds_type='or', rmvD=args.rmvD)
    get_feat_stats(in_dir=FEAT_MHD_OR_DIR, ds_cat_level='mhd', ds_type='or', rmvD=args.rmvD)
    get_feat_stats(in_dir=FEAT_LHD_OR_DIR, ds_cat_level='lhd', ds_type='or', rmvD=args.rmvD)
    get_feat_stats(in_dir=FEAT_MHD_effect_OR_DIR, ds_cat_level='mhd_effect', ds_type='or', rmvD=args.rmvD)