import os
import pandas as pd

from datacat4ml.utils import mkdirs
from datacat4ml.Scripts.data_prep.data_featurize.feat_smi_list import Cura_Feat_Dic

def merge_feat_pkls(in_dir):
    """
    After featurization by different descriptors, 
    get the descriptor column from each pickle file and append it to the original curated dataframe,
    and finally save the merged dataframe to the corresponding "all" subfolder in FEAT_*_DIR.

    Params:
    ------
    in_dir: str
        The input directory contains the original curated files. e.g. FEAT_HHD_GPCR_DIR, FEAT_MHD_OR_DIR

    Returns:
    -------
    None
    """
    descriptors = ['ECFP4', 'ECFP6', 'MACCS', 'RDKITFP', 'PHARM2D', 'ERG', 
                   'PHYSICOCHEM', 
                   'SHAPE3D', 'AUTOCORR3D', 'RDF', 'MORSE', 'WHIM', 'GETAWAY']
    

    # GET the base filenames from the original curated csv files
    files = os.listdir(in_dir) # in_dir should be the subfolder in CURA_DATA_DIR
    original_files = [file for file in files if file.endswith('_curated.csv')]

    base_names = []
    for original_file in original_files:
        base_name = "_".join(os.path.basename(original_file).split("_")[:-1]) # drop '_curated.csv'
        base_names.append(base_name)

    # GET the descriptor column from each pickle file
    for base_name in base_names:
        original_df = pd.read_csv(os.path.join(in_dir, f"{base_name}_curated.csv"))
        # if df contains column 'Unnamed: 0', drop it
        original_df = original_df.drop(columns=['Unnamed: 0'], errors='ignore')

        pkl_dir = Cura_Feat_Dic[in_dir]
        for descriptor in descriptors:
            pkl_file = os.path.join(pkl_dir, f"{base_name}_{descriptor}.pkl")
            pkl_df = pd.read_pickle(pkl_file)
            pkl_df = pkl_df.drop(columns=['Unnamed: 0'], errors='ignore')
            # append the descriptor column to the original dataframe
            original_df[descriptor] = pkl_df[descriptor].values
        
        # save the merged dataframe to out_dir
        out_dir = os.path.join(pkl_dir, "all")
        mkdirs(out_dir)
        merged_pkl_file = os.path.join(out_dir, f"{base_name}_featurized.pkl")
        original_df.to_pickle(merged_pkl_file)

if __name__ == "__main__":

    keys_list = list(Cura_Feat_Dic.keys())

    for in_dir in keys_list:
        print(f'Processing {in_dir}...')
        merge_feat_pkls(in_dir)

    # run this script in the terminal:
    # $ conda activate datacat
    # $ python3 merge_feat_pkls.py