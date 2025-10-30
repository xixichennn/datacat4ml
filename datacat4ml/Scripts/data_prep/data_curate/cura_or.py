# conda env: datacat (Python3.8.20)
import shutil
import argparse
import os

from datacat4ml.const import CURA_HHD_OR_DIR, CURA_MHD_OR_DIR #Yu
from datacat4ml.const import CAT_HHD_OR_DIR, CAT_MHD_OR_DIR, OR_chemblids
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.curate_dataset_type import run_curation, group_by_effect


def curate_ORs(rmvD):

    """
    params
    ------
    rmvD: int: whether to remove duplicate SMILES with different values.

    
    run the curation process on datasets including:
    - hhd_or
    - mhd_or
    - lhd_or

    and get the stats for each dataset type.

    """
    print('rmvD:', rmvD)
    # ==== hhd_or ====
    print('Processing hhd_or datasets ...')

    run_curation(ds_cat_level='hhd', input_path=CAT_HHD_OR_DIR, output_path=CURA_HHD_OR_DIR,
                 targets_list=OR_chemblids, effect=None, assay=None, standard_types=["Ki", "IC50", 'EC50'], ds_type='or', rmvD=rmvD)

    # ==== mhd_or ====
    print('Processing mhd_or datasets ...')

    # Binding affinity
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='bind', assay='RBA', standard_types=["Ki", 'IC50'], ds_type='or', rmvD=rmvD)

    # Agonism
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-GTP', standard_types=["EC50"], ds_type='or', rmvD=rmvD)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-Ca', standard_types=["EC50"], ds_type='or', rmvD=rmvD)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-cAMP', standard_types=["IC50", "EC50"], ds_type='or', rmvD=rmvD)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='B-arrest', standard_types=["EC50"], ds_type='or', rmvD=rmvD)

    ## Antagonism
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='antag', assay='G-GTP', standard_types=["IC50", "Ki", "Kb", "Ke"], ds_type='or', rmvD=rmvD)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='antag', assay='B-arrest', standard_types=["IC50"], ds_type='or', rmvD=rmvD)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rmvD', type=int, required=True, help='whether to remove duplicate SMILES with different values')

    args = parser.parse_args()

    curate_ORs(args.rmvD)

    group_by_effect(ds_type='or', ds_cat_level='mhd', rmvD=args.rmvD)

