# conda env: datacat (Python3.8.20)
import shutil
import argparse
import os

from datacat4ml.const import CURA_HHD_OR_DIR, CURA_MHD_OR_DIR #Yu
from datacat4ml.const import CAT_HHD_OR_DIR, CAT_MHD_OR_DIR, OR_chemblids
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.curate_dataset_type import run_curation, group_by_effect


def curate_ORs(rmv_dupMol):

    """
    params
    ------
    rmv_dupMol: int: whether to remove duplicate SMILES with different values.

    
    run the curation process on datasets including:
    - hhd_or
    - mhd_or
    - lhd_or

    and get the stats for each dataset type.

    """
    # ==== hhd_or ====
    print('Processing hhd_or datasets ...')

    run_curation(ds_cat_level='hhd', input_path=CAT_HHD_OR_DIR, output_path=CURA_HHD_OR_DIR,
                 targets_list=OR_chemblids, effect=None, assay=None, std_types=["Ki", "IC50", 'EC50'], ds_type='or', rmv_dupMol=rmv_dupMol)

    # ==== mhd_or ====
    print('Processing mhd_or datasets ...')

    # Binding affinity
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='bind', assay='RBA', std_types=["Ki", 'IC50'], ds_type='or', rmv_dupMol=rmv_dupMol)

    # Agonism
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-GTP', std_types=["EC50"], ds_type='or', rmv_dupMol=rmv_dupMol)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-Ca', std_types=["EC50"], ds_type='or', rmv_dupMol=rmv_dupMol)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-cAMP', std_types=["IC50", "EC50"], ds_type='or', rmv_dupMol=rmv_dupMol)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='B-arrest', std_types=["EC50"], ds_type='or', rmv_dupMol=rmv_dupMol)

    ## Antagonism
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='antag', assay='G-GTP', std_types=["IC50", "Ki", "Kb", "Ke"], ds_type='or', rmv_dupMol=rmv_dupMol)
    run_curation(ds_cat_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='antag', assay='B-arrest', std_types=["IC50"], ds_type='or', rmv_dupMol=rmv_dupMol)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rmv_dupMol', type=int, required=True, help='whether to remove duplicate SMILES with different values')

    args = parser.parse_args()
    
    curate_ORs(args.rmv_dupMol)

    group_by_effect(ds_type='or', ds_cat_level='mhd', rmv_dupMol=args.rmv_dupMol)

