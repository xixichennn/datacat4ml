# conda env: datacat (Python3.8.20)
import shutil
import argparse
import os

from datacat4ml.const import CURA_HHD_OR_DIR, CURA_MHD_OR_DIR #Yu
from datacat4ml.const import CAT_HHD_OR_DIR, CAT_MHD_OR_DIR, OR_chemblids
from datacat4ml.Scripts.data_prep.data_curate.curate_utils.curate_dataset_type import run_curation


def curate_ORs():

    """
    run the curation process on datasets including:
    - hhd_or
    - mhd_or
    - lhd_or

    and get the stats for each dataset type.

    """
    ## ==== hhd_or ====
    print('Processing hhd_or datasets ...')

    run_curation(ds_level='hhd', input_path=CAT_HHD_OR_DIR, output_path=CURA_HHD_OR_DIR,
                 targets_list=OR_chemblids, effect=None, assay=None, std_types=["Ki", "IC50", 'EC50'], ds_type='or')

    ## ==== mhd_or ====
    print('Processing mhd_or datasets ...')

    # Binding affinity
    run_curation(ds_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='bind', assay='RBA', std_types=["Ki", 'IC50'], ds_type='or')

    # Agonism
    run_curation(ds_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-GTP', std_types=["EC50"], ds_type='or')
    run_curation(ds_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-Ca', std_types=["EC50"], ds_type='or')
    run_curation(ds_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='G-cAMP', std_types=["IC50", "EC50"], ds_type='or')
    run_curation(ds_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='agon', assay='B-arrest', std_types=["EC50"], ds_type='or')

    ## Antagonism
    run_curation(ds_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='antag', assay='G-GTP', std_types=["IC50", "Ki", "Kb", "Ke"], ds_type='or')
    run_curation(ds_level='mhd', input_path=CAT_MHD_OR_DIR, output_path= CURA_MHD_OR_DIR,
                targets_list=OR_chemblids, effect='antag', assay='B-arrest', std_types=["IC50"], ds_type='or')

    
if __name__ == '__main__':
    curate_ORs()

