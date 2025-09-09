# conda env: datacat(Python 3.8.20)
import os
import pandas as pd

from datacat4ml.utils import mkdirs
from datacat4ml.const import FETCH_DATA_DIR, CAT_DATA_DIR, HHD_GPCR_DIR, HHD_OR_DIR, CAT_FIG_DIR
from datacat4ml.const import OR_chemblids
from datacat4ml.Scripts.data_prep.data_categorize.cat_utils import hhd, mhd_lhd

#===========================================================================
# Read data 
#==========================================================================
def read_data():
    ki_gpcr_df = pd.read_csv(os.path.join(FETCH_DATA_DIR, 'Ki_gpcr_maxcur_8_data.csv'))
    ic50_gpcr_df = pd.read_csv(os.path.join(FETCH_DATA_DIR, 'IC50_gpcr_maxcur_8_data.csv'))
    ec50_gpcr_df = pd.read_csv(os.path.join(FETCH_DATA_DIR, 'EC50_gpcr_maxcur_8_data.csv'))

    GPCR_dfs = {}
    GPCR_chemblids = set(ki_gpcr_df['target_chembl_id']) | set(ic50_gpcr_df['target_chembl_id']) | set(ec50_gpcr_df['target_chembl_id'])
    for target_chembl_id in GPCR_chemblids:
        ki_df = ki_gpcr_df[ki_gpcr_df['target_chembl_id'] == target_chembl_id]
        ic50_df = ic50_gpcr_df[ic50_gpcr_df['target_chembl_id'] == target_chembl_id]
        ec50_df = ec50_gpcr_df[ec50_gpcr_df['target_chembl_id'] == target_chembl_id]

        act_df = pd.concat([ki_df, ic50_df, ec50_df], ignore_index=True)
        GPCR_dfs[target_chembl_id] = act_df
        print(f'The shape of {target_chembl_id}_df is \n ki: {ki_df.shape}, ic50: {ic50_df.shape}, ec50: {ec50_df.shape}')

    return GPCR_dfs, GPCR_chemblids
#==========================================================================
#  hhd_gpcr_datasets: All GPCRs including ORs
# ==========================================================================
def run_hhd(GPCR_dfs, GPCR_chemblids):
    # generate hhd datasets for ORs
    or_dfs, or_dfs_len =hhd(OR_chemblids, GPCR_dfs, HHD_OR_DIR)
    # generate hhd datasets for GPCRs
    gpcr_dfs, gpcr_dfs_len = hhd(GPCR_chemblids, GPCR_dfs, HHD_GPCR_DIR)

    hhd_or_dfs_len = pd.DataFrame()
    for key, len_df in or_dfs_len.items():
        hhd_or_dfs_len = pd.concat([hhd_or_dfs_len, len_df], axis=0, sort=False)


    hhd_gpcr_dfs_len = pd.DataFrame()
    for key, len_df in gpcr_dfs_len.items():
        hhd_gpcr_dfs_len = pd.concat([hhd_gpcr_dfs_len, len_df], axis=0, sort=False)

    return hhd_or_dfs_len, hhd_gpcr_dfs_len

#==========================================================================
# # mhd_datasets (ORs) & mhd_gpcr_datasets
# ==========================================================================
def run_mhd_lhd(GPCR_dfs, GPCR_chemblids):
    #===================== Radio =========================
    #radioligand replacement binding assay

    # create a boolean mask to filter rows that match p_bind 
    # and do not match p_bind_ex
    p_bind_RBA = r"(?i)affinity|displacement|3H|125I"
    p_bind_RBA_ex = r"(?i)camp|gtp|calcium|ca2+|IP1|IP3|arrest|agonis"

    ## mhd_datasets: ORs
    (or_type_dfs, 
     or_bind_plus_dfs, 
     or_bind_exclude_dfs, 
     or_bind_mhd_dfs,
     or_bind_mhd_dfs_len, 
     or_bind_lhd_dfs,
     or_bind_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=OR_chemblids, use_lookup=True, 
                            effect='bind', assay='RBA', std_types=['Ki', 'IC50'], 
                            pattern=p_bind_RBA, pattern_ex=p_bind_RBA_ex)

    # mhd_datasets: GPCRs
    (gpcr_type_dfs, 
    gpcr_bind_mhd_dfs,
    gpcr_bind_mhd_dfs_len,
    gpcr_bind_lhd_dfs,
    gpcr_bind_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                              effect='bind', assay='RBA', std_types=['Ki', 'IC50'], 
                              pattern=p_bind_RBA, pattern_ex=p_bind_RBA_ex)

    ########################### G-GTP ##########################
    # ======= Agonism: EC50 =======
    # G-protein dependent functional assays
    # GTPgamma binding assay
    p_agon_G_GTP = r"(?i)gtp"
    p_agon_G_GTP_ex = r"(?i)arrestin|camp|calcium|IP1|IP3|antagonis|inverse agonist|allosteric" 

    # mhd_datasets: ORs
    (or_type_dfs, 
    or_agon_G_GTP_plus_dfs, 
    or_agon_G_GTP_exclude_dfs, 
    or_agon_G_GTP_mhd_dfs, 
    or_agon_G_GTP_mhd_dfs_len,
    or_agon_G_GTP_lhd_dfs,
    or_agon_G_GTP_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=OR_chemblids, use_lookup=True, 
                                  effect='agon', assay='G-GTP', std_types=['EC50'], 
                                  pattern=p_agon_G_GTP, pattern_ex=p_agon_G_GTP_ex)

    # mhd_datasets: GPCRs
    (gpcr_type_dfs, 
     gpcr_agon_G_GTP_mhd_dfs, 
     gpcr_agon_G_GTP_mhd_dfs_len,
     gpcr_agon_G_GTP_lhd_dfs,
     gpcr_agon_G_GTP_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                                    effect='agon', assay='G-GTP', std_types=['EC50'], 
                                    pattern=p_agon_G_GTP, pattern_ex=p_agon_G_GTP_ex)

    # ======= Antagonism: IC50, Ki, Ke, Kb =======
    # GTPgammaS binding assay
    p_antag_G_GTP = r"(?i)gtp"
    p_antag_G_GTP_ex = r"(?i)arrestin|camp|calcium|IP1|IP3|allosteric"

    # mhd_datasets: ORs
    (or_type_dfs, 
     or_antag_G_GTP_plus_dfs, 
     or_antag_G_GTP_exclude_dfs, 
     or_antag_G_GTP_mhd_dfs,
     or_antag_G_GTP_mhd_dfs_len,
     or_antag_G_GTP_lhd_dfs,
     or_antag_G_GTP_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=OR_chemblids, use_lookup=True, 
                                   effect='antag', assay='G-GTP', std_types=['IC50', 'Ki'], 
                                   pattern=p_antag_G_GTP, pattern_ex=p_antag_G_GTP_ex)

    # mhd_datasets: GPCRs
    (gpcr_type_dfs, 
     gpcr_antag_G_GTP_mhd_dfs, 
     gpcr_antag_G_GTP_mhd_dfs_len,
     gpcr_antag_G_GTP_lhd_dfs,
     gpcr_antag_G_GTP_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                                     effect='antag', assay='G-GTP', std_types=['IC50', 'Ki'], 
                                     pattern=p_antag_G_GTP, pattern_ex=p_antag_G_GTP_ex)

    ########################### G-cAMP ##########################
    #========= Agonism: IC50, EC50 ==========
    # cAMP accumulation assay
    p_ago_G_cAMP = r"(?i)camp"
    p_ago_G_cAMP_ex = r"(?i)arrestin|gtp|calcium|IP1|IP3|antagonis|inverse agonist|allosteric"

    # mhd_datasets: ORs
    (or_type_dfs, 
     or_agon_G_cAMP_plus_dfs, 
     or_agon_G_cAMP_exclude_dfs,
     or_agon_G_cAMP_mhd_dfs, 
     or_agon_G_cAMP_mhd_dfs_len,
     or_agon_G_cAMP_lhd_dfs,
     or_agon_G_cAMP_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=OR_chemblids, use_lookup=True, 
                                   effect='agon', assay='G-cAMP', std_types=['IC50', 'EC50'], 
                                   pattern=p_ago_G_cAMP, pattern_ex=p_ago_G_cAMP_ex)

    # mhd_datasets: GPCRs
    (gpcr_type_dfs, 
     gpcr_agon_G_cAMP_mhd_dfs,
     gpcr_agon_G_cAMP_mhd_dfs_len,
     gpcr_agon_G_cAMP_lhd_dfs,
     gpcr_agon_G_cAMP_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                                     effect='agon', assay='G-cAMP', std_types=['IC50', 'EC50'], 
                                     pattern=p_ago_G_cAMP, pattern_ex=p_ago_G_cAMP_ex)

    #============= ### Antagonism ==========
    # neither IC50 data not EC50 data within G_cAMP assay is related to antagonism

    ########################### G-Ca ##########################
    #============ Agonism: EC50 ===========
    # IP3/IP1 and Ca2+ assay
    p_agon_G_Ca = r"(?i)calcium|ca2+|IP1|IP3"
    p_agon_G_Ca_ex = r"(?i)arrestin|gtp|camp|antagonis|inverse agonist|allosteric"

    # mhd_datasets: ORs
    (or_type_dfs, 
     or_agon_G_Ca_plus_dfs, 
     or_agon_G_Ca_exclude_dfs, 
     or_agon_G_Ca_mhd_dfs,
     or_agon_G_Ca_mhd_dfs_len,
     or_agon_G_Ca_lhd_dfs,
     or_agon_G_Ca_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=OR_chemblids, use_lookup=True, 
                                 effect='agon', assay='G-Ca', std_types=['EC50'], 
                                 pattern=p_agon_G_Ca, pattern_ex=p_agon_G_Ca_ex)

    # mhd_datasets: GPCRs
    (gpcr_type_dfs, 
     gpcr_agon_G_Ca_mhd_dfs,
     gpcr_agon_G_Ca_mhd_dfs_len,
     gpcr_agon_G_Ca_lhd_dfs,
     gpcr_agon_G_Ca_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                                   effect='agon', assay='G-Ca', std_types=['EC50'], 
                                   pattern=p_agon_G_Ca, pattern_ex=p_agon_G_Ca_ex)

    # ===============  Antagonism: IC50 ==========
    # IP3/IP1 and Ca2+ assay
    p_antag_G_Ca = r"(?i)calcium|ca2+|IP1|IP3"
    p_antag_G_Ca_ex = r"(?i)arrestin|gtp|camp|allosteric"

    # mhd_datasets: ORs
    # nearly no data points

    # mhd_datasets: GPCRs
    (gpcr_type_dfs, 
     gpcr_antag_G_Ca_mhd_dfs,
     gpcr_antag_G_Ca_mhd_dfs_len,
     gpcr_antag_G_Ca_lhd_dfs,
     gpcr_antag_G_Ca_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                                    effect='antag', assay='G-Ca', std_types=['IC50'], 
                                    pattern=p_antag_G_Ca, pattern_ex=p_antag_G_Ca_ex)

    ########################### B-arrest ##########################
    #============  Agonism: EC50 ===========
    # G-protein independent functional assays
    # Beta-arrestin recruitment assay
    p_agon_B_arrest = r"(?i)arrest"
    p_agon_B_arrest_ex = r"(?i)gtp|camp|calcium|IP1|IP3|antagonis|inverse agonist|allosteric"

    # mhd_datasets: ORs
    (or_type_dfs,  
     or_agon_B_arrest_plus_dfs, 
     or_agon_B_arrest_exclude_dfs, 
     or_agon_B_arrest_mhd_dfs,
     or_agon_B_arrest_mhd_dfs_len,
     or_agon_B_arrest_lhd_dfs,
     or_agon_B_arrest_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=OR_chemblids, use_lookup=True, 
                                     effect='agon', assay='B-arrest', std_types=['EC50'], 
                                     pattern=p_agon_B_arrest, pattern_ex=p_agon_B_arrest_ex)

    # mhd_datasets: GPCRs
    (gpcr_type_dfs, 
     gpcr_agon_B_arrest_mhd_dfs,
     gpcr_agon_B_arrest_mhd_dfs_len,
     gpcr_agon_B_arrest_lhd_dfs,
     gpcr_agon_B_arrest_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                                       effect='agon', assay='B-arrest', std_types=['EC50'], 
                                       pattern=p_agon_B_arrest, pattern_ex=p_agon_B_arrest_ex)

    #============ Antagonism: IC50 ===========
    # G-protein independent functional assays
    # Beta-arrestin recruitment assay
    p_antag_B_arrest = r"(?i)arrest"
    p_antag_B_arrest_ex = r"(?i)gtp|camp|calcium|IP1|IP3|allosteric"

    # mhd_datasets: ORs
    (or_type_dfs, 
     or_antag_B_arrest_plus_dfs, 
     or_antag_B_arrest_exclude_dfs, 
     or_antag_B_arrest_mhd_dfs,
     or_antag_B_arrest_mhd_dfs_len,
     or_antag_B_arrest_lhd_dfs,
     or_antag_B_arrest_lhd_dfs_len)  = mhd_lhd(dfs=GPCR_dfs,targets_list=OR_chemblids, use_lookup=True, 
                                       effect='antag', assay='B-arrest', std_types=['IC50'], 
                                       pattern=p_antag_B_arrest, pattern_ex=p_antag_B_arrest_ex)

    # cat_datasets: GPCRs
    (gpcr_type_dfs, 
     gpcr_antag_B_arrest_mhd_dfs,
     gpcr_antag_B_arrest_mhd_dfs_len,
     gpcr_antag_B_arrest_lhd_dfs,
     gpcr_antag_B_arrest_lhd_dfs_len) = mhd_lhd(dfs=GPCR_dfs,targets_list=GPCR_chemblids, use_lookup=False, 
                                        effect='antag', assay='B-arrest', std_types=['IC50'], 
                                        pattern=p_antag_B_arrest, pattern_ex=p_antag_B_arrest_ex)

    ############# mhd_or_dfs ###############
    mhd_or_dfs_len_list = [or_bind_mhd_dfs_len, 
                          or_agon_G_GTP_mhd_dfs_len, 
                          or_agon_G_cAMP_mhd_dfs_len, 
                          or_agon_G_Ca_mhd_dfs_len, 
                          or_agon_B_arrest_mhd_dfs_len, 
                          or_antag_G_GTP_mhd_dfs_len,  
                          or_antag_B_arrest_mhd_dfs_len]
    
    concat_mhd_or_dfs_len = pd.DataFrame()
    for mhd_dfs_len in mhd_or_dfs_len_list:
        for key, mhd_len_df in mhd_dfs_len.items():
            #print(key)
            # use method 'concat' to append len_df to final_len_df
            concat_mhd_or_dfs_len = pd.concat([concat_mhd_or_dfs_len, mhd_len_df], axis=0, sort=False)
    
    ############# lhd_or_dfs ###############
    lhd_or_dfs_len_list = [or_bind_lhd_dfs_len, 
                          or_agon_G_GTP_lhd_dfs_len, 
                          or_agon_G_cAMP_lhd_dfs_len, 
                          or_agon_G_Ca_lhd_dfs_len, 
                          or_agon_B_arrest_lhd_dfs_len, 
                          or_antag_G_GTP_lhd_dfs_len,  
                          or_antag_B_arrest_lhd_dfs_len]

    concat_lhd_or_dfs_len = pd.DataFrame()
    for lhd_dfs_len in lhd_or_dfs_len_list:
        for key, lhd_len_df in lhd_dfs_len.items():
            #print(key)
            # use method 'concat' to append len_df to final_len_df
            concat_lhd_or_dfs_len = pd.concat([concat_lhd_or_dfs_len, lhd_len_df], axis=0, sort=False)

    ############# mhd_gpcr_dfs ###############
    mhd_gpcr_dfs_len_list = [gpcr_bind_mhd_dfs_len, 
                            gpcr_agon_G_GTP_mhd_dfs_len, 
                            gpcr_agon_G_cAMP_mhd_dfs_len, 
                            gpcr_agon_G_Ca_mhd_dfs_len, 
                            gpcr_agon_B_arrest_mhd_dfs_len, 
                            gpcr_antag_G_GTP_mhd_dfs_len,  
                            gpcr_antag_G_Ca_mhd_dfs_len,
                            gpcr_antag_B_arrest_mhd_dfs_len]

    concat_mhd_gpcr_dfs_len = pd.DataFrame()
    for mhd_len_dfs in mhd_gpcr_dfs_len_list:
        for key, mhd_len_df in mhd_len_dfs.items():
            concat_mhd_gpcr_dfs_len = pd.concat([concat_mhd_gpcr_dfs_len, mhd_len_df], axis=0, sort=False)

    ############# lhd_gpcr_dfs ###############
    lhd_gpcr_dfs_len_list = [gpcr_bind_lhd_dfs_len, 
                            gpcr_agon_G_GTP_lhd_dfs_len, 
                            gpcr_agon_G_cAMP_lhd_dfs_len, 
                            gpcr_agon_G_Ca_lhd_dfs_len, 
                            gpcr_agon_B_arrest_lhd_dfs_len, 
                            gpcr_antag_G_GTP_lhd_dfs_len,  
                            gpcr_antag_G_Ca_lhd_dfs_len,
                            gpcr_antag_B_arrest_lhd_dfs_len]

    concat_lhd_gpcr_dfs_len = pd.DataFrame()
    for lhd_len_dfs in lhd_gpcr_dfs_len_list:
        for key, lhd_len_df in lhd_len_dfs.items():
            concat_lhd_gpcr_dfs_len = pd.concat([concat_lhd_gpcr_dfs_len, lhd_len_df], axis=0, sort=False)

    return concat_mhd_or_dfs_len, concat_mhd_gpcr_dfs_len, concat_lhd_or_dfs_len, concat_lhd_gpcr_dfs_len

if __name__ == "__main__":

    GPCR_dfs, GPCR_chemblids = read_data()
    hhd_or_dfs_len, hhd_gpcr_dfs_len = run_hhd(GPCR_dfs, GPCR_chemblids)
    mhd_or_dfs_len, mhd_gpcr_dfs_len, lhd_or_dfs_len, lhd_gpcr_dfs_len = run_mhd_lhd(GPCR_dfs, GPCR_chemblids)

    hhd_or_dfs_len.to_csv(os.path.join(CAT_DATA_DIR, 'hhd_or_dfs_len.csv'), index=False)
    hhd_gpcr_dfs_len.to_csv(os.path.join(CAT_DATA_DIR, 'hhd_gpcr_dfs_len.csv'), index=False)
    mhd_or_dfs_len.to_csv(os.path.join(CAT_DATA_DIR, 'mhd_or_dfs_len.csv'), index=False)
    mhd_gpcr_dfs_len.to_csv(os.path.join(CAT_DATA_DIR, 'mhd_gpcr_dfs_len.csv'), index=False)
    lhd_or_dfs_len.to_csv(os.path.join(CAT_DATA_DIR, 'lhd_or_dfs_len.csv'), index=False)
    lhd_gpcr_dfs_len.to_csv(os.path.join(CAT_DATA_DIR, 'lhd_gpcr_dfs_len.csv'), index=False)
