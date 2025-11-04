import os

from datacat4ml.const import Algos, Input_dirs, DESCRIPTORS, rmvSs, Aim_Spl_combinations
from datacat4ml.Scripts.model_dev.ml_models import RF, GB, SVM, KNN
from datacat4ml.Scripts.model_dev.ml_tune import get_config
from datacat4ml.Scripts.model_dev.ml_dataloader import MLData
from datacat4ml.Scripts.model_dev.ml_pipelines import PL_FUNCS
from datacat4ml.Scripts.data_prep.data_split.alnSplit_mldata import alignment_map, get_pd_cd_pairs, get_pfp_cfps_all
from datacat4ml.const import FEAT_HHD_OR_DIR, FEAT_MHD_effect_OR_DIR, FEAT_MHD_OR_DIR, FEAT_LHD_OR_DIR
from datacat4ml.const import ML_HP_DIR, ML_DIR

import argparse


#=======================================
# internal benchmark
#=======================================
algo_dict = {
    'RF': RF,
    'GB': GB,
    'SVM': SVM,
    'KNN': KNN
}

def int_bmk(bmk_file, algo_name, input_dir, descriptor, rmvS, aim_spl_combo, pl,
            save_config=False, save_model=False, verbose=True):
    """
    bmk_file: path to save benchmark results

    algo: ML algorithm function
    input_dir: directory containing input data files. E.g., FEAT_HHD_OR_DIR
    descriptor: descriptor to use. E.g., 'ECFP4'
    rmvS: remove stereosiblings level for data split. E.g., 0, 1
    aim: aim for data split. E.g., 'lo' for lead optimization, 'vs' for virtual screening
    spl: split strategy. E.g., 'rs_lo', 'rs_vs', 'cs', 'ch'
    pl: pipeline to use. E.g., 'holdout_cv', 'single_nested_cv', 'nested_cv', 'consensus_nested_cv'
    """

    if not os.path.isfile(bmk_file):
        with open(bmk_file, 'w') as f:
            f.write('algo,'
                    'ds_cat_level,rmvD,'
                    'f_prefix,target_chembl_id,effect,assay,standard_type,assay_chembl_id,'
                    'descriptor,SPL,rmvS,aim,spl,'
                    'ds_size,ds_size_level,threshold,percent_a,outer_n_fold,'
                    'pipeline,'
                    'auroc,auprc,balanced,kappa,bedroc\n'
            )
    count = 0

    algo = algo_dict[algo_name]
    print(f'algo: {algo.__name__}') if verbose else None

    #----config----
    config = get_config(os.path.join(ML_HP_DIR, f'{algo.__name__}.json'))
    print(f'config: \n{config}') if verbose else None
    #----fpath----
    # input_dir

    # rmvD
    rmvD = 1 # To avoid data leakage, only use rmvD1 datasets for ML model benchmarking.
    print(f'rmvD: {rmvD}') if verbose else None

    file_dir = os.path.join(input_dir, f'rmvD{rmvD}')
    files = os.listdir(file_dir)

    # file
    for f in files:
        print(f'file: {f}') if verbose else None
        fpath = os.path.join(file_dir, f)

        # ======> initialize MLData object <=======
        data = MLData(fpath)
        # identifiers in filename: part 1
        ds_cat_level = data.ds_cat_level
        #rmvD = data.rmvD
        f_prefix = data.f_prefix
        target_chembl_id = data.target_chembl_id
        effect = data.effect
        assay = data.assay
        standard_type = data.standard_type
        assay_chembl_id = data.assay_chembl_id

        #----- x ------
        # descriptor
        print(f'descriptor: {descriptor}') if verbose else None
                    
        #----- y -----
        #---- split columns ----
        # rmvS
        print(f'rmvS: {rmvS}') if verbose else None

        # aim and spl
        aim, spl = aim_spl_combo.split(',')
        print(f'aim: {aim}') if verbose else None

        print(f'split column: rmvS{rmvS}_{spl}') if verbose else None
                                
        # ==========> prepare data splits <==========
        data(descriptor, rmvS, aim, spl)
        # get data stats: part 2
        (ds_size,
         data_size_level,
         threshold,
         percent_a) = (data.data_size,
                       data.data_size_level,
                       data.threshold,
                       data.percent_a)
                        
        data.get_int_splits()
        outer_n_fold = data.outer_n_fold

        # pipeline
        func = PL_FUNCS[pl]
        print(f'Pipeline: {pl}, Function: {func.__name__}')
        metrics = func(config, algo, data, 
                       save_config=save_config, save_model=save_model, verbose=verbose,
                       SPL='int', position=None)
        count += 1
                                    
        with open(bmk_file, 'a') as f:
            f.write(f'{algo.__name__},'
                    f'{ds_cat_level},rmvD{rmvD},'
                    f'{f_prefix},{target_chembl_id},{effect},{assay},{standard_type},{assay_chembl_id},'
                    f'{descriptor},int,rmvS{rmvS},{aim},{spl},'
                    f'{ds_size},{data_size_level},{threshold},{percent_a},{outer_n_fold},'
                    f'{pl},'
                    f'{metrics["auroc"]},{metrics["auprc"]},{metrics["balanced"]},{metrics["kappa"]},{metrics["bedroc"]}\n'
                    )
    return count


#=======================================
# aligned benchmark
#=======================================
feat_dir_name_dict = {
    'hhd': FEAT_HHD_OR_DIR,
    'mhd_effect': FEAT_MHD_effect_OR_DIR,
    'mhd': FEAT_MHD_OR_DIR,
    'lhd': FEAT_LHD_OR_DIR
}

pd_cd_pairs = get_pd_cd_pairs(alignment_map)
pfp_cfps_all = get_pfp_cfps_all(rmvD=1)

def aln_bmk(bmk_file, algo_name, descriptor, rmvS, aim_spl_combo, pl,
            save_config=False, save_model=False, verbose=True):

    if not os.path.isfile(bmk_file):
        with open(bmk_file, 'w') as f:
            f.write('algo,'
                    'pd_cat_level,cd_cat_level,rmvD,'
                    'pf_prefix,pf_target_chembl_id,pf_effect,pf_assay,pf_standard_type,pf_assay_chembl_id,'
                    'cf_prefix,cf_target_chembl_id,cf_effect,cf_assay,cf_standard_type,cf_assay_chembl_id,'
                    'descriptor,SPL,rmvS,aim,spl,' 
                    'pf_ds_size,pf_ds_size_level,pf_threshold,pf_percent_a,pf_outer_n_fold,'
                    'cf_ds_size,cf_ds_size_level,cf_threshold,cf_percent_a,cf_outer_n_fold,'
                    'pipeline,'
                    'pf_auroc,pf_auprc,pf_balanced,pf_kappa,pf_bedroc'
                    'cf_auroc,cf_auprc,cf_balanced,cf_kappa,cf_bedroc\n'
            )

    count = 0
    algo = algo_dict[algo_name]
    print(f'algo: {algo.__name__}') if verbose else None

    #----config----
    config = get_config(os.path.join(ML_HP_DIR, f'{algo.__name__}.json'))
    print(f'config: \n{config}') if verbose else None
    #----fpath----
    # input_dir
    for pd_cd_pair, pfp_cfps_map in pfp_cfps_all.items():
        print(f'pd_cd_pair: {pd_cd_pair}\n') if verbose else None
        pd_cat_level, cd_cat_level = pd_cd_pair

        pf_path = os.path.join(feat_dir_name_dict[pd_cat_level], f'rmvD1')
        cf_path = os.path.join(feat_dir_name_dict[cd_cat_level], f'rmvD1')

        for pf_prefix, cf_prefixes in pfp_cfps_map.items():
            print(f'parent file: {pf_prefix}, \nchild files: \n{cf_prefixes}') if verbose else None
            print('---------------------------------') if verbose else None

            if len(cf_prefixes) == 0:
                print('No child files found, skip this pair.') if verbose else None
            else:
                for cf_prefix in cf_prefixes:
                    print(f'child file: {cf_prefix}') if verbose else None
            
                    # pf
                    pf = [f for f in os.listdir(pf_path) if f.startswith(pf_prefix)][0] # there is only one such file
                    pf_full_path = os.path.join(pf_path, pf)
                    # cf
                    cf = [f for f in os.listdir(cf_path) if f.startswith(cf_prefix)][0] # there is only one such file
                    cf_full_path = os.path.join(cf_path, cf)

                    # ======> initialize MLData object <=======
                    pf_data = MLData(pf_full_path)
                    cf_data = MLData(cf_full_path)

                    # identifiers in filename: part 1
                    (pf_target_chembl_id,
                     pf_effect,
                     pf_assay,
                     pf_standard_type,
                     pf_assay_chembl_id) = (pf_data.target_chembl_id,
                                            pf_data.effect,
                                            pf_data.assay,
                                            pf_data.standard_type,
                                            pf_data.assay_chembl_id
                    )

                    (cf_target_chembl_id,
                     cf_effect,
                     cf_assay,
                     cf_standard_type,
                     cf_assay_chembl_id) = (cf_data.target_chembl_id,
                                            cf_data.effect,
                                            cf_data.assay,
                                            cf_data.standard_type,
                                            cf_data.assay_chembl_id
                    )

                    #----- x ------
                    # descriptor
                    print(f'descriptor: {descriptor}') if verbose else None
                            
                    #----- y -----
                    #---- split columns ----
                    # rmvS
                    print(f'rmvS: {rmvS}') if verbose else None
                    # aim and spl
                    aim, spl = aim_spl_combo.split(',')
                    print(f'aim: {aim}') if verbose else None
                                    
                    print(f'split column: rmvS{rmvS}_{spl}') if verbose else None
                                        
                    # ==========> prepare data splits <==========
                    pf_data(descriptor, rmvS, aim, spl)
                    cf_data(descriptor, rmvS, aim, spl)

                    # get data stats: part 2
                    (pf_ds_size, 
                     pf_data_size_level, 
                     pf_threshold, 
                     pf_percent_a) = (pf_data.data_size, 
                                      pf_data.data_size_level, 
                                      pf_data.threshold, 
                                      pf_data.percent_a
                    )

                    (cf_ds_size, 
                     cf_data_size_level, 
                     cf_threshold, 
                     cf_percent_a) = (cf_data.data_size, 
                                      cf_data.data_size_level, 
                                      cf_data.threshold, 
                                      cf_data.percent_a
                    )

                    # =========== get split data =============
                    pf_data.get_aln_split_data(pf_prefix=None, cf_prefix=cf_prefix)
                    cf_data.get_aln_split_data(pf_prefix=pf_prefix, cf_prefix=None)

                    pf_outer_n_fold = pf_data.outer_n_fold
                    cf_outer_n_fold = cf_data.outer_n_fold

                    # pipeline
                    func = PL_FUNCS[pl]
                    print(f'Pipeline: {pl}, Function: {func.__name__}')
                    pf_metrics = func(config, algo, pf_data, 
                                      save_config=save_config, save_model=save_model, verbose=verbose, 
                                      SPL='aln', position='parent')
                    cf_metrics = func(config, algo, cf_data, 
                                      save_config=save_config, save_model=save_model, verbose=verbose, 
                                      SPL='aln', position='child')
                    count += 1
                                            
                    with open(bmk_file, 'a') as f:
                        f.write(f'{algo.__name__},' # algo: e.g. RF
                                f'{pd_cd_pair[0]},{pd_cd_pair[1]},1' # pd_cat_level, cd_cat_level:e.g. hhd,mhd
                                f'{pf_prefix},{pf_target_chembl_id},{pf_effect},{pf_assay},{pf_standard_type},{pf_assay_chembl_id},'# pf_prefix: e.g. CHEMBL233_None_None_Ki_None_hhd,
                                f'{cf_prefix},{cf_target_chembl_id},{cf_effect},{cf_assay},{cf_standard_type},{cf_assay_chembl_id}' # cf_prefix: e.g. CHEMBL233_antag_G-GTP_Ki_None_mhd
                                f'{descriptor},aln,rmvS{rmvS},{aim},{spl},'
                                f'{pf_ds_size},{pf_data_size_level},{pf_threshold},{pf_percent_a},{pf_outer_n_fold},'
                                f'{cf_ds_size},{cf_data_size_level},{cf_threshold},{cf_percent_a},{cf_outer_n_fold},'
                                f'{pl},'
                                f'{pf_metrics["auroc"]},{pf_metrics["auprc"]},{pf_metrics["balanced"]},{pf_metrics["kappa"]},{pf_metrics["bedroc"]},'
                                f'{cf_metrics["auroc"]},{cf_metrics["auprc"]},{cf_metrics["balanced"]},{cf_metrics["kappa"]},{cf_metrics["bedroc"]}\n'
                                )

#=======================================
# main
#=======================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ML benchmarks.')
    parser.add_argument('--bmk_type', type=str, choices=['int', 'aln'], required=True,
                        help='Type of benchmark to run: "int" for internal benchmark, "aln" for aligned benchmark.')
    parser.add_argument('--algo', type=str, required=True,
                        help='ML algorithm to use, e.g., RF, SVM, XGB.')
    parser.add_argument('--input_dir', type=str,
                        help='Input directory containing data files. Required for internal benchmark.')
    parser.add_argument('--descriptor', type=str, required=True,
                        help='Descriptor to use, e.g., ECFP4, MACCS.')
    parser.add_argument('--rmvS', type=int, choices=rmvSs, required=True,
                        help='Remove stereosiblings level for data split.')
    #parser.add_argument('--aim', type=str, choices=['lo', 'vs'], required=True,
    #                    help='Aim for the benchmark: "lo" for lower, "vs" for versus.')
    #parser.add_argument('--spl', type=str, required=True,
    #                    help='Data split strategy, e.g., rs_lo, rs_vs, cs, ch.')
    parser.add_argument('--aim_spl_combo', type=str, required=True,
                        help='Combination of aim and split strategy, e.g., "lo,rs_lo", "vs,rs_vs"')
    parser.add_argument('--pl', type=str, required=True,
                        help='Pipeline to use, e.g., holdout_cv, single_nested_cv, nested_cv, consensus_nested_cv.')
    args = parser.parse_args()

    # Run the specified benchmark
    if args.bmk_type == 'int':
        bmk_file=os.path.join(ML_DIR, 'ml_internal_benchmark.csv') 

        total_runs = int_bmk(bmk_file, args.algo, args.input_dir, args.descriptor, args.rmvS, args.aim_spl_combo, args.pl,
                             save_config=False, save_model=False, verbose=True)
        print(f'Total benchmark runs completed: {total_runs}')

    elif args.bmk_type == 'aln':
        bmk_file=os.path.join(ML_DIR, 'ml_aligned_benchmark.csv')
        total_runs = aln_bmk(bmk_file, args.algo, args.descriptor, args.rmvS, args.aim_spl_combo, args.pl,
                                save_config=False, save_model=False, verbose=True)
        print(f'Total benchmark runs completed: {total_runs}')
