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

def get_mldata_info(fpath, descriptor, aim_spl_combo, rmvS, 
                    cf_prefix=None, pf_prefix=None,
                    verbose=False):
    
    #============== intialize MLData object ==============
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

    # ----- x ------
    # descriptor
    print(f'descriptor: {descriptor}') if verbose else None

    # ----- y -----
    # aim and spl
    aim, spl = aim_spl_combo.split(',')
    print(f'aim: {aim}') if verbose else None # e.g., lo, vs
    print(f'spl: {spl}') if verbose else None # e.g., rs_lo, rs_vs, cs, ch

    # ========== prepare data splits ==========
    data(descriptor, aim, spl)
    # get data stats: part 2
    (ds_size,
     ds_size_level,
     threshold,
     percent_a) = (data.ds_size,
                   data.ds_size_level,
                   data.threshold,
                   data.percent_a)
    
    # ----- internal split columns -----
    # rmvS
    print(f'rmvS: {rmvS}') if verbose else None
    print(f'split column: rmvS{rmvS}_{spl}') if verbose else None
    data.get_int_splits(rmvS, verbose)

    int_outer_n_fold = data.int_outer_n_fold
    int_outer_x_test_pick = data.int_outer_x_test_pick
    print(f'int_outer_n_fold: {int_outer_n_fold}')

    # ----- aligned split columns -----
    if cf_prefix:
        data.get_aln_splits(pf_prefix=None, cf_prefix=cf_prefix)
        pf_aln_outer_n_fold = data.pf_aln_outer_n_fold
        pf_aln_outer_x_test_pick = data.pf_aln_outer_x_test_pick
        print(f'pf_aln_outer_n_fold: {pf_aln_outer_n_fold}')

    if pf_prefix:
        data.get_aln_splits(pf_prefix=pf_prefix, cf_prefix=None)
        cf_aln_outer_n_fold = data.cf_aln_outer_n_fold
        cf_aln_outer_x_test_pick = data.cf_aln_outer_x_test_pick
        print(f'cf_aln_outer_n_fold: {cf_aln_outer_n_fold}')

    return {'data': data,
            'ds_cat_level': ds_cat_level,
            'f_prefix': f_prefix,
            'target_chembl_id': target_chembl_id,
            'effect': effect,
            'assay': assay,
            'standard_type': standard_type,
            'assay_chembl_id': assay_chembl_id,
            'aim': aim,
            'spl': spl,
            'ds_size': ds_size,
            'ds_size_level': ds_size_level,
            'threshold': threshold,
            'percent_a': percent_a,
            'int_outer_n_fold': int_outer_n_fold,
            'pf_aln_outer_n_fold': pf_aln_outer_n_fold if cf_prefix else None,
            'cf_aln_outer_n_fold': cf_aln_outer_n_fold if pf_prefix else None,
            'int_outer_x_test_pick': int_outer_x_test_pick,
            'pf_aln_outer_x_test_pick': pf_aln_outer_x_test_pick if cf_prefix else None,
            'cf_aln_outer_x_test_pick': cf_aln_outer_x_test_pick if pf_prefix else None
            }


def int_bmk(bmk_file, algo_name, input_dir, 
            descriptor, aim_spl_combo, rmvS, 
            pl,
            save_config=False, save_model=False, verbose=False):
    """
    Benchmark ML models with internal splitting columns.

    params
    ------
    bmk_file: string
        path to save benchmark results
    algo_name: string
        the name of ML algorithm function
    input_dir: string
        directory containing input data files. E.g., FEAT_HHD_OR_DIR

    descriptor: string
        descriptor to use. E.g., 'ECFP4'
    rmvS: string
        remove stereosiblings level for data split. E.g., 0, 1
    aim_spl_combo: string
        combination of aim and split strategy, e.g., 'lo,rs_lo', 'vs,rs_vs'
    
    pl: string
        pipeline to use. E.g., 'holdout_cv', 'single_nested_cv', 'nested_cv', 'consensus_nested_cv'

    save_config: bool
        whether to save the used config file
    save_model: bool
        whether to save the trained models
    verbose: bool
        whether to print out messages

    returns
    -------
    count: int
        number of benchmark runs completed
    """

    if not os.path.isfile(bmk_file):
        with open(bmk_file, 'w') as f:
            f.write('algo,'
                    'ds_cat_level,rmvD,'
                    'f_prefix,target_chembl_id,effect,assay,standard_type,assay_chembl_id,'
                    'descriptor,SPL,aim,spl,rmvS,'
                    'ds_size,ds_size_level,threshold,percent_a,int_outer_n_fold,'
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
    print(f'rmvD: {rmvD}')

    file_dir = os.path.join(input_dir, f'rmvD{rmvD}')
    files = os.listdir(file_dir)

    # file
    for f in files:
        print(f'file: {f}')
        fpath = os.path.join(file_dir, f)

        # ======= get mldata info ========
        print(f'========>Getting MLData info')
        data_info = get_mldata_info(fpath, descriptor, aim_spl_combo, rmvS, verbose=verbose)

        required = (
            data_info['int_outer_n_fold'],
            data_info['int_outer_x_test_pick'],
        )
        if all(required):
            # pipeline
            func = PL_FUNCS[pl]

            print(f'\n========>Run pipeline {pl} ...') 
            metrics = func(config, algo, data_info['data'], 
                        save_config=save_config, save_model=save_model, verbose=verbose,
                        SPL='int', position=None)
            count += 1
                                        
            with open(bmk_file, 'a') as f:
                f.write(f'{algo.__name__},'
                        f'{data_info["ds_cat_level"]},rmvD{rmvD},'
                        f'{data_info["f_prefix"]},{data_info["target_chembl_id"]},{data_info["effect"]},{data_info["assay"]},{data_info["standard_type"]},{data_info["assay_chembl_id"]},'
                        f'{descriptor},int,{data_info["aim"]},{data_info["spl"]},rmvS{rmvS},'
                        f'{data_info["ds_size"]},{data_info["ds_size_level"]},{data_info["threshold"]},{data_info["percent_a"]},{data_info["int_outer_n_fold"]},'
                        f'{pl},'
                        f'{metrics["auroc"]},{metrics["auprc"]},{metrics["balanced"]},{metrics["kappa"]},{metrics["bedroc"]}\n'
                        )
        else:
            print(f'Skipping file {f} due to missing required data splits.')
    return count


#=======================================
# aligned benchmark
#=======================================
feat_dir_name_dict = {
    'hhd': FEAT_HHD_OR_DIR,
    'mhd-effect': FEAT_MHD_effect_OR_DIR,
    'mhd': FEAT_MHD_OR_DIR,
    'lhd': FEAT_LHD_OR_DIR
}

pd_cd_pairs = get_pd_cd_pairs(alignment_map)
pfp_cfps_all = get_pfp_cfps_all(rmvD=1)

def aln_bmk(bmk_file, algo_name, 
            descriptor, aim_spl_combo, rmvS, 
            pl,
            save_config=False, save_model=False, verbose=False):

    if not os.path.isfile(bmk_file):
        with open(bmk_file, 'w') as f:
            f.write('algo,'
                    'pd_cat_level,cd_cat_level,rmvD,'
                    'pf_prefix,pf_target_chembl_id,pf_effect,pf_assay,pf_standard_type,pf_assay_chembl_id,'
                    'cf_prefix,cf_target_chembl_id,cf_effect,cf_assay,cf_standard_type,cf_assay_chembl_id,'
                    'descriptor,SPL,aim,spl,rmvS,'
                    'pf_ds_size,pf_ds_size_level,pf_threshold,pf_percent_a,pf_aln_outer_n_fold,'
                    'cf_ds_size,cf_ds_size_level,cf_threshold,cf_percent_a,cf_aln_outer_n_fold,'
                    'pipeline,'
                    'pf_auroc,pf_auprc,pf_balanced,pf_kappa,pf_bedroc,'
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

        print(f'pd_cd_pair: {pd_cd_pair}\n')
        pd_cat_level, cd_cat_level = pd_cd_pair

        # rmvD
        rmvD = 1 # To avoid data leakage, only use rmvD1 datasets for ML model benchmarking.
        print(f'rmvD: {rmvD}') if verbose else None

        pf_path = os.path.join(feat_dir_name_dict[pd_cat_level], f'rmvD{rmvD}')
        cf_path = os.path.join(feat_dir_name_dict[cd_cat_level], f'rmvD{rmvD}')

        for pf_prefix, cf_prefixes in pfp_cfps_map.items():
            print(f'=================================\nparent file: {pf_prefix}, \nchild files: \n{cf_prefixes}')
            
            if len(cf_prefixes) == 0:  
                print('No child files found, skip this pair.') 
            else:
                for cf_prefix in cf_prefixes:
                    print(f'---------------------------------\nchild file: {cf_prefix}')

                    #if pd_cd_pair == ('hhd', 'lhd') and pf_prefix == 'CHEMBL236_None_None_Ki_None_hhd' and cf_prefix == 'CHEMBL236_bind_RBA_Ki_CHEMBL3887031_lhd': # for debugging
                        
                    # Yu: move the below lines indentation back
                    # pf
                    pf = [f for f in os.listdir(pf_path) if f.startswith(pf_prefix)][0] # there is only one such file
                    pf_full_path = os.path.join(pf_path, pf)
                    # cf
                    cf = [f for f in os.listdir(cf_path) if f.startswith(cf_prefix)][0] # there is only one such file
                    cf_full_path = os.path.join(cf_path, cf)

                    # ======> initialize MLData object <=======
                    print(f'\n========>pf: Getting MLData {pf_full_path.split("/")[-1]} ...')
                    pf_data_info = get_mldata_info(pf_full_path, descriptor, aim_spl_combo, rmvS,
                                                    cf_prefix=cf_prefix, pf_prefix=None,
                                                    verbose=verbose)
                    print(f'\n========>cf: Getting MLData {cf_full_path.split("/")[-1]} ...')
                    cf_data_info = get_mldata_info(cf_full_path, descriptor, aim_spl_combo, rmvS,
                                                    cf_prefix=None, pf_prefix=pf_prefix, verbose=verbose)

                    required = (
                        pf_data_info['pf_aln_outer_n_fold'],
                        cf_data_info['cf_aln_outer_n_fold'],
                        pf_data_info['pf_aln_outer_x_test_pick'],
                        cf_data_info['cf_aln_outer_x_test_pick']
                    )

                    if all(required):
                        # pipeline
                        func = PL_FUNCS[pl]

                        print(f'\n========>pf: Run pipeline {pl} ...')
                        pf_metrics = func(config, algo, pf_data_info['data'], 
                                        save_config=save_config, save_model=save_model, verbose=verbose, 
                                        SPL='aln', position='parent')
                        print(f'\n========>cf: Run pipeline {pl} ...')
                        cf_metrics = func(config, algo, cf_data_info['data'], 
                                        save_config=save_config, save_model=save_model, verbose=verbose, 
                                        SPL='aln', position='child')
                        count += 1

                        with open(bmk_file, 'a') as f:
                            f.write(f'{algo.__name__},'# algo: e.g. RF
                                    f'{pd_cd_pair[0]},{pd_cd_pair[1]},1,'# pd_cat_level, cd_cat_level:e.g. hhd,mhd
                                    f'{pf_data_info["f_prefix"]},{pf_data_info["target_chembl_id"]},{pf_data_info["effect"]},{pf_data_info["assay"]},{pf_data_info["standard_type"]},{pf_data_info["assay_chembl_id"]},'# pf_prefix: e.g. CHEMBL233_None_None_Ki_None_hhd,
                                    f'{cf_data_info["f_prefix"]},{cf_data_info["target_chembl_id"]},{cf_data_info["effect"]},{cf_data_info["assay"]},{cf_data_info["standard_type"]},{cf_data_info["assay_chembl_id"]},'# cf_prefix: e.g. CHEMBL233_antag_G-GTP_Ki_None_mhd
                                    f'{descriptor},aln,{pf_data_info["aim"]},{pf_data_info["spl"]},rmvS{rmvS},' # pf_data_info and cf_data_info have the same aim and spl
                                    f'{pf_data_info["ds_size"]},{pf_data_info["ds_size_level"]},{pf_data_info["threshold"]},{pf_data_info["percent_a"]},{pf_data_info["pf_aln_outer_n_fold"]},'
                                    f'{cf_data_info["ds_size"]},{cf_data_info["ds_size_level"]},{cf_data_info["threshold"]},{cf_data_info["percent_a"]},{cf_data_info["cf_aln_outer_n_fold"]},'
                                    f'{pl},'
                                    f'{pf_metrics["auroc"]},{pf_metrics["auprc"]},{pf_metrics["balanced"]},{pf_metrics["kappa"]},{pf_metrics["bedroc"]},'
                                    f'{cf_metrics["auroc"]},{cf_metrics["auprc"]},{cf_metrics["balanced"]},{cf_metrics["kappa"]},{cf_metrics["bedroc"]}\n'
                                    )
                    else:
                        print(f'Skipping pf {pf_prefix} and cf {cf_prefix} due to missing required data splits.')
    return count

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
    #parser.add_argument('--aim', type=str, choices=['lo', 'vs'], required=True,
    #                    help='Aim for the benchmark: "lo" for lower, "vs" for versus.')
    #parser.add_argument('--spl', type=str, required=True,
    #                    help='Data split strategy, e.g., rs_lo, rs_vs, cs, ch.')
    parser.add_argument('--aim_spl_combo', type=str, required=True,
                        help='Combination of aim and split strategy, e.g., "lo,rs_lo", "vs,rs_vs"')
    parser.add_argument('--rmvS', type=int, choices=rmvSs, required=True,
                    help='Remove stereosiblings level for data split.')
    parser.add_argument('--pl', type=str, required=True,
                        help='Pipeline to use, e.g., holdout_cv, single_nested_cv, nested_cv, consensus_nested_cv.')
    args = parser.parse_args()

    # Run the specified benchmark
    if args.bmk_type == 'int':
        bmk_fname = f'bmkML_int_{args.algo}_{args.descriptor}_{args.aim_spl_combo.split(",")[0]}_rmvS{args.rmvS}_{args.aim_spl_combo.split(",")[1]}_{args.pl}_{args.input_dir.split("/")[-1]}.csv'
        bmk_file=os.path.join(ML_DIR, bmk_fname)

        # Print the benchmark configuration
        print(f'algo: {args.algo}')
        config = get_config(os.path.join(ML_HP_DIR, f'{args.algo.__name__}.json'))
        print(f'config: \n{config}')
        print(f'input_dir: {args.input_dir}')
        print(f'rmvD: 1')
        print (f'descriptor: {args.descriptor}')
        print(f'aim_spl_combo: {args.aim_spl_combo}')
        print(f'rmvS: {args.rmvS}')
        print(f'pipeline: {args.pl}\n')

        total_runs = int_bmk(bmk_file, args.algo, args.input_dir, args.descriptor, args.aim_spl_combo, args.rmvS, args.pl,
                             save_config=False, save_model=True, verbose=False)
        print(f'Total benchmark runs completed: {total_runs}')

    elif args.bmk_type == 'aln':
        bmk_fname = f'bmkML_aln_{args.algo}_{args.descriptor}_{args.aim_spl_combo.split(",")[0]}_rmvS{args.rmvS}_{args.aim_spl_combo.split(",")[1]}_{args.pl}.csv'
        bmk_file=os.path.join(ML_DIR, bmk_fname)
        
        # Print the benchmark configuration
        print(f'algo: {args.algo}')
        print(f'rmvD: 1')
        print (f'descriptor: {args.descriptor}')
        print(f'aim_spl_combo: {args.aim_spl_combo}')
        print(f'rmvS: {args.rmvS}')
        print(f'pipeline: {args.pl}\n')

        total_runs = aln_bmk(bmk_file, args.algo, args.descriptor, args.aim_spl_combo, args.rmvS, args.pl,
                                save_config=False, save_model=True, verbose=False)
        print(f'Total benchmark runs completed: {total_runs}')
