import os
import joblib
import numpy as np

from datacat4ml.Scripts.model_dev.ml_dataloader import MLData
from datacat4ml.Scripts.model_dev.ml_tune import optuna_hpo, write_hparams
from datacat4ml.Scripts.model_dev.metrics import calc_auroc, calc_auprc, calc_balanced_acc, calc_cohen_kappa, calc_ml_bedroc
from datacat4ml.const import ML_HP_DIR, ML_MODEL_DIR


#====================== pipeline functions ======================
def calc_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate various metrics given the true labels, predicted labels, and predicted probabilities.
    """
    return {
        'auroc': calc_auroc(y_true, y_pred_proba),
        'auprc': calc_auprc(y_true, y_pred_proba),
        'balanced': calc_balanced_acc(y_true, y_pred),
        'kappa': calc_cohen_kappa(y_true, y_pred),
        'bedroc': calc_ml_bedroc(y_true, y_pred_proba) # Yu: to do: use deepcoy negatives
    }

def holdout_cv(config, model, data:MLData, 
               save_config=False, save_model=False, verbose=False,
               SPL:str=None, position:str=None, metric='auroc'):

    """
    Pipeline (1): simple CV with an independent test set.
    Perform simple cross-validation with hyperparameter optimization, and evaluate on the outer test set.

    params
    ------
    SPL: str
        Split a dataset internally or align to an external split. Options: 'int', 'aln'.
    position: str
        If SPL=='aln', specify whether the position is 'parent' or 'child'.
    metric: str
        Metric to optimize during hyperparameter optimization. Default is 'auroc'.

    returns
    -------
    metrics: dict
        A dictionary containing various evaluation metrics on the outer test set.
    """
    # x and y
    x = data.x
    y = data.y
    
    if SPL == 'int':
        inner_splits_pick = data.int_inner_splits_pick

        x_train_pick = data.int_outer_x_train_pick
        y_train_pick = data.int_outer_y_train_pick
        x_test_pick = data.int_outer_x_test_pick
        y_test_pick = data.int_outer_y_test_pick
    elif SPL == 'aln':
        if position == 'parent':
            inner_splits_pick = data.pf_aln_inner_splits_pick

            x_train_pick = data.pf_aln_outer_x_train_pick
            y_train_pick = data.pf_aln_outer_y_train_pick
            x_test_pick = data.pf_aln_outer_x_test_pick
            y_test_pick = data.pf_aln_outer_y_test_pick
        elif position == 'child':
            inner_splits_pick = data.cf_aln_inner_splits

            x_train_pick = data.cf_aln_outer_x_train_pick
            y_train_pick = data.cf_aln_outer_y_train_pick
            x_test_pick = data.cf_aln_outer_x_test_pick
            y_test_pick = data.cf_aln_outer_y_test_pick

    # 2. HPO using the picked outer training set
    best_hparams, best_mean_score = optuna_hpo(config, model, inner_splits_pick, x, y, metric, verbose)
    print(f'\nBest Hparams: {best_hparams}\n')

    # 3. Train the model with the best_hparams, and evaluate on the outer test set.
    print(f'train model ...') if verbose else None
    f = model(**best_hparams)
    f.train(x_train=x_train_pick, y_train=y_train_pick)

    y_pred = f.predict(x_test_pick)
    y_pred_proba = f.predict_proba(x_test_pick) 
    
    # calculate metrics
    metrics = calc_metrics(y_test_pick, y_pred, y_pred_proba)

    # save hparams and model
    ds_path = data.ds_path # e.g. feat_mhd_or

    descriptor = data.descriptor
    aim = data.aim
    rmvS = data.rmvS
    spl = data.spl
    
    pipeline = 'HoldoutCV'

    f_prefix = data.f_prefix

    save_name = f'{model.__name__}_{descriptor}_{aim}_rmvS{rmvS}_{spl}_{pipeline}_{f_prefix}'

    if save_config: 
        config_path = os.path.join(ML_HP_DIR, ds_path.replace('feat_', ''))
        os.makedirs(config_path, exist_ok=True)
        write_hparams(os.path.join(config_path, f'{save_name}.json'), best_hparams)

    if save_model:
        model_path = os.path.join(ML_MODEL_DIR, ds_path.replace('feat_', ''))
        os.makedirs(model_path, exist_ok=True)
        with open(os.path.join(model_path, f'{save_name}.joblib'), 'wb') as handle:
            joblib.dump(f, handle)

    return metrics

def single_nested_cv(config, model, data:MLData, 
                     save_config=False, save_model=False, verbose=False,
                     SPL:str=None, position:str=None, metric='auroc'):
    """
    Pipeline (2): Nested CV with just one fold for tuning.

    params
    ------
    SPL: str
        Split a dataset internally or align to an external split. Options: 'int', 'aln'.
    position: str
        If SPL=='aln', specify whether the position is 'parent' or 'child'.
    metric: str
        Metric to optimize during hyperparameter optimization. Default is 'auroc'.

    returns
    -------
    metrics: dict
        A dictionary containing various evaluation metrics on the outer test set.
    """
    # x and y
    x = data.x
    y = data.y
    
    if SPL == 'int':
        inner_splits_pick = data.int_inner_splits_pick
        outer_splits = data.int_outer_splits

    elif SPL == 'aln':
        if position == 'parent':
            inner_splits_pick = data.pf_aln_inner_splits_pick
            outer_splits = data.pf_aln_outer_splits

        elif position == 'child':
            inner_splits_pick = data.cf_aln_inner_splits_pick
            outer_splits = data.cf_aln_outer_splits

    # 2. HPO: tune only on the picked outer training set
    best_hparams, best_mean_score = optuna_hpo(config, model, inner_splits_pick, x, y, metric, verbose)
    print(f'\nBest Hparams: {best_hparams}')

    # Optional: save hparams
    ds_path = data.ds_path # e.g. feat_mhd_or

    descriptor = data.descriptor
    aim = data.aim
    rmvS = data.rmvS
    spl = data.spl

    pipeline = 'SingleNestedCV'

    f_prefix = data.f_prefix
    
    save_name = f'{model.__name__}_{descriptor}_{aim}_rmvS{rmvS}_{spl}_{pipeline}_{f_prefix}'

    if save_config: 
        config_path = os.path.join(ML_HP_DIR, ds_path.replace('feat_', ''))
        os.makedirs(config_path, exist_ok=True)
        write_hparams(os.path.join(config_path, f'{save_name}.json'), best_hparams)


    # 3. Outer CV: train and evaluate on each outer fold using the best_hparams
    aurocs, auprcs, balanceds, kappas, bedrocs = [], [], [], [], []
    
    for i, outer_split in enumerate(outer_splits):
        print(f'\nOuter Fold {i+1}/{len(outer_splits)}')

        # train data for the current outer fold
        x_train = [data.x[j] for j in outer_split['outer_train_idx']]
        y_train = [data.y[j] for j in outer_split['outer_train_idx']]

        # test data for the current outer fold
        x_test = [data.x[j] for j in outer_split['outer_test_idx']]
        y_test = [data.y[j] for j in outer_split['outer_test_idx']]

        # Train the model with the best_hparams, and evaluate on the outer test set.
        print(f'train model ...') if verbose else None
        f = model(**best_hparams)
        f.train(x_train=x_train, y_train=y_train)

        y_pred = f.predict(x_test)
        y_pred_proba = f.predict_proba(x_test)

        # calculate metrics
        metrics = calc_metrics(y_test, y_pred, y_pred_proba)

        aurocs.append(metrics['auroc'])
        auprcs.append(metrics['auprc'])
        balanceds.append(metrics['balanced'])
        kappas.append(metrics['kappa'])
        bedrocs.append(metrics['bedroc'])

        # Optional: save hparamsmodel for each fold
        if save_model:
            model_path = os.path.join(ML_MODEL_DIR, ds_path.replace('feat_', ''))
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, f'{save_name}_fold{i+1}.joblib'), 'wb') as handle:
                joblib.dump(f, handle)

    print(f'outer_fold_aurocs: {aurocs}\n'
        f'outer_fold_auprcs: {auprcs}\n'
        f'outer_fold_balanceds: {balanceds}\n'
        f'outer_fold_kappas: {kappas}\n'
        f'outer_fold_bedrocs: {bedrocs}\n'
    ) if verbose else None

    return {'auroc': np.nanmean(aurocs), # to ignore nan values
        'auprc': np.nanmean(auprcs),
        'balanced': np.nanmean(balanceds),
        'kappa': np.nanmean(kappas),
        'bedroc': np.nanmean(bedrocs)} #Yu: error line 1

def nested_cv(config, model, data:MLData, 
              save_config=False, save_model=False, verbose=False,
              SPL:str=None, position:str=None, metric='auroc'):
    """
    Pipeline (3): Full nested CV with per-fold tunning.

    params
    ------
    SPL: str
        Split a dataset internally or align to an external split. Options: 'int', 'aln'.
    position: str
        If SPL=='aln', specify whether the position is 'parent' or 'child'.
    
    returns
    -------
    metrics: dict
        A dictionary containing various evaluation metrics on the outer test set.
    """
    # x and y
    x = data.x
    y = data.y

    if SPL == 'int':
        outer_splits = data.int_outer_splits
        inner_splits_all = data.int_inner_splits_all

    elif SPL == 'aln':
        if position == 'parent':
            outer_splits = data.pf_aln_outer_splits
            inner_splits_all = data.pf_aln_inner_splits_all

        elif position == 'child':
            outer_splits = data.cf_aln_outer_splits
            inner_splits_all = data.cf_aln_inner_splits_all
    
    print(f'The length of outer_splits: {len(outer_splits)}')
    print(f'The length of inner_splits_all: {len(inner_splits_all)}')

    # 2. Iterate through all outer folds
    aurocs, auprcs, balanceds, kappas, bedrocs = [], [], [], [], []
    
    for i, outer_split in enumerate(outer_splits):
        print(f'\nOuter Fold {i+1}/{len(outer_splits)}') 

        # Get training data (for HPO and final model training)
        x_hpo = [data.x[j] for j in outer_split['outer_train_idx']]
        y_hpo = [data.y[j] for j in outer_split['outer_train_idx']]

        # Get test data (for final evaluation)
        x_test = [data.x[j] for j in outer_split['outer_test_idx']]
        y_test = [data.y[j] for j in outer_split['outer_test_idx']]

        # Get the inner splits corresponding to this outer fold
        splits_hpo = inner_splits_all[i]

        # A. HPO: Find best hparams for this outer fold
        best_hparams, _ = optuna_hpo(config, model, splits_hpo, x, y, metric, verbose)
        print(f'\nFold {i+1} Best Hparams: {best_hparams}') #if verbose else None

        # B. Train final model on the outer training set with best_hparams
        print(f'train model ...') if verbose else None
        f = model(**best_hparams)
        f.train(x_train=x_hpo, y_train=y_hpo)

        # C. Evaluate on the outer test set
        y_pred = f.predict(x_test)
        y_pred_proba = f.predict_proba(x_test)

        # calculate metrics
        metrics = calc_metrics(y_test, y_pred, y_pred_proba)

        aurocs.append(metrics['auroc'])
        auprcs.append(metrics['auprc'])
        balanceds.append(metrics['balanced'])
        kappas.append(metrics['kappa'])
        bedrocs.append(metrics['bedroc'])

        # Optional: save hparams and model for each fold
        ds_path = data.ds_path # e.g. feat_mhd_or # Set `config_path` and `model_path` based on `filepath`

        descriptor = data.descriptor
        aim = data.aim
        rmvS = data.rmvS
        spl = data.spl

        pipeline = 'NestedCV'

        f_prefix = data.f_prefix

        save_name = f'{model.__name__}_{descriptor}_{aim}_rmvS{rmvS}_{spl}_{pipeline}_{f_prefix}_fold{i+1}'

        if save_config: 
            config_path = os.path.join(ML_HP_DIR, ds_path.replace('feat_', ''))
            os.makedirs(config_path, exist_ok=True)
            write_hparams(os.path.join(config_path, f'{save_name}.json'), best_hparams)
        
        if save_model:
            model_path = os.path.join(ML_MODEL_DIR, ds_path.replace('feat_', ''))
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, f'{save_name}.joblib'), 'wb') as handle:
                joblib.dump(f, handle)

    print(f'outer_fold_aurocs: {aurocs}\n'
        f'outer_fold_auprcs: {auprcs}\n'
        f'outer_fold_balanceds: {balanceds}\n'
        f'outer_fold_kappas: {kappas}\n'
        f'outer_fold_bedrocs: {bedrocs}\n'
    ) if verbose else None

    return {'auroc': np.nanmean(aurocs),
        'auprc': np.nanmean(auprcs),
        'balanced': np.nanmean(balanceds),
        'kappa': np.nanmean(kappas),
        'bedroc': np.nanmean(bedrocs)}

def consensus_nested_cv(config, model, data:MLData, 
                        save_config=False, save_model=False, verbose=False,
                        SPL:str=None, position:str=None, metric='auroc'):
    """
    Pipeline (4): Consensus nested CV. Identify the most frequent optimal hparams across all inner loops.
    
    params
    ------
    SPL: str
        Split a dataset internally or align to an external split. Options: 'int', 'aln'.
    position: str
        If SPL=='aln', specify whether the position is 'parent' or 'child'.
    
    returns
    -------
    metrics: dict
        A dictionary containing various evaluation metrics on the outer test set.
    
    """
    # x and y
    x = data.x
    y = data.y
    
    if SPL == 'int':
        outer_splits = data.int_outer_splits
        inner_splits_all = data.int_inner_splits_all

    elif SPL == 'aln':
        if position == 'parent':
            outer_splits = data.pf_aln_outer_splits
            inner_splits_all = data.pf_aln_inner_splits_all

        elif position == 'child':
            outer_splits = data.cf_aln_outer_splits
            inner_splits_all = data.cf_aln_inner_splits_all


    print(f"\n--- Phase 1: Find consensus hyperparameters ---") if verbose else None
    all_best_hparams = []

    # A. Run HPO for all outer folds to find the optimal hparams
    for i, outer_split in enumerate(outer_splits):
        print(f'\nOuter Fold {i+1}/{len(outer_splits)}')
        
        # Get the inner splits corresponding to this outer fold
        splits_hpo = inner_splits_all[i]

        # HPO: Find best hparams for this outer fold
        best_hparams, _ = optuna_hpo(config, model, splits_hpo, x, y, metric, verbose)
        # Convert dict to tuple for counting
        hparams_tuple = tuple(sorted(best_hparams.items()))
        all_best_hparams.append(hparams_tuple)
        print(f'Fold {i+1} Best Hparams: {best_hparams}') #if verbose else None

    # B Identify the consensus(most frequent) hparams 
    from collections import Counter
    hparam_counts = Counter(all_best_hparams)
    # The most common hyperparameter set (as a tuple)
    consensus_hparams_tuple = hparam_counts.most_common(1)[0][0]
    # Convert back to dict
    consensus_hparams = dict(consensus_hparams_tuple)
    print(f'\nConsensus Hyperparameters: {consensus_hparams}')
    
    # Optimal: store the best hparams for saving later
    ds_path = data.ds_path # e.g. feat_mhd_or 

    descriptor = data.descriptor
    aim = data.aim
    rmvS = data.rmvS
    spl = data.spl

    pipeline = 'ConsensusNestedCV'
    
    f_prefix = data.f_prefix

    save_name = f'{model.__name__}_{descriptor}_{aim}_rmvS{rmvS}_{spl}_{pipeline}_{f_prefix}'

    if save_config: 
        config_path = os.path.join(ML_HP_DIR, ds_path.replace('feat_', ''))
        os.makedirs(config_path, exist_ok=True)
        write_hparams(os.path.join(config_path, f'{save_name}.json'), best_hparams)


    # 2. Outer CV: Evaluate using the consensus_hparams across all outer folds
    print(f"\n--- Phase 2: Outer cross-validation with consensus hparams") if verbose else None
    aurocs, auprcs, balanceds, kappas, bedrocs = [], [], [], [], []

    for i, outer_split in enumerate(outer_splits):
        print(f'\nOuter Fold {i+1}/{len(outer_splits)}')

        # train data for the current outer fold
        x_train = [data.x[j] for j in outer_split['outer_train_idx']]
        y_train = [data.y[j] for j in outer_split['outer_train_idx']]

        # test data for the current outer fold
        x_test = [data.x[j] for j in outer_split['outer_test_idx']]
        y_test = [data.y[j] for j in outer_split['outer_test_idx']]

        # Train the model with the consensus_hparams, and evaluate on the outer test set.
        print(f'train model ...') if verbose else None
        f = model(**consensus_hparams)
        f.train(x_train=x_train, y_train=y_train)

        y_pred = f.predict(x_test)
        y_pred_proba = f.predict_proba(x_test)

        # calculate metrics
        metrics = calc_metrics(y_test, y_pred, y_pred_proba)

        aurocs.append(metrics['auroc'])
        auprcs.append(metrics['auprc'])
        balanceds.append(metrics['balanced'])
        kappas.append(metrics['kappa'])
        bedrocs.append(metrics['bedroc'])

        # Optional: save hparams and model for each fold
        if save_model:
            model_path = os.path.join(ML_MODEL_DIR, ds_path.replace('feat_', ''))
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, f'{save_name}_fold{i+1}.joblib'), 'wb') as handle:
                joblib.dump(f, handle)
                print('Done')

    print(f'outer_fold_aurocs: {aurocs}\n'
        f'outer_fold_auprcs: {auprcs}\n'
        f'outer_fold_balanceds: {balanceds}\n'
        f'outer_fold_kappas: {kappas}\n'
        f'outer_fold_bedrocs: {bedrocs}\n'
    ) if verbose else None

    return {
        'auroc': np.nanmean(aurocs), # to ignore nan values
        'auprc': np.nanmean(auprcs),
        'balanced': np.nanmean(balanceds),
        'kappa': np.nanmean(kappas),
        'bedroc': np.nanmean(bedrocs)
    }

PL_FUNCS = {
    'HoldoutCV': holdout_cv,
    'SingleNestedCV': single_nested_cv,
    'NestedCV': nested_cv,
    'ConsensusNestedCV': consensus_nested_cv
    }