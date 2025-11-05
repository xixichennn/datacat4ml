"""
utilities for hyperparameter optimization using Optuna.
"""
import optuna
from tqdm import tqdm
import numpy as np
import json
from typing import List, Union, Dict, Any
import gc # to free memory
import torch

from datacat4ml.Scripts.model_dev.metrics import METRICS_FUNC
#======================== functions ======================================
def get_config(file:str):
    "load a json config file"
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    
    # convert the single value to list
    config = {k: [v] if type(v) is not list else v  for k, v in config.items()}

    return config

def write_hparams(file:str, args:dict):
    "write a dictionary to a json file"
    args = {k:v.item() if isinstance(v, np.generic) else v for k,v in args.items()}
    with open(file, 'w') as f:
        json.dump(args, f, indent=4)

def count_hparams_combinations(config:Dict[str, List[Union[float, str, int]]]):
    from itertools import product 
    values = [v if isinstance(v, list) else [v] for v in config.values()]

    combinations = list(product(*values))

    return len(combinations)

def set_hpspace(trial: optuna.trial.Trial, config: dict) -> Dict[str, Any]:

    "Get a hyperparameters set for the current trial by sampling from the config"

    hparams = {}
    for k, v in config.items():
        if isinstance(v, list):
            # optuna samples one value from this list for the current trial
            hparams[k] = trial.suggest_categorical(k, v)
        else:
            # if not a list, it's a fixed hyperparameter set for all trials
            hparams[k] = v

    return hparams

def cross_validate(hparams:Dict[str, Any], model, splits, x, y, metric='auroc', verbose=False,
                   early_stopping: int=10):
    """
    Run inner-loop cross-validation to evaluate a given hyperparameter space.

    params
    ------
    model: MLmodel
        a trainable model class
    hparams: dict
        a dictionary containing hyperparameters for the model. Note: not like config, each key-value pair in `hparams` is a key and a single value.
    splits: list of dictionaries with two keys, i.e. 'train_idx' and 'val_idx'
        e.g. [{'inner_train_idx': [2, 3, 4, 5, 6, 7], 'inner_valid_idx': [0, 1]},
              {'inner_train_idx': [0, 1, 3, 5, 6, 7], 'inner_valid_idx': [2, 4]}, 
              {'inner_train_idx': [0, 1, 2, 3, 4, 7], 'inner_valid_idx': [5, 6]}, 
              {'inner_train_idx': [0, 1, 2, 4, 5, 6, 7], 'inner_valid_idx': [3]}, 
              {'inner_train_idx': [0, 1, 2, 3, 4, 5, 6], 'inner_valid_idx': [7]}]
    x: list
        a list of input features for the picked outer training set. i.e. outer_x_train_pick
    y: list
        a list of labels for the picked outer training set. i.e. outer_y_train_pick
    """

    def _run_cv_folds(hparams, model, splits, x, y, metric, verbose):
        """Helper: run CV evaluation for a given metric function."""
        
        metric_func = METRICS_FUNC[metric]

        fold_scores, epochs = [], []

        for i, split in tqdm(enumerate(splits)):
            print(f"\nInner Fold {i+1}/{len(splits)} for {model.__name__}") if verbose else None

            print("Creating model...") if verbose else None
            f = model(**hparams)
            print("Model created.") if verbose else None

            train_idx = split['inner_train_idx']
            valid_idx = split['inner_valid_idx']

            x_tr = [x[j] for j in train_idx]
            x_val = [x[j] for j in valid_idx]
            y_tr = [y[j] for j in train_idx]
            y_val = [y[j] for j in valid_idx]

            print("Training model...") if verbose else None
            f.train(x_train=x_tr, y_train=y_tr)
            print("Model trained.") if verbose else None

            print("Validating model...") if verbose else None
            y_pred = f.predict(x_val)
            y_pred_proba = f.predict_proba(x_val)

            if metric == 'accuracy':
                fold_score = metric_func(y_val, y_pred)
            else:
                if len(set(y_tr)) == 1:
                    print(f"Only one class in y_tr. Unable to calculate {metric}, thus return NaN.")
                    fold_score = float('nan')
                else:
                    fold_score = metric_func(y_val, y_pred_proba)

            fold_scores.append(fold_score)
            print(f"Fold {i+1}: {metric}={fold_score:.4f}") if verbose else None
            print("Model validated.") if verbose else None

            # For deep learning models, we can track the number of epochs
            if getattr(f, "epoch", None) is not None:
                epochs.append(f.epoch)

            del f
            torch.cuda.empty_cache()
            gc.collect()

        mean_score = np.nanmean(fold_scores)
        avg_epoch = int(np.mean(epochs)) if epochs else None

        return mean_score, fold_scores, avg_epoch

    # First try: run cv folds using 'auroc' metric
    mean_score, fold_scores, avg_epoch = _run_cv_folds(hparams, model, splits, x, y, metric, verbose)
    print(f'Mean {metric} across folds: {mean_score:.4f}')
    print(f'np.isnan(mean_score): {np.isnan(mean_score)}') if verbose else None

    # if all fold scores are nan
    # Second try: switch to 'accuracy' metric
    if np.isnan(mean_score):
        print(f"All fold scores are NaN for metric '{metric}'.\n-->Switching to 'Accuracy'")
        mean_score, fold_scores, avg_epoch = _run_cv_folds(hparams, model, splits, x, y, 'accuracy', verbose)
        
    return mean_score, fold_scores, avg_epoch

def optuna_hpo(config, model, splits, x, y, metric='auroc', verbose=False):

    #def _objective(trial: optuna.trial.Trial, config, model, splits, x, y):
    def _objective(trial: optuna.trial.Trial):
        """
        Optuna objective function to maximize the mean_score from cross_validate.
        This function represents a single trial in the hyperparameter search space.
        """
        hparams = set_hpspace(trial, config)

        print(f"\n<Trial {trial.number}>")
        print(f'Sampled HParams: {hparams}')

        try:
            mean_score, _, _= cross_validate(hparams, model, splits, x, y, metric, verbose)
            if np.isnan(mean_score):
                print("NaN score -> assigning fallback score 0.0")
                mean_score = 0.0
        except Exception as e:
            print(f'Error during cross-validation for trial {trial.number}: {e}')
            mean_score = 0.0 # mark this trial as worst-performing

        return mean_score

    # define the objective wrapper
    #objective_wrapper = lambda trial: _objective(trial, config, model, splits, x, y)

    # perform optimization using optuna
    sampler = optuna.samplers.GridSampler(config)
    study = optuna.create_study(sampler=sampler, direction='maximize',
                                #storage=f"sqlite:///{os.path.join(ML_HP_DIR,'db.sqlite3')}",
                                #study_name=f'cv_inner_{model.__name__}'
                                )

    #study.optimize(objective_wrapper)
    study.optimize(_objective)

    return study.best_params, study.best_value