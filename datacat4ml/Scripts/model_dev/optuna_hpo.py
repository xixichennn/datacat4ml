import optuna

from datacat4ml.Scripts.model_dev.ml_models import MLmodel, RF, GB, SVM, KNN
from datacat4ml.Scripts.model_dev.metrics import calc_auroc, calc_auprc
from datacat4ml.const import RANDOM_SEED
from datacat4ml.const import ML_HP_DIR

from sklearn.model_selection import StratifiedShuffleSplit
import torch
from tqdm import tqdm

import os
import pandas as pd
import numpy as np
import json
from typing import List, Union, Dict, Any

#==============================================================================
# Optuna utilities
#==============================================================================
def get_config(file:str):
    "load a json config file"
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    
    # convert the single value to list
    config = {k: [v] if type(v) is not list else v  for k, v in config.items()}

    return config

def write_config(file:str, args:dict):
    "write a dictionary to a json file"
    args = {k:v.item() if isinstance(v, np.generic) else v for k,v in args.items()}
    with open(file, 'w') as f:
        json.dump(args, f, indent=4)

def count_hparams_combinations(config:Dict[str, List[Union[float, str, int]]]):
    from itertools import product 
    values = [v if isinstance(v, list) else [v] for v in config.values()]

    combinations = list(product(*values))

    return len(combinations)

def set_hpspace(trial: optuna.trial.Trial, config: dict) -> dict:

    "get a hyperparameters set for the current trial"

    hparams = {}
    for k, v in config.items():
        if type(v) is list:
            hparams[k] = trial.suggest_categorical(k, v)
        else:
            hparams[k] = v

    return hparams

def cross_validate(model, hparams:Dict[str, Any], x, y, n_folds: int=5, test_size: float=0.2, early_stopping: int=10):
    """
    For hyperparameter optimization, do cross validation function that receives FIXED haparams for the current trial

    params
    ------
    model: MLmodel
        a trainable model class
    hparams: dict
        a dictionary containing hyperparameters for the model
    x: list or np.array
        input features
    y: list or np.array
        target values
    n_folds: int
        number of folds for cross-validation
    test_size: float
        proportion of the dataset to include in the validation split
    early_stopping: int
        number of epochs with no improvement to stop training early
    """

    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=RANDOM_SEED)
    splits = []
    for i, j in sss.split(x, y):
        splits.append({'train_idx':i, 'val_idx':j})

    auroc = []
    epochs = []
    for i, split in tqdm(enumerate(splits)):
        print("Model:", model.__name__)
        print ("Starting fold:", i)

        print("Creating the model...")
        f = model(**hparams)
        print("Model created.")

        x_tr_fold = [x[i] for i in split['train_idx']] if type(x) is list else x[split['train_idx']]
        x_val_fold = [x[i] for i in split['val_idx']] if type(x) is list else x[split['val_idx']]
        y_tr_fold = [y[i] for i in split['train_idx']] if type(y) is list else y[split['train_idx']]
        y_val_fold = [y[i] for i in split['val_idx']] if type(y) is list else y[split['val_idx']]

        print("Training the model...")
        f.train(x_train=x_tr_fold, y_train=y_tr_fold)
        print("Model trained.")

        print('Validating the model...')
        y_pred = f.predict(x_val_fold)
        y_pred_proba = f.predict_proba(x_val_fold)
        print('Model validated.')

        auroc_score = calc_auroc(y_val_fold, y_pred_proba)
        print(f'Fold {i} AUROC: {auroc_score}')

        auroc.append(auroc_score)
        # for deep learning models,
        if f.epoch is not None:
            if 'epochs' in hparams.keys():
                if hparams['epochs'] > f.epoch:
                    f.epoch = f.epoch - early_stopping
            epochs.append(f.epoch)
        del f
        torch.cuda.empty_cache()

    if len(epochs) > 0:
        epochs = int(sum(epochs)/len(epochs)) # average epochs
    else:
        epochs = None  

    return sum(auroc)/len(auroc), epochs

class OptunaHPO:
    def __init__(self, model):
        """ Init the class with a trainable model. The model class should contain a train(), test(), predict() function
        and be initialized with its hyperparameters """
        self.best_auroc = 0
        self.model = model
        self.study = None # store the optuna study object
    
    def optimize(self, x, y, config:Dict[str, List[Union[float, str, int]]], n_folds:int=5, test_size:float=0.2, early_stopping:int=10):

        """
        params
        ------
        config: dict
            a dictionary containing hyperparameter names and their possible values. Different from the variable `hparams` which contains the FIXED hparams.
        """
        combinations = count_hparams_combinations(config)
        print(f'Number of hyperparameter combinations: {combinations}')

        def objective(trial):
            hparams = set_hpspace(trial, config)
            print(f'\nTrial {trial.number}:')
            epochs = None

            try:
                if n_folds > 1: # cross-validation
                    auroc, epochs = cross_validate(self.model, hparams, x, y, n_folds, test_size, early_stopping)
                else: # single train-validation split
                    f = self.model(**hparams)
                    f.train(x, y)
                    epochs = f.epoch
                    pred_proba = f.predict_proba(x)
                    auroc = calc_auroc(y, pred_proba)
                    del f
                    torch.cuda.empty_cache()
            except Exception as e:
                print(">> Failed")
                print("Error:", e)
                auroc = 0
            
            if auroc > self.best_auroc:
                self.best_auroc = auroc
            
            if epochs is not None:
                config['epochs'] = epochs

            return auroc

        # Perform optimization using Optuna
        sampler = optuna.samplers.GridSampler(config)
        self.study = optuna.create_study(sampler=sampler, direction='maximize',
                                         storage=f"sqlite:///{os.path.join(ML_HP_DIR,'db.sqlite3')}",
                                         study_name=f'cv_inner_{self.model.__name__}-2',
                                    )
        self.study.optimize(objective)