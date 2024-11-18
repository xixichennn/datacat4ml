"""
mainly adopted from MoleculeACE

This module contains functions for hyperparameter optimization using Bayesian optimization.
- def count_hyperparam_combinations(hyperparameters: Dict[str, List[Union[float, str, int]]])
- def dict_to_search_space(hyperparams: Dict[str, List[Union[float, str, int]]])
- def cross_validate(model, x, y, n_folds: int = 5, early_stopping: int = 10, task = 'regression', **hyperparameters)

- class BayesianOptimization4reg
- class BayesianOptimization4cls
"""

import os
import sys

from typing import Dict, List, Union
from skopt.space.space import Categorical
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml
from yaml import dump, load, Loader
import json

from skopt import gp_minimize
from skopt.utils import use_named_args

from datacat4ml.Scripts.model_dev.metrics import *




#======================== utility functions ========================#
def get_config(file: str):
    """ Load a yml config file"""
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    return config

def write_config(filename: str, args: dict):
    """ Write a dictionary to a .yml file"""
    args = {k: v.item() if isinstance(v, np.generic) else v for k, v in args.items()}
    with open(filename, 'w') as file:
        documents = dump(args, file)

def count_hyperparam_combinations(hyperparameters: Dict[str, List[Union[float, str, int]]]):
    from itertools import product # `product` function is used to compute the Cartesian product of input iterables.
    return list(product(*[v for k, v in hyperparameters.items()])) # select only the values from the dict, 
                                                                   # `*` passes the values as argments to the function `product`
                                                                   # `product` returns an iteator of tuples representing the combinations

def dict_to_search_space(hyperparams: Dict[str, List[Union[float, str, int]]]):
    """  
    This function transforms a dictionary of hyperparameter options into a list of Categorical objects. 
    Each Categorical object represents a key from the dictionary and contains all the possible categories (values) associated with that key. 
    The resulting list of Categorical objects can be used as a search space for hyperparameter optimization algorithms or grid search.
    """
    return [Categorical(categories=list(v), name=k) for k, v in hyperparams.items()]

def cross_validate(model, x, y, n_folds: int = 5, early_stopping: int = 10, task = 'reg', **hyperparameters):
    """ cross validation function for hyperparameter optimization
    :param model: model class (defined in yuml.models.ml)
    :param x: input data
    :param y: labels
    :param n_folds: number of folds for cross validation
    :param early_stopping: number of epochs to wait for improvement before stopping training
    :param task: (string)classification or regression
    """
    ss = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)
    cutoff = np.median(y)
    labels = [0 if i < cutoff else 1 for i in y]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]

    rmse_scores = []
    accuracy_scores = []
    epochs = []
    for split in tqdm(splits):

        print("Model:", model, model.__name__)

        print("Starting a new fold")
        try:
            # Convert numpy types to regular python type (this bug cost me ages)
            hyperparameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyperparameters.items()}
            print("Hyperparameters:", hyperparameters)

            print("Creating the model...")
            f = model(task=task, **hyperparameters)
            print("Model created:", f)


            x_tr_fold = [x[i] for i in split['train_idx']] if type(x) is list else x[split['train_idx']]
            x_val_fold = [x[i] for i in split['val_idx']] if type(x) is list else x[split['val_idx']]

            y_tr_fold = [y[i] for i in split['train_idx']] if type(y) is list else y[split['train_idx']]
            y_val_fold = [y[i] for i in split['val_idx']] if type(y) is list else y[split['val_idx']]

            print("Training the model...")
            f.train(x_tr_fold, y_tr_fold, x_val_fold, y_val_fold, early_stopping)
            print("Model training completed.")

            print("Testing the model...")
            pred = f.predict(x_val_fold)
            if task == 'cls':
                pred_proba = f.predict_proba(x_val_fold)
            print("Model testing completed.")

            if task == 'reg':
                rmse_scores.append(calc_rmse(pred, y_val_fold))
            elif task == 'cls':
                accuracy_scores.append(calc_accuracy(pred, y_val_fold))

            if f.epoch is not None:
                if 'epochs' in hyperparameters:
                    if hyperparameters['epochs'] > f.epoch:
                        f.epoch = f.epoch - early_stopping
                epochs.append(f.epoch)

            del f
            torch.cuda.empty_cache()

        except Exception as e:
            print("Exception occurred during cross-validation:", str(e))
            # Handle the exception or re-raise it if necessary

    if len(epochs) > 0:
        epochs = int(sum(epochs)/len(epochs))
    else:
        epochs = None

    if task == 'reg':
        return sum(rmse_scores)/len(rmse_scores), epochs
    elif task == 'cls':
        return sum(accuracy_scores)/len(accuracy_scores), epochs
    
#======================== main function ========================#
class BayesianOptimization4reg:
    def __init__(self, model, task = 'reg'):
        """ Init the class with a trainable model. The model class should contain a train(), test(), predict() function
        and be initialized with its hyperparameters """
        self.task = task
        self.best_rmse = 100
        self.history = []
        self.model = model
        self.results = None

    def best_param(self):
        if len(self.history) is not None:
            return self.history[[i[0] for i in self.history].index(min([i[0] for i in self.history]))][1]

    def optimize(self, x, y, dimensions: Dict[str, List[Union[float, str, int]]], x_val=None, y_val=None,
                 n_folds: int = 5, early_stopping: int = 10, n_calls: int = 50, min_init_points: int = 10):

        # Prevent too many calls if there aren't as many possible hyperparameter combi's as calls (10 in the min calls)

        # Load dimensions from YAML file
        with open(dimensions, 'r') as f:
            dimensions = yaml.safe_load(f)

        dimensions = {k: [v] if type(v) is not list else v for k, v in dimensions.items()}
        combinations = count_hyperparam_combinations(dimensions)
        if len(combinations) < n_calls:
            n_calls = len(combinations)
        if len(combinations) < min_init_points:
            min_init_points = len(combinations)

        dimensions = dict_to_search_space(dimensions)

        # Objective function for Bayesian optimization
        @use_named_args(dimensions=dimensions)
        def objective(**hyperparameters):

            epochs = None
            try:
                print(f"Current hyperparameters: {hyperparameters}")
                if n_folds > 0:
                    rmse, epochs = cross_validate(self.model, x, y, n_folds=n_folds, early_stopping=early_stopping, task=self.task,
                                                  **hyperparameters)
                else:
                    f = self.model(**hyperparameters)
                    f.train(x, y, x_val, y_val)
                    epochs = f.epoch
                    pred = f.predict(x_val)
                    rmse = calc_rmse(pred, y_val)

            except:  # If this combination of hyperparameters fails, we use a dummy rmse that is worse than the best
                print(">>  Failed")
                rmse = self.best_rmse + 1

            if rmse < self.best_rmse:
                self.best_rmse = rmse

            if epochs is not None:
                hyperparameters['epochs'] = epochs

            self.history.append((rmse, hyperparameters))

            return rmse

        # Perform Bayesian hyperparameter optimization with 5-fold cross-validation
        self.results = gp_minimize(func=objective,
                                   dimensions=dimensions,
                                   acq_func='EI',  # expected improvement
                                   n_initial_points=min_init_points,
                                   n_calls=n_calls,
                                   verbose=True)

    def plot_progress(self, filename: str):
        import matplotlib.pyplot as plt
        rmse_values = [i[0] for i in self.history] # rmse value for each attempt
        min_rmse = sorted([i[0] for i in self.history], reverse=True) # the minimal rmse value among all attempts
        tries = list(range(len(self.history)))

        plt.figure()
        plt.plot(tries, rmse_values, label='history')
        # plot the best y value but ignore the x value
        plt.plot(tries, min_rmse, label='best')
        plt.xlabel("Optimization attempts")
        plt.ylabel("RMSE")
        plt.legend(loc="upper right")
        plt.savefig(f"{filename}")

    def history_to_csv(self, filename: str):
        results = []
        for i in self.history:
            d = {'rmse': i[0]}
            d.update(i[1])
            results.append(d)

        results = {k: [dic[k] for dic in results] for k in results[0]}

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

    def save_config(self, filename: str):
        write_config(filename, self.best_param())


class BayesianOptimization4cls:
    def __init__(self, model, task='cls'):
        """ Init the class with a trainable model. The model class should contain a train(), test(), predict() function
        and be initialized with its hyperparameters """
        self.task = task
        self.best_accuracy= -2 # Initial the best accuracy score to a very low value, thus the first accuracy score will always be better
        self.history = []
        self.model = model
        self.results = None

    def best_param(self):
        """
        returns the hyperparameters corresponding to the best accuracy score(the max is the the best)
        """
        if len(self.history) is not None:
            return self.history[[i[0] for i in self.history].index(max([i[0] for i in self.history]))][1]

    def optimize(self, x, y, dimensions: Dict[str, List[Union[float, str, int]]], x_val=None, y_val=None,
                 n_folds: int = 5, early_stopping: int = 10, n_calls: int = 50, min_init_points: int = 10):

        # Prevent too mant calls if there aren't as many possible hyperparameter combi's as calls (10 in the min calls)

        # Load dimensions from YAML file
        with open(dimensions, 'r') as f:
            dimensions = yaml.safe_load(f)
            
        dimensions = {k: [v] if type(v) is not list else v for k, v in dimensions.items()}
        combinations = count_hyperparam_combinations(dimensions)
        if len(combinations) < n_calls:
            n_calls = len(combinations)
        if len(combinations) < min_init_points:
            min_init_points = len(combinations)

        dimensions = dict_to_search_space(dimensions)

        # Objective function for Bayesian optimization
        @use_named_args(dimensions=dimensions)
        def objective(**hyperparameters):

            epochs = None
            try:
                print(f"Current hyperparameters: {hyperparameters}")
                if n_folds > 0:
                    print(">>  Cross-validating with {} folds".format(n_folds))
                    accuracy, epochs = cross_validate(self.model, x, y, n_folds=n_folds, early_stopping=early_stopping,task=self.task,
                                                  **hyperparameters)
                    print(f"Done {n_folds}-fold cross-validation. accuracy: {accuracy}")
                else:
                    f = self.model(**hyperparameters)
                    f.train(x, y, x_val, y_val)
                    epochs = f.epoch
                    pred = f.predict(x_val)
                    pred_proba = f.predict_proba(x_val)
                    accuracy = calc_accuracy(pred, y_val)

            except:  
                print(">>  Failed")
                # If this combination of hyperparameters fails, we use a dummy accuracy that is worse than the best
                accuracy = self.best_accuracy - 1

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy

            if epochs is not None:
                hyperparameters['epochs'] = epochs

            self.history.append((accuracy, hyperparameters))

            return accuracy

        # Perform Bayesian hyperparameter optimization with 5-fold cross-validation
        self.results = gp_minimize(func=objective,
                                   dimensions=dimensions,
                                   acq_func='EI',  # expected improvement
                                   n_initial_points=min_init_points,
                                   n_calls=n_calls,
                                   verbose=True)

    def plot_progress(self, filename: str):
        import matplotlib.pyplot as plt
        accuracy_values = [i[0] for i in self.history] # accuracy value for each attempt
        max_accuracy = sorted([i[0] for i in self.history]) # the maximal accuracy value among all attempts
        tries = list(range(len(self.history)))

        plt.figure()
        plt.plot(tries, accuracy_values, label='history')
        # plot the best y value but ignore the x value
        plt.plot(tries, max_accuracy, label='best')
        plt.xlabel("Optimization attempts")
        plt.ylabel("accuracy")
        plt.legend(loc="upper right")
        plt.savefig(f"{filename}")

    def history_to_csv(self, filename: str):
        results = []
        for i in self.history:
            d = {'accuracy': i[0]}
            d.update(i[1])
            results.append(d)

        results = {k: [dic[k] for dic in results] for k in results[0]}

        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

    def save_config(self, filename: str):
        write_config(filename, self.best_param())