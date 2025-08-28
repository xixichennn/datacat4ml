# conda env: datacat (python=3.8.2)
# for `utils.py`
# ==== all.py ====
from pathlib import Path
import os
import numpy as np
import pandas as pd


import json
from typing import Any, Iterable, List, Optional, Tuple, Union
from loguru import logger
import mlflow
import mlflow.entities

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
from torch.utils.data import Subset, RandomSampler, SequentialSampler, BatchSampler

from scipy.special import expit as sigmoid
from scipy import sparse

import datacat4ml.Scripts.model_dev.cl_models as cl_models
from datacat4ml.Scripts.model_dev.cl_models import DotProduct
from datacat4ml.Scripts.model_dev.dataloader import InMemoryClamp
from datacat4ml.Scripts.model_dev.metrics import swipe_threshold_sparse, top_k_accuracy


# ==== train.py ====
import argparse
#import mlflow
import random
import wandb
from time import time
#================================================================================================================
#                                       utils.py
#================================================================================================================
NAME2FORMATTER = {
    'verbose': bool,
    'seed': int,
    'gpu': int,
    'patience': int,
    'model': str,
    'embedding_size': int,
    'optimizer': str,
    'lr_ini': float,
    'l2': float, # L2 regularization, i.e weight decay.
    'dropout_input': float,
    'dropout_hidden': float,
    'loss_fun': str,
    'label_smoothing': float,
    'lr_factor': float,
    'batch_size': int,
    'warmup_epochs': int,
    'epoch_max': int, 
    'train_balanced': int,
    'train_subsample': float, # subsample the training data to this fraction for faster training.
    'beta': float,
    'assay_mode': str,
    'multitask_temperature': float, #?Yu: if not used later, remove
    'nonlinearity': str,
    'pooling_mode': str,
    'attempts': int, # not used in public version #?Yu: don't understand, remove?
    'tokenizer': str,
    'transformer': str,
    'norm': bool,
    'checkpoint': str,
    'hyperparams': str,
    'format': str,
    'f': str, #?Yu: file path?
    'support_set_size': int,
    'train_only_actives': bool,
    'random': int, 
    'dataset': str,
    'experiment': float,
    'split': str,
    'wandb': str,
    'compound_mode': str,
}

EVERY = 50000 # The frequency of printing a message (logging) during training to reduce verbosity.

def get_hparams(path, mode='logs', verbose=False):
    """
    Get hyperparameters from a path. 
    If mode is 'logs': uses path /params/* files from mlflow.
    If mode is 'json': loads in the file from Data/model_dev/hparams/default.json.

    Params
    ------
    path: str
        Path to the hyperparameters file.
    mode: str
        Mode of the hyperparameters file. Default is 'logs'.
    verbose: bool
        Be verbose if True.
    """
    if isinstance(path, str):
        path = Path(path)
    hparams = {}
    if mode =='logs':
        for fn in os.listdir(path/'params'):
            try:
                with open(path/f'params/{fn}') as f:
                    lines = f.readlines()
                    try:
                        hparams[fn] = NAME2FORMATTER.get(fn, str)(lines[0])
                    except:
                        hparams[fn] = None if len(lines)==0 else lines[0]
            except:
                pass
    elif mode == 'json':
        with open(path) as f:
            hparams = json.load(f)
    if verbose:
        logger.info("loaded hparams:\n", hparams)
    
    return hparams

def seed_everything(seed=70135):
    """ adopted from https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335"""
    import numpy as np
    import random
    import os
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_device(gpu=0, verbose=False):
    "Set device to gpu or cpu."
    if gpu == 'any':
        gpu = 0
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
    else:
        device = torch.device('cpu')
    if verbose:
        logger.info(f'Set device to {device}.')

    return device

def init_checkpoint(path, device, verbose=False):
    """
    load from path if path is not None, otherwise return empty dict.
    """
    if path is not None:
        if verbose:
            logger.info('Load checkpoint.')
        return torch.load(path, map_location=device)
    return {}

def get_mlflow_log_paths(run_info: mlflow.entities.RunInfo):
    """
    Return paths to the artifacts directory and the model weights from mlflow.
    """
    artifacts_dir = Path('mlruns', run_info.experiment_id, run_info.run_id, 'artifacts')
    checkpoint_file_path = artifacts_dir / 'checkpoint.pt'
    metrics_file_path = artifacts_dir / 'metrics.parquet'
    return artifacts_dir, checkpoint_file_path, metrics_file_path

class EarlyStopper:
    # adapted from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    # During training, it monitors the validation loss and stops training when the validation loss does not improve for a specified number of epochs (patience).
    # The logic here: the lower the validation loss, the better the model performance.
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience # number of epochs with no improvement after which training will be stopped
        self.min_delta = min_delta # minimum change to consider it an improvement
        self.counter = 0 # counter for the number of epochs with no improvement
        self.min_validation_loss = np.inf # the best (lowest) validation loss seen so far
        self.improved = False # flag to indicate if the last validation loss has improved

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.improved = True
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.improved = False
            if self.counter >= self.patience:
                return True
        return False       

def init_dp_model(
        compound_features_size: int,
        assay_features_size: int,
        hp: dict,
        verbose: bool = False
) -> DotProduct:
    """
    Initialize a DotProduct model instance based on the hyperparameters provided in `hp`.

    Params
    -------
    compound_features_size: int
        Input size of the compound encoder.
    assay_features_size: int
        Input size of the assay encoder.
    hp: dict
        Hyperparameters.
    verbose: bool
        Be verbose if True.

    Returns
    -------
    :class:`DotProduct`
        Model instance.
    """
    if verbose:
        logger.info(f'Initialize "{hp["model"]}" model.')

    init_dict = hp.copy() # copy hp to avoid mutate the original.
    init_dict.pop('embedding_size') # remove the embedding size, since it has to be provided as positional argument. If not removed, the `getattr` will get two `embedding_size` arguments, one positional and one keyword, which will cause an error.

    # For getattr(clamp.models, hp['model']) to work, the class must be exposed at the package level in /clamp/models/__init__.py. 
    # Typically, __init__.py will import selected classes from those submodules, making them accessible like clamp.models.MyModelClass.
    model = getattr(cl_models, hp['model'])( #?Yu model_def: class, hp['model']: the specified attribute. 
        compound_features_size=compound_features_size,
        assay_features_size=assay_features_size,
        embedding_size=hp['embedding_size'],
        **init_dict)

    if wandb.run:
        wandb.watch(model, log_freq=100, log_graph=(True))  # Log model weights and gradients for visualization in wandb, generate and log the computational graph automatically.

    return model

def filter_dict(dict_to_filter, thing_with_kwargs):
    """
    Examine the callable(`things_with_kwargs`, e.g. function/class) and inspect its signature to determine what keyword arguments it accepts.
    Then, filter the `dict_to_filter` to only include those keys that are valid arguments.
    modified from https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function
    """
    import inspect # a python standard library
    sig = inspect.signature(thing_with_kwargs) # get the list of valid argument names for the function or class `thing_with_kwargs`.
    filter_keys =[p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD] # only `POSITIONAL_OR_KEYWORD` parameters are considered.
    inters = set(dict_to_filter.keys()).intersection(filter_keys) # do filter

    return {k:dict_to_filter[k] for k in inters}

def init_optimizer(model, hp, verbose=False):
    """
    Initialize optimizer.
    """
    if verbose:
        logger.info(f"Trying to initialize '{hp['optimizer']}' optimizer from torch.optim.")
    
    # rename 'lr_ini' to 'lr', and 'l2' to 'weight_decay' to match the PyTorch optimizer's expected argument names.
    hp['lr'] = hp.pop('lr_ini') 
    hp['weight_decay'] = hp.pop('l2') 
    optimizer = getattr(torch.optim, hp['optimizer']) # fetch the optimizer class (e.g., `Adam`)
    filtered_dict = filter_dict(hp, optimizer) # remove any keys in `hp` that are valid arguments for the selected optimizer class.
    
    return optimizer(params=model.parameters(), **filtered_dict) # optimize all the model's parameters by using the filtered hyperparameters of the optimizer.

def run_stage(
    stage: str, # could be 'train', 'valid', or 'test'.
    stage_batcher: Iterable, # a batcher that yields batches of data indices.
    stage_idx: np.ndarray, # could be `train_idx`, `valid_idx`, or `test_idx`.
    InMemory: InMemoryClamp, #Yu: rename it based on my project.
    device: str, # device to run the model on, e.g., 'cuda:0' or 'cpu'.
    hparams: dict,
    model: DotProduct,
    criterion: torch.nn.Module, # loss function
    epoch: int, # current epoch number.
    verbose: bool,
    optimizer: torch.optim.Optimizer = None, # optimizer, could be None if stage is  'validation' or 'testing'.
    scheduler: torch.optim.lr_scheduler._LRScheduler = None # learning rate scheduler, could be None if stage is 'validation' or 'testing'.
):
    """
    """

    print(f'============================\n Starting {stage} \n============================')

    loss_sum = 0. # accumulate the total loss (float) for the epoch
    preactivations_l = [] #store model outputs for each batch
    topk_pos_l, arocc_pos_l = [], [] 
    topk_neg_l, arocc_neg_l = [], []
    activity_idx_l = [] # track the indices of activities processed in the batches.
    for batch_num, batch_indices in enumerate(stage_batcher):

         # get and unpack batch data
        batch_data = Subset(InMemory, indices=stage_idx)[batch_indices]
        activity_idx, compound_features, assay_features, activity = batch_data #?Yu: what is no assay_onehot?
        
        # move data to device
        if isinstance(compound_features, torch.Tensor):
            compound_features = compound_features.to(device)
        assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features
        #assay_onehot = assay_onehot.to(device).float() if not isinstance(assay_onehot[0], str) else assay_onehot #?Yu
        activity = activity.to(device)
        
        # forward
        #?YU: the 'multitask' related code is put off. 
        if hparams.get('loss_fun') in ('CE', 'Con'): # why in the two cases, `forward_dense` is used?
            preactivations = model.forward_dense(compound_features, #?Yu: go to check the difference between 'forward' and 'forward_dense'
                                                 assay_features) #?Yu: consider whether to remove 'assay_onehot' #?Yu: 'assay_onehot' is not defined in the loop.
        else:
            preactivations = model(compound_features, assay_features) #?Yu: why not `assay_onehot`?
        
        # loss
        beta = hparams.get('beta', 1) 
        if beta is None: beta = 1
        preactivations = preactivations*1/beta 
        loss = criterion(preactivations, activity) # if 'loss_fun' in ('CE', 'Con'), the 'preactivations' are the output of 'forward_dense', containing All Compound * Assays combination. 
        
        # zero gradients, backpropagation, update
        if stage == 'train':

            # ===================== For warmup_step =====================
            num_steps_per_epoch = len(stage_idx)/hparams['batch_size']
            # Warmup learning rate scheduler.
            class Linwarmup():
                def __init__(self, steps=10000): # i.e. warmup_steps = 10000.
                    self.step = 0
                    self.max_step = steps
                    self.step_size = 1/steps
                def get_lr(self, lr): #`lr` is required by `lambdaLR`, thus must be present for compatibility, even if not used in this function.
                    if self.step>self.max_step:
                        return 1
                    new_lr = self.step * self.step_size
                    self.step += 1
                    return new_lr

            #Todo Bug when set to 0
            if hparams.get('warmup_step'): 
                scheduler2 = LambdaLR(optimizer,lr_lambda=Linwarmup(steps=num_steps_per_epoch*hparams.get('warmup_epochs', 0)).get_lr)
            else:
                scheduler2 = None

            # ===================== zero gradients, backpropagation, update =====================
            optimizer.zero_grad() #set gradients from previous batch to zero.
            loss.backward()
            if hparams.get('optimizer') == 'SAM': 
                def closure():
                    """SAM (Sharpness-Aware Minimization) optimizer requires a closure function"""
                    preactivations = model(compound_features, assay_features) 
                    loss = criterion(preactivations, activity)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                optimizer.step() # update model weights using the gradients.
                scheduler.step() # Yu: if 'optimizer' is 'SAM', is there no need to call `scheduler.step()`?
                if scheduler2: scheduler2.step() #?Yu

        # accumulate loss 
        loss_sum += loss.item()

        # top_k highest score accuracy.
        if hparams.get('loss_fun') in ('CE', 'Con'): 
            # check whether the diagnnal items (matched compoundi and assay i) is within the top-k highest scores for each row. 
            ks = [1, 5, 10, 50] 

            # Ground-truth matches = diagonal indices
            y_true = torch.arange(0, len(preactivations), device=preactivations.device)

            # --- positives only ---
            pos_mask = activity == 1
            #print(f'The length of pos_mask is {len(pos_mask)}')
            #print(f'The length of y_true[pos_mask] is {len(y_true[pos_mask])}')
            #print(f'y_true[pos_mask] is {y_true[pos_mask]}')
            #print(f'preactivations[pos_mask] is {preactivations[pos_mask]}')
            if pos_mask.any(): # Check if there are any True values, i.e., this batch of activities contains at least one '1'.
                tkaccs_pos, arocc_pos = top_k_accuracy(y_true[pos_mask],preactivations[pos_mask], k=ks, ret_arocc=True)
            
            topk_pos_l.append(tkaccs_pos)
            arocc_pos_l.append(arocc_pos)

            # --- negatives only ---
            neg_mask = activity == 0
            #print(f'The length of neg_mask is {len(neg_mask)}')
            #print(f'The length of y_true[neg_mask] is {len(y_true[neg_mask])}')
            #print(f'y_true[neg_mask] is {y_true[neg_mask]}')
            if neg_mask.any():
                tkaccs_neg, arocc_neg = top_k_accuracy(y_true[neg_mask],preactivations[neg_mask], k=ks, ret_arocc=True)
        
            topk_neg_l.append(tkaccs_neg)
            arocc_neg_l.append(arocc_neg)

            #tkaccs, arocc = top_k_accuracy(torch.arange(0, len(preactivations)), preactivations, k=ks, ret_arocc=True)
            #topk_l.append(tkaccs)
            #arocc_l.append(arocc)
        
        # preactivations
        if hparams.get('loss_fun') in ('CE', 'Con'): #?Yu: combine this condition with the one above? and similarily, how to calc preactivations if `loss_fun` is neither 'CE' nor 'Con'?
            #preactivations = preactivations.sum(axis=1) #?Yu: why keep it here?
            preactivations =torch.diag(preactivations) # get only diag elements. This preactivations only contain the matched compound and assay pairs, e.g. C1A1, C2A2. 

        # accumulate preactivations
        preactivations_l.append(preactivations.detach().cpu())

        # accumulate_indices to track order in which the dataset is visited
        activity_idx_l.append(activity_idx) # activity_idx is a np.array, not a torch.tensor

        if batch_num % EVERY == 0 and verbose: 
            logger.info(f'Epoch{epoch}: Training batch {batch_num} out of {len(stage_batcher) - 1}.')

    # log mean loss over all minibatches
    stage_loss = loss_sum / len(stage_batcher)
    mlflow.log_metric(f'{stage}_loss', stage_loss, step=epoch)
    if wandb.run:
        if stage == 'train':
            wandb.log({
                'train/loss': stage_loss, # the mean training loss per batch for the epoch.
                'lr': scheduler2.get_last_lr()[0] if scheduler2 else scheduler.get_last_lr()[0]
            }, step=epoch)
        else:
            wandb.log({f'{stage}/loss': stage_loss}, step=epoch)


    # compute metrics for each assay (on the cpu) #?Yu: modify here to calc metrics on OR datasets.
    # md, metrics dictionary.
    preactivations = torch.cat(preactivations_l, dim=0)
    #print(f'During {stage}, preactivations.shape: {preactivations.shape}; \n preactivations: {preactivations[:5]}')
    probabilities = torch.sigmoid(preactivations).detach().cpu().numpy().astype(np.float32)
    #print(f'During {stage}, probabilities.shape: {probabilities.shape}; \n probabilities: {probabilities[:5]}')

    activity_idx = np.concatenate(activity_idx_l, axis=0)
    #print(f'During {stage}, The length of activity_idx {len(activity_idx)}, including {activity_idx}')

    targets = sparse.csc_matrix(
        (
            InMemory.activity.data[activity_idx],
            (
                InMemory.activity.row[activity_idx],
                InMemory.activity.col[activity_idx]
            )
        ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.bool_
    )
    #print(f'During {stage}, targets.shape: {targets.shape}; \ntargets.toarray().shape:{targets.toarray().shape}; \ntargets: {targets[:5, :5].toarray()}')

    scores = sparse.csc_matrix(
        (
            probabilities,
            (
                InMemory.activity.row[activity_idx],
                InMemory.activity.col[activity_idx]
            )
        ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.float32
    )
    #print(f'During {stage}, scores.shape: {scores.shape}; \nscores.toarray().shape:{scores.toarray().shape}; \n scores: {scores[:5, :5].toarray()}')

    #?Yu: `metrics` should be changed according to my implementation.
    #md = metrics.swipe_threshold_sparse(targets=targets, scores=scores,verbose=verbose>=2, ret_dict=True) # returns dict for with metric per assay in the form of {metric: {assay_nr: value}}
    bedroc_alpha = hparams.get('bedroc_alpha')
    md = swipe_threshold_sparse(targets=targets, scores=scores, bedroc_alpha=bedroc_alpha, verbose=verbose, ret_dict=True)
    for k, v in md.items():
        print(f"Metric '{k}': {v}")
        print(f'the length of v is {len(v)}')

    if hparams.get('loss_fun') in ('CE', 'Con'):
        #for i, k in enumerate(ks):
        #    md[f'top_{k}_acc'] = {0:np.vstack(topk_l)[:-1, i].mean()} # drop last (might be not full) #?Yu why
        #md['arocc'] = {0:np.hstack(arocc_l)[:-1].mean()} # drop last (might be not full) #?Yu why
        
        #Yu's first try
        #for i, k in enumerate(ks):
        #    md[f'top_{k}_acc_pos'] = {0:np.vstack(topk_pos_l)[:-1, i].mean()} if len(topk_pos_l)>0 and topk_pos_l[0] is not None else {0:np.nan}
        #    print(f"During {stage}, the top_{k}_acc_pos is {md[f'top_{k}_acc_pos']}")
        #    md[f'top_{k}_acc_neg'] = {0:np.vstack(topk_neg_l)[:-1, i].mean()} if len(topk_neg_l)>0 and topk_neg_l[0] is not None else {0:np.nan}
        #md['arocc_pos'] = {0:np.hstack(arocc_pos_l)[:-1].mean()} if len(arocc_pos_l)>0 and arocc_pos_l[0] is not None else {0:np.nan}
        #md['arocc_neg'] = {0:np.hstack(arocc_neg_l)[:-1].mean()} if len(arocc_neg_l)>0 and arocc_neg_l[0] is not None else {0:np.nan}

        for i, k in enumerate(ks):
            md[f'top_{k}_acc_pos'] = {0:np.vstack(topk_pos_l)[:-1, i].mean()}
            md[f'top_{k}_acc_neg'] = {0:np.vstack(topk_neg_l)[:-1, i].mean()}

        md['arocc_pos'] = {0:np.hstack(arocc_pos_l)[:-1].mean()}
        md['arocc_neg'] = {0:np.hstack(arocc_neg_l)[:-1].mean()}

    # log metrics mean over assays #?Yu: modify here to calc metrics on OR datasets.
    logdic = {f'{stage}_mean_{k}': np.nanmean(list(v.values())) for k,v in md.items() if v}
    logdic[f'{stage}_loss'] = stage_loss
    mlflow.log_metrics(logdic, step=epoch)
    if wandb.run: wandb.log({k.replace('_', '/'):v for k, v in logdic.items()}, step=epoch)

    return stage_loss, md, logdic
        

def train_and_test(
    InMemory: InMemoryClamp, #Yu: rename it based on my project.
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    hparams: dict,
    run_info: mlflow.entities.RunInfo,
    checkpoint_file: Optional[Path] = None,
    keep: bool = True,
    device: str = 'cpu',
    bf16: bool = False, #?Yu: when to set bf16=True?
    verbose: bool = True
) -> None:
    """
    Train a model on `InMemory[train_idx]` while validating on `InMemory[valid_idx]`.
    Once the training is finished, evaluate the model on `InMemory[test_idx]`.

    A model PyTorch checkpoint can be passed to resume training.

    Params:
    -------
    InMemory: :class:`dataset.InMemoryClamp`
         Dataset instance.
    train_idx: :class:`numpy.ndarray`
        Activity indices of the training split.
    valid_idx: :class:`numpy.ndarray`
        Activity indices of the validation split.
    test_idx: :class:`numpy.ndarray`
        Activity indices of the test split.
    hparams: dict
        Model characteristics and training strategy.
    run_info: :class:`mlflow.entities.RunInfo`
        MLflow's run details (for logging purposes).
    checkpoint_file: str or :class:`pathlib.Path`
        Path to a model PyTorch checkpoint from which to resume training.
    keep: bool
        Keep the persisted model weights if True, remove them otherwise.
    device: str
        Device to use for training (e.g., "cpu" or "cuda").
    verbose: bool
        Print verbose messages if True.
    """

    if verbose:
        if checkpoint_file is None:
            message = 'Strat training.'
        else:
            message = f'Resume training from {checkpoint_file}.'
        logger.info(message)


    # ================================= Function signature and parameters =================================
    # initialize checkpoint. If checkpoint_file is None, an empty dict is returned.
    checkpoint = init_checkpoint(checkpoint_file, device)
    # get paths to the artifacts directory and the model weights.
    artifacts_dir, checkpoint_file_path, metrics_file_path = get_mlflow_log_paths(run_info)

    early_stopping = EarlyStopper(patience=hparams['patience'], min_delta=0.0001)

    # ================================= Model initialization =================================
    print(hparams)
    
    #?Yu: Regard different assays or targets as different tasks. Keep the below `Multitask` related code if used later, otherwise remove it.
    #?Yu: why `setup_assay_onehot` is used here?`
    #if 'Multitask' in hparams.get('model'):
#
    #    _, train_assays = InMemory.get_unique_names(train_idx)
    #    InMemory.setup_assay_onehot(size=train_assays.index.max() + 1)
    #    train_assay_features = InMemory.assay_features[:train_assays.index.max() + 1] #?Yu: no `assay_features` defined neither in the primary code or 'InMemoryClamp` before.
    #    train_assay_features_norm = F.normalize(torch.from_numpy(train_assay_features), #?Yu: why set this here but use it quite later?
    #        p=2, dim=1 #Yu: p=2: the exponent value in the norm formulation; dim=1: the dimension to reduce.
    #    ).to(device)
#
    #    model = init_dp_model(
    #        compound_features_size=InMemory.compound_features_size,
    #        assay_features_size=InMemory.assay_onehot.size, #?Yu: `assay_onehot` has not been defined in the `InMemoryClamp` class before.
    #        hp=hparams,
    #        verbose=verbose
    #    )
    # 
    #else:
    model = init_dp_model(
        compound_features_size=InMemory.compound_features_size,
        assay_features_size=InMemory.assay_features_size,
        hp=hparams,
        verbose=verbose
    )
    
    if 'model_state_dict' in checkpoint:
        if verbose:
            logger.info('Load model_state_dict from checkpoint into model.')
        model.load_state_dict(checkpoint['model_state_dict']) # load weights from the checkpoint into the model.
        model.train() # `train` is a method of `nn.Module` that sets the module in training mode.
    
    model = model.to(device)

    # ================================= Optimizer initialization =================================
    # moving a model to the GPU should be done before the creation of its optimizer.
    # initialize optimizer
    optimizer = init_optimizer(model, hparams, verbose)

    if 'optimizer_state_dict' in checkpoint:
        if verbose:
            logger.info('Load optimizer_state_dict from checkpoint into optimizer.')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ================================= Learning rate Scheduler =================================
    # Constant learning rate scheduler.
    # `MultiplicativeLR` requires a function (lr_lambda). Here, lambda _: defines an anonymous function and the `_` is throwaway argument. Therefore, it is a constant function that always returns a constant `lr_factor`.
    lr_factor = hparams.get('lr_factor', 1) # if 'lr_factor' exists in `hparams`, use it, otherwise set it to 1.
    if lr_factor is None: lr_factor = 1
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda _: lr_factor) 

    if lr_factor !=1:
        logger.info(f'Scheduler enabled with lr_factor={hparams["lr_factor"]}. Note that this makes different runs difficult to compare.')
    else:
        logger.info('Scheduler enabled with lr_factor=1. This keeps the interface but results in no reduction.')

    # ================================= Loss function initialization =================================
    # initialize loss function #Yu: the core of clamp.

    # Binary cross-entropy loss
    criterion = nn.BCEWithLogitsLoss() # default, allowing `loss_fun` to be optional.
    if 'loss_fun' in hparams:
        class CustomCE(nn.CrossEntropyLoss): 
            """Cross entropy loss"""
            def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                """
                One-direction cross entropy loss (compounds -> assays)

                param
                -------
                input: predicted unnormalized logits. This is the raw output (logits, i.e. preactivations, no softmax/sigmoid) from the model, typically of shape [batch_size, batch_size] in contrastive/self-supervised settings.
                target: ground truth class indices or class probabilities.

                return
                -------
                for `F.cross_entropy`:
                """
                beta = 1/(input.shape[0]**(0.5)) # scaling factor, normalizes the logits so that their magnitude is independent of batch size, which can help stabilize training.
                input = input * (target*2-1) * beta # target from [0, 1] to [-1, 1]
                target = torch.arange(0, len(input)).to(input.device) # 'target' here is the ground truth class indices. However, the 'target' in last line is the 'true or false' labels?

                return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        class ConLoss(nn.CrossEntropyLoss):
            """
            Contrastive Loss
            
            Two-direction cross entropy (compounds <-> assays)
            """
            def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

                sigma = 1 # scaling factor. set to 1 here, so it does not affect the result.
                bs = target.shape[0] 
                modif = (1-torch.eye(bs)).to(target.device) + (torch.eye(bs).to(target.device)*(target*2-1)) # `torch.eye`: returns a 2-D tensor with ones on the diagonal and zeros elsewhere.`bs` is the number of rows.
                input = input*modif/sigma
                target = torch.arange(0, len(input)).to(input.device)

                label_smoothing = hparams.get('label_smoothing', 0.0)
                if label_smoothing is None:
                    label_smoothing = 0.0

                mol2txt = F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=label_smoothing)
                #print(f'In ConLoss, mol2txt: {mol2txt}')
                text2mol = F.cross_entropy(input.T, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=label_smoothing)
                #print(f'In ConLoss, text2mol: {text2mol}')
                return mol2txt + text2mol
            
        str2loss_fun = {
            'BCE': nn.BCEWithLogitsLoss(),
            'CE': CustomCE(),
            'Con': ConLoss(),
        }
        assert hparams['loss_fun'] in str2loss_fun, "loss_fun not implemented"
        criterion = str2loss_fun[hparams['loss_fun']]

    criterion = criterion.to(device)

    # ================================= Batch sampler =================================
    # shuffle the training indices at each epoch for stochastic gradient descent.
    train_sampler = RandomSampler(data_source=train_idx) 

    # no shuffling (i.e. sequential sampling) the validation/test indices for reproducibility during evaluation.
    valid_sampler = SequentialSampler(data_source=valid_idx)
    valid_batcher = BatchSampler(sampler=valid_sampler, batch_size=hparams['batch_size'], drop_last=False)

    test_sampler = SequentialSampler(data_source=test_idx)
    test_batcher = BatchSampler(sampler=test_sampler, batch_size=hparams['batch_size'], drop_last=False)

    epoch = checkpoint.get('epoch', 0) # resume from checkpoint if exists, otherwise start from 0.
    new_train_idx = None
    best_epoch = None 
    while epoch < checkpoint.get('epoch', 0) + hparams['epoch_max']:
        #Yu: remove the two options below if not used later
        # Optionally balance training data by downsampling negatives to match positives.
        if hparams.get('train_balanced', False): # if `train_balanced` is set to True, balance the training data. Otherwise, skip this step.
            logger.info('sampling balanced')
            num_pos = InMemory.activity.data[train_idx].sum() #  the number of positives
            remove_those = train_idx[((InMemory.activity.data[train_idx]) == 0)] # indices of the negatives
            remove_those = np.random.choice(remove_those, size=int(len(remove_those)-num_pos)) # randomly select negatives to remove.
            idx = np.in1d(train_idx, remove_those) # create a boolean mask where True indicates the indices to be removed.
            new_train_idx = train_idx[~idx] # use the boolean mask to filter out the indices to be removed.
            if isinstance(hparams['train_balanced'], int): # e.g. `train_balanced=1000`
                max_samples_per_epoch = hparams['train_balanced']
                if max_samples_per_epoch > 1:
                    logger.info(f'using only {max_samples_per_epoch} for one epoch')
                    new_train_idx = np.random.choice(new_train_idx, size=max_samples_per_epoch)
            train_sampler = RandomSampler(data_source=new_train_idx)
        # Optionally subsample training data to a fraction, which can be used for quick experiments by controlling training set size.
        if hparams.get('train_subsample', 0) > 0: 
            if hparams['train_subsample']<1: # e.g. `train_subsample=0.1`
                logger.info(f'subsample training set to {hparams["train_subsample"]*100}%')
                hparams['train_subsample'] = int(hparams['train_subsample']*len(train_idx))
            logger.info(f'subsample training set to {hparams["train_subsample"]}')
            sub_train_idx = np.random.choice(train_idx if new_train_idx is None else new_train_idx, size=int(hparams['train_subsample'])) # e.g. `train_subsample=1000`
            train_sampler = RandomSampler(data_source=sub_train_idx)
        
        train_batcher = BatchSampler(sampler=train_sampler, batch_size=hparams['batch_size'], drop_last=False)
 
        # ================================= Training loop =================================
        print(f'============================\n epoch {epoch} \n============================')
        
        run_stage(stage='train', stage_batcher=train_batcher, stage_idx=train_idx, InMemory=InMemory,
            device=device, hparams=hparams, model=model, criterion=criterion,
            epoch=epoch,verbose=verbose,
            optimizer=optimizer, scheduler=scheduler
        )

        # ================================= Validation loop =================================
        with torch.no_grad():
            
            model.eval()

            valid_loss, _, logdic = run_stage(stage='valid', stage_batcher=valid_batcher, stage_idx=valid_idx, InMemory=InMemory,
                device=device, hparams=hparams, model=model, criterion=criterion,
                epoch=epoch,verbose=verbose
            )

            # monitor metric
            evaluation_metric = 'valid_mean_davgp'
            #evaluation_metric = 'valid_mean_bedroc' #Yu edited

            if evaluation_metric not in logdic:
                logger.info('Using -valid_loss because valid_mean_davgp not in logdic.')
            log_value = logdic.get(evaluation_metric, -valid_loss) # get `evaluation_metric` first, otherwise, get the second argument '-valid_loss'
            # metric_monitor(logdic['valid_mean_davgp'], epoch)
            do_early_stop = early_stopping(-log_value) # early_stopper expected the small is better, but `-log_value` reverses the logic
            print(f'Validation loop: \n do_early_stop={do_early_stop}')

            # log model checkpoint dir
            if wandb.run:
                wandb.run.config.update({'model_save_dir':checkpoint_file_path})
            
            if early_stopping.improved:
                best_epoch = epoch
                logger.info(f'Epoch {epoch}: Save model and optimizer checkpoint with val-davgp: {log_value}.')
                torch.save({
                    'value': log_value,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file_path) # The model is saved at ./mlruns/runid/artifacts
                print(f'Checkpoint saved to {checkpoint_file_path}.')

            if do_early_stop:
                logger.info(f'Epoch {epoch}: Out of patience. Early stop!')
                break

            model.train() # set the model back to training model for the next epoch.
        
        epoch +=1
    
    # ================================= Testing loop =================================
    # test with best model
    with torch.no_grad():
        
        if best_epoch is None:
            logger.warning('No best model recorded. Testing with initial model.')
        else:
            logger.info(f'Testing best model from epoch {best_epoch}.The checkpoint file is {checkpoint_file_path}.')
            checkpoint = torch.load(checkpoint_file_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        _, md, logdic = run_stage(stage='test', stage_batcher=test_batcher, stage_idx=test_idx, InMemory=InMemory,
                  device=device, hparams=hparams, model=model, criterion=criterion,
                  epoch=epoch, verbose=verbose)
        # ===============================================

        if verbose: logger.info(pd.DataFrame.from_dict([logdic]).T) #?Yu: print a dataframe?

        # Yu: remove the code calculate counts and positives if not used later
        # compute test activity counts and positives
        #counts, positives = {}, {}
        #for idx, col in enumerate(targets.T):
        #    if col.nnz == 0:
        #        continue
        #    counts[idx] = col.nnz
        #    positives[idx] = col.sum()

        # 'test_mean_bedroc': 0.6988015835969245, 'test_mean_davgp': 0.16930837444561778, 'test_mean_dneg_avgp': 0.17522445272085613, 
        # 'test/mean/auroc': 0.6709850363704437, 'test/mean/avgp': 0.6411171492554743, 'test/mean/neg/avgp': 0.7034156779109996, 
        # 'test/mean/argmax/j': 0.4308185
        # store test metrics and counts in a parquet file
        metrics_df = pd.DataFrame(md)
        metrics_df['argmax_j'] = metrics_df['argmax_j'].apply(sigmoid)
        #?Yu: why is the below code commented out in the primary code.
        # metrics_df['counts'] = counts # for PC_large: ValueError: Length of values (3933) does not match length of index (615)
        # metrics_df['positives'] = positves

        metrics_df.index.rename('assay_idx', inplace=True)

        metrics_df = InMemory.assay_names.merge(metrics_df, left_index=True, right_index=True)
        logger.info(f'Writing test metrics to {metrics_file_path}')
        metrics_df.to_parquet(metrics_file_path, compression=None, index=True)

        with pd.option_context('float_format', "{:.2f}".format):
            print(metrics_df)
            print(metrics_df.mean(0, numeric_only=True))

        model.train()
    
    if not keep:
        logger.info('Delete model checkpoint.')
        checkpoint_file_path.unlink() #unlink_ remove file or link.

def test(
        InMemory: InMemoryClamp,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        hparams: dict, 
        run_info: mlflow.entities.RunInfo,
        device: str = 'cpu',
        verbose: bool = False, 
        model = None
) -> None: #?Yu: isn't `metric.df` returned?
    """
    Test a model on `InMemory[test_idx]`if test metrics are not yet to be found under the `actifacts` directory. 
    If so, interrupt the program.

    Params
    ----------
    InMemory: InMemoryClamp
        Dataset instance
    train_idx: :class:`numpy.ndarray``
        Activity indices of the training split. Only for multitask models. #?Yu: why only for multitask models?
    test_idx: :class:`numpy.ndarray``
        Activity indices of the test split.
    run_info: class:`mlflow.entities.RunInfo`
        MLflow's run details (for logging purposes).
    device: str
        Computing device.
    verbose: bool
        Be verbose if True.
    """

    if verbose:
        logger.info('Start evaluation.')
    
    bedroc_alpha = hparams.get('bedroc_alpha')

    artifacts_dir = Path('mlruns', run_info.experiment_id, run_info.run_id, 'artifacts')

    # for logging new checkpoints
    checkpoint_file_path = artifacts_dir / 'checkpoint.pt'
    metrics_file_path = artifacts_dir / 'metrics.parquet'

    # initialize checkpoint
    if model != None:
        checkpoint = init_checkpoint(checkpoint_file_path, device)
        assert checkpoint, 'No checkpoint found'
        assert 'model_state_dict' in checkpoint, 'No model found in checkpoint' #?Yu 'model_state_dict' in checkpoint? how this attribute be gotten?

    artifacts_dir, checkpoint_file_path, metrics_file_path = get_mlflow_log_paths(run_info)

    # initialize model
    if 'Multitask' in hparams['model']:
        _, train_assays = InMemory.get_unique_names(train_idx)
        InMemory.setp_assay_onehot(size=train_assays.index.max() + 1)
        train_assay_features = InMemory.assay_features[:train_assays.index.max() + 1]
        train_assay_features_norm = F.normalize(
            torch.from_numpy(train_assay_features), p=2, dim=1
        ).to(device)

        if model != None:
            model = init_dp_model(
                compound_features_size= InMemory.compound_features_size,
                assay_features_size= InMemory.assay_onehot.size,
                hp=hparams, #?Yu: how can the hparams that get the best model be used here?
                verbose=verbose
            )
    else:
        if model != None:
            model = init_dp_model(
                compound_features_size=InMemory.compound_features_size,
                assay_features_size=InMemory.assay_features_size,
                hp=hparams, verbose=verbose
            )
        
    if verbose:
        logger.info('Load model from checkpoint.')
    if model != None:
        model.load_state_dict(checkpoint['model_state_dict'])

    # assignment is not necessary when moving modules, but it is for tensors.
    # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
    # here I only assign for consistency
    model = model.to(device)

    # initialize loss function
    criterion = nn.BCEWithLogitsLoss # why is it enough to use this function instead of `CustomCE` or `ConLoss` during testing?
    criterion = criterion.to(device)

    test_sampler = SequentialSampler(data_source=test_idx)
    test_batcher = BatchSampler(
        sampler=test_sampler,
        batch_size=hparams['batch_size'],
        drop_last=False #?Yu:
    )

    epoch = checkpoint.get('epoch', 0)
    with torch.no_grad():

        model.eval()

        loss_sum = 0.
        preactivations_l = []
        activity_idx_l = []
        for batch_num, batch_indices in enumerate(test_batcher):

            # get and unpack batch data
            batch_data = Subset(InMemory, indices=test_idx)[batch_indices]
            activity_idx, compound_features, assay_features, activity = batch_data #?Yu: why `assay_onehot` is added here?

            # move data to device
            if isinstance(compound_features, torch.Tensor):
                compound_features = compound_features.to(device)
            assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features
            activity = activity.to(device)

            # forward
            if 'Multitask' in hparams['model']:
                assay_features_norm = F.normalize(assay_features, p=2, dim=1)
                sim_to_train = assay_features_norm @ train_assay_features_norm.T
                sim_to_train_weights = F.softmax(sim_to_train * hparams['multitask_temperature'], dim=1)
                preactivations = model(compound_features, sim_to_train_weights)
            else:
                preactivations = model(compound_features, assay_features)   
            
            # loss
            loss = criterion(preactivations, activity) 

            # accumulate loss
            loss_sum += loss.item()

            # accumulate preactivations
            # - need to detach; preactivations.requires_grad is True
            # - move it to cpu
            preactivations_l.append(preactivations.detach().cpu()) #?Yu: why is `detach` used here but not in the `def train_and_test``

            # accumulate indices just to double check
            # - activity_idx is a np.array, not a torch.tensor
            activity_idx_l.append(activity_idx)

            if batch_num % EVERY == 0 and verbose:
                logger.info(f'Epoch {epoch}: Test batch {batch_num} out of {len(test_batcher) - 1}.')

        # log mean loss over all minibatches
        mlflow.log_metric('test_loss', loss_sum / len(test_batcher), step=epoch)
        if wandb.run: wandb.log({'test/loss': loss_sum/len(test_batcher)}, step=epoch)

        # compute test auroc and avgp for each assay (on the cpu)
        preactivations = torch.cat(preactivations_l, dim=0)
        probabilities = torch.sigmoid(preactivations).numpy()

        activity_idx = np.concatenate(activity_idx_l, axis=0)
        assert np.array_equal(activity_idx, test_idx) #?Yu: this code line is commented out in the `def train_and_test`

        targets = sparse.csc_matrix(
            (
                InMemory.activity.data[test_idx],
                (
                    InMemory.activity.row[test_idx],
                    InMemory.activity.col[test_idx]
                )
            ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.bool_
        )

        scores = sparse.csc_matrix(
            (
                probabilities,
                (
                    InMemory.activity.row[test_idx],
                    InMemory.activity.col[test_idx]
                )
            ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.float32
        )

        #md = metrics.swipe_threshold_sparse(targets=targets, scores=scores, verbose=verbose>=2, ret_dict=True)
        md = swipe_threshold_sparse(targets=targets, scores=scores, bedroc_alpha=bedroc_alpha, verbose=verbose>=2, ret_dict=True)

        # log metrics mean over assays
        logdic = {f'test_mean_{mdk}': np.mean(list(md[f'{mdk}'].values())) for mdk in md.keys()} #?Yu: why is `mdk` used here? different from the one in the `def train_and_test`
        mlflow.log_metrics(logdic, step=epoch)

        if wandb.run: wandb.log({k.replace('_', '/'):v for k, v in logdic.items()}, step=epoch)
        if verbose: logger.info(logdic)

        # compute test activity counts and positives
        counts, positives = {}, {}
        for idx, col in enumerate(targets.T):#?Yu: why `targets.T`?
            if col.nnz == 0:
                continue
            counts[idx] = col.nnz
            positives[idx] = col.sum()
        
        # store test metrics and counts in a parquet file
        metrics_df = pd.DataFrame(md)
        metrics_df['argmax_j'] = metrics_df['argmax_j'].apply(sigmoid)
        metrics_df['counts'] = counts
        metrics_df['positives'] = positives

        metrics_df.index.rename('assay_idx', inplace=True)

        metrics_df = InMemory.assay_names.merge(metrics_df, left_index=True, right_index=True)
        logger.info(f'Writing test metrics to {metrics_file_path}')
        metrics_df.to_parquet(metrics_file_path, compression=None, index=True)

        if wandb.run:
            wandb.log({"metrics_per_assay": wandb.Table(data=metrics_df)})
        
        logger.info(f'Saved best test-metrics to {metrics_file_path}')
        logger.info(f'Saved best checkpoint to {checkpoint_file_path}')

        model.train() #?Yu: why is this line here?

        with pd.option_context('float_format', "{:.2f}".format):
            print(metrics_df)
        
        return metrics_df

def load_model_from_mlflow(mlrun_path='', compound_features_size=4096, assay_features_size=2048, device='cuda:0', ret_hparams=False):
    """
    Load a model from a mlflow run.

    Params
    ----------
    mlrun_path: str
        Path to the mlflow run.

    Returns
    ----------
    if ret_hparams:
        model: torch.nn.Module; hparams: dict
    else: 
        model: torch.nn.Module
    """
    if isinstance(mlrun_path, str):
        mlrun_path = Path(mlrun_path)
    
    hparams = get_hparams(path=mlrun_path, mode='logs', vervose=True)

    if compound_features_size is None:
        elp = Path(hparams['dataset'])/('compound_features_'+hparams['compound_mode']+'.npy')
        try:
            compound_features_size = np.load(elp).shape[1]
        except FileNotFoundError:
            raise FileNotFoundError(f'Compound features file {elp} not found.')

    if assay_features_size is None:
        elp = Path(hparams['dataset'])/('assay_features_'+hparams['assay_mode']+'.npy')
        try:
            assay_features_size = np.load(elp).shape[1]
        except FileNotFoundError:
            raise FileNotFoundError(f'Assay features file {elp} not found.')
    
    model = init_dp_model(
        compound_features_size=compound_features_size,
        assay_features_size=assay_features_size,
        hp=hparams, verbose=True
    )

    # load in the model an generate hidden layers
    checkpoint = init_checkpoint(mlrun_path/'artifacts/checkpoint.pt', device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if ret_hparams:
        return model, hparams
    return model

#================================================================================================================
#                                       train.py
#================================================================================================================
"""example call:
python clamp/train.py \
    --dataset=./data/fsmol \
    --split=FSMOL_split \
    --assay_mode=clip \
    --compound_mode=morganc+rdkc 
"""

""" training pubchem23 without downstream datasets
python clamp/train.py \
    --dataset=./data/pubchem23/ \
    --split=time_a \
    --assay_mode=clip \
    --batch_size=8192 \
    --dropout_hidden=0.3 \ #?Yu: this parameter is not implemented in the primary code
    --drop_cidx_path=./data/pubchem23/cidx_overlap_moleculenet.npy \
    --train_subsample=10e6 \
    --wandb --experiment=pretrain
"""
def parse_args_override(override_hpjson=True): #?Yu: why set this to True?
    parser = argparse.ArgumentParser('Train and test a single run of clamp-Activity model. Overrides arguments from hyperparam-file')
    parser.add_argument('-f', type=str) #?Yu 
    parser.add_argument('--dataset', type=str, default='./data/fsmol', help='Path to a prepared dataset directory') #?Yu: parquet file or npy file or others?
    parser.add_argument('--assay_mode', type=str, default='lsa', help='Type of assay features("clip", "biobert", or "lsa")') #?Yu: why lsa is default? where is lsa implemented?#
    parser.add_argument('--assay_columns_list', type=str, default='columns_short', help='Name of the assay columns to use in the dataset (default: columns_short).') 
    parser.add_argument('--compound_mode', type=str, default='morganc+rdkc', help='Type of compound features (default: morganc+rdkc)') 
    parser.add_argument('--hyperparams', type=str, default='./hparams/default.json', help='Path to hyperparameters to use in training (json, Hyperparams, or logs).')

    parser.add_argument('--checkpoint', help='Path to a model-optimizer PyTorch checkpoint from which to resume training.', metavar='')
    parser.add_argument('--experiment', type=str, default='debug', help='Name of MLflow experiment where to assign this run.', metavar='')
    parser.add_argument('--random', action='store_true', help='Forget about the specified model and run a random baseline.') #?Yu: delete it later if not used?

    #?Yu: why the arguments below are commented out in the primary code?
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use for training (default: AdamW).')
    parser.add_argument('--lr_ini', type=float, default=1e-5, help='Initial learning rate (default: 1e-5).' )
    parser.add_argument('--l2', type=float, default=0.01, help='Weight decay to use for training (default: 0.01).')
    parser.add_argument('--loss_fun', type=str, default='BCE', help='Loss function to use for training (default: BCE).')
    parser.add_argument('--epoch_max', type=int, default=50, help='Maximum number of epochs to train for (default: 100).')

    parser.add_argument('--compound_layer_sizes', type=str, default=None, help='Hidden layer sizes for compound features (default: None, i.e. use hidden_layers).')
    parser.add_argument('--assay_layer_sizes', type=str, default=None, help='Hidden layer sizes for assay features (default: None, i.e. use hidden_layers).')
    parser.add_argument('--hidden_layers', type=str, default=[2048,1024], help='Hidden layer sizes for the model (default: [512, 256]).')

    parser.add_argument('--verbose','-v', type=int, default=0, help='verbosity level (default:0)') 
    parser.add_argument('--seed', type=int, default=None, help='seed everything with provided seed, default no seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number to use (default: 0).', metavar='')
    
    parser.add_argument('--split', type=str, default='time_a_c', help='split-type. Default:time_a_c for time based assay and compound split. Options: time_a, time_c, random:{seed}, or column of activity.parquet triplet.') #?Yu: shall I modify these split options?
    parser.add_argument('--support_set_size', type=int, default=0, help='per task how many to add from test- as well as valid- to the train-set (Default:0, i.e. zero-shot).') #?Yu: '0' -> 0
    parser.add_argument('--train_only_actives', action='store_true', help='train only with active molecules.')
    parser.add_argument('--drop_cidx_path', type=str, default=None, help='Path to a file containing a np of cidx (NOT CIDs) to drop from the dataset.') #?Yu: a np of cidx?

    parser.add_argument('--bedroc_alpha', type=float, default=20.0, help='alpha for bedroc metric (default: 20.0)')
    
    parser.add_argument('--wandb', '-w', action='store_true', help='wandb logging on')
    parser.add_argument('--bf16', action='store_true', help='use bfloat16 for training') #?Yu: bfloat16?

    args, unknown = parser.parse_known_args() #?Yu: ?
    keypairs = dict([unknown[i:i+2] for i in range(0, len(unknown), 1) if unknown[i].startswith('--') and not (unknown[i+1:i+2]+["--"])[0].startswith('--')]) #?Yu: don't understand. delete it?

    hparams = get_hparams(path=args.hyperparams, mode='json', verbose=args.verbose)

    if override_hpjson:
        for k, v in NAME2FORMATTER.items():
            if (k not in args):
                default = hparams.get(k, None)
                parser.add_argument('--'+k, type=v, default=default)
                if (k in keypairs):
                    logger.info(f'{k} from hparams file will be overwritten')
        args = parser.parse_args()
    
    if args.nonlinearity is None:
        args.nonlinearity = 'ReLU'
    # Without the code below, error related to `def _encoder`` will be raised.
    if args.compound_layer_sizes is None:
        logger.info('no compound_layer_sizes provided, setting to hidden_layers')
        args.compound_layer_sizes = args.hidden_layers
    if args.assay_layer_sizes is None:
        logger.info('no assay_layer_sizes provided, setting to hidden_layers')
        args.assay_layer_sizes =  args.hidden_layers


    return args

def setup_dataset(dataset='./data/fsmol', assay_mode='lsa', assay_columns_list='columns_short',compound_mode='morganc+rdkc', split='split', 
                  verbose=False, support_set_size=0, drop_cidx_path=None, train_only_actives=False, **kwargs):
    """
    Setup the dataset by given a dataset-path.
    Loads an InMemoryClamp object containing:
    - split: 'split' is the column name in the activity.parquet, 'time_a_c' is in the primary code
    - support_set_size: 0, adding {support_set_size} samples from test and from valid to train (per assay/task);
    - train_only_actives: False, only uses the active compounds;
    - drop_cidx_path: None, path to a npy file containing cidx (NOT CIDs) to drop from the dataset.
    """
    dataset = Path(dataset)
    clamp_dl = InMemoryClamp(
        root=dataset,
        assay_mode=assay_mode,
        assay_column_list=assay_columns_list,
        compound_mode=compound_mode,
        verbose=verbose,
    )
    print(f"'assay_mode' is {assay_mode},\n"
          f"'assay_column_list' is {assay_columns_list},\n"
          f"'compound_mode' is {compound_mode},\n"
          f"'split' is {split},\n")

    # ===== split =====
    logger.info(f'loading split info from activity.parquet triplet-list under the column split={split}')
    try:
        splits = pd.read_parquet(dataset/'activity.parquet')[split]
    except KeyError:
        raise ValueError(f'no split column {split} in activity.parquet', pd.read_parquet(dataset/'activity.parquet').columns, 'columns available')
    train_idx, valid_idx, test_idx =[splits[splits==sp].index.values for sp in ['train', 'valid', 'test']]
    print(f'Found {len(train_idx)} train,\n'
          f'{len(valid_idx)} valid, \n'
          f'and {len(test_idx)} test samples in the dataset.')

    # ===== support_set_size =====

    # ===== train_only_actives =====

    # ===== drop_cidx_path =====

    # ===== verbose =====

    return clamp_dl, train_idx, valid_idx, test_idx

#================================================================================================================
#                                       main function in train.py
#================================================================================================================
def main(args):
    # Hyperparameter Preparation
    hparams = args.__dict__

    # MLflow Experiment Setup
    mlflow.set_experiment(args.experiment)

    # Seeding (Optional)
    if args.seed:
        seed_everything(args.seed)
        logger.info(f'seeded everything with seed {args.seed}') #?Yu: if not needed, delete it?
    
    # Dataset Preparation
    clamp_dl, train_idx, valid_idx, test_idx = setup_dataset(**args.__dict__)
    assert set(train_idx).intersection(set(valid_idx)) == set() # assert no overlap between the splits.
    assert set(train_idx).intersection(set(test_idx)) == set()

    # Weights & Biases (wandb) Logging
    if args.wandb:
        runname = args.experiment+args.split[-1]+args.assay_mode[-1]
        if args.random:
            runname += 'random'
        else:
            runname = str(runname)+''
            runname += str(args.model) #?Yu: what could `args.model` be?
        runname += ''.join([chr(random.randrange(97, 97 +26)) for _ in range(3)]) # to add some randomness to the run name
        #wandb.init(project='clipGPCR', entity='xixichennn', name=runname, config=args.__dict__)
        import os

        print("WANDB_API_KEY:", os.environ.get("WANDB_API_KEY"))
        print("WANDB_PROJECT:", os.environ.get("WANDB_PROJECT"))
        print("WANDB_ENTITY:", os.environ.get("WANDB_ENTITY"))
        wandb.login(key=os.environ["WANDB_API_KEY"])
        wandb.init(project=os.environ.get("WANDB_PROJECT"), entity=os.environ.get("WANDB_ENTITY"), name=runname, config=args.__dict__)

    # Device Setup
    device = set_device(gpu=args.gpu, verbose=args.verbose)

    # Metrics DataFrame Initialization
    metrics_df = pd.DataFrame()

    # Training/Testing Run (with MLflow Logging)
    try:
        with mlflow.start_run(): # begins a new experiment run.
            mlflowi = mlflow.active_run().info # provides metadata (like run id, experiment id) for this run.
        
        # Checkpoint Resume Logging
        if args.checkpoint is not None:
            mlflow.set_tag(
                'mlflow.note.content',
                f'Resumed training from {args.checkpoint}.'
            )
        
        # Assay Mode Consistency and Logging #?Yu: why this only applies to assay_mode, but not other hyperparameters?
        if 'assay_mode' in hparams:
            if hparams['assay_mode'] != args.assay_mode:
                # Warn if there's a mismatch.
                logger.warning(f'Assay features are "{args.assay_mode}" in command line but \"{hparams["assay_mode"]}\" in hyperparameter file.')
                logger.warning(f'Command line "{args.assay_mode}" is the prevailing option.')
                hparams['assay_mode'] = args.assay_mode
        else:
            mlflow.log_param('assay_mode', args.assay_mode)
        mlflow.log_params(hparams) # Logs all hyperparamters to MLflow for easy reference and reproducibility.

        # Comment out the below code block because the random baseline is seemed unnecessary for my current plan.
        #if args.random:
        #    mlflow.set_tag(
        #        'mlflow.note.content',
        #        'Ignore the displayed parameters. Metrics correspond to predictions randomly drawn from U(0, 1).'
        #    )
        #    utils.random(
        #        clamp_dl,
        #        test_idx=test_idx,
        #        run_info=mlflowi,
        #        verbose=args.verbose)
        #else:
        #metrics_df = utils.train_and_test(
        metrics_df = train_and_test(
            clamp_dl, 
            train_idx=train_idx,
            valid_idx=valid_idx,
            test_idx=test_idx,
            hparams=hparams,
            run_info=mlflowi,
            checkpoint_file=args.checkpoint,
            device=device,
            bf16=args.bf16,
            verbose=args.verbose)
    
    except KeyboardInterrupt:
        logger.error('Training manually interrupted. Trying to test with last checkpoint.')
        # If the training is manually interrupted, it still tries to evaluate (test) the model using the last checkpoint, and logs results to the same MLflow run.
        #?Yu: delete the below code if not used.
        #metrics_df = utils.test(
        metrics_df = test(
            clamp_dl,
            train_idx=train_idx,
            test_idx=test_idx,
            hparams=hparams,
            run_info=mlflowi,
            device=device,
            verbose=args.verbose,
        )
    
if __name__ == '__main__':
    args = parse_args_override()

    run_id = str(time()).split('.')[0]
    fn_postfix = f'{args.experiment}_{run_id}'

    if args.verbose>=1:
        logger.info('Run args:', os.getcwd()+__file__, args.__dict__)

    main(args)