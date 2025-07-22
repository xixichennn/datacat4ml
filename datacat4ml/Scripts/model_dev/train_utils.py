# conda env: datacat (python=3.8.2)
# for `utils.py`
# ==== all.py ====
from pathlib import Path
import os
import numpy as np

# ==== metrics.py ====
from sklearn import metrics
from rdkit.ML.Scoring.Scoring import CalcBEDROC

# ==== dataloader.py ====
import torch
from torch.utils.data import Dataset
from typing import Any, Iterable, List, Optional, Tuple, Union
import pandas as pd
from scipy import sparse
try: # only if Graph-Model is used
    import dgl
except: pass

# ==== utils.py ====
import json
from loguru import logger
import mlflow
import mlflow.entities
from datacat4ml.Scripts.model_dev.model_def import DotProduct
import datacat4ml.Scripts.model_dev.model_def as model_def
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import Subset, RandomSampler, SequentialSampler, BatchSampler
from scipy.special import expit as sigmoid

# ==== train.py ====
import argparse
#import mlflow
import random
import wandb
from time import time

#================================================================================================================
#                                       metrics.py
#================================================================================================================
def get_sparse_data(m, i):
    """
    Get the non-zero data from a sparse matrix by an index.
    This function will be called in `swipe_threshold_sparse`

    Params
    ------
    m : scipy sparse matrix.
    i: index

    Returns
    -------
    m.indptr[i] to m.indptr[i + 1] data: List[float]
        A list of non-zero values in the sparse matrix row at index `i`.
    """
    return [m.data[index] for index in range(m.indptr[i], m.indptr[i + 1])]

def calc_bedroc_on_ml(y_true, y_pred_proba, alpha: float = 20.0):
    """ Calculates the bedroc score unsing rdkit.ML.Scoring.CalcBEDROC.
    The source code is available at https://github.com/rdkit/rdkit/blob/master/rdkit/ML/Scoring/Scoring.py#L103
    This function is defined as `def CalcBEDROC(score, col, alpha)`, 
        where `score` is ordered list with tuples of (pred_proba, true value), with pred_proba being descendingly sorted,
        'col' is the column index for true values, i.e. 1 for the positive class (1), 
        and `alpha` is the early recognition parameter.

    
    Params
    ------
    y_pred_proba: (lst/array) a list of predicted probabilities for all compounds, i.e. the value of model.predict_proba(x_test). 
                   y_pred_proba[:, 1] is the probability of the positive class (1).
    y_true: (lst/array) a list of true values for all compounds.
    alpha: (float)  early recognition parameter. 
            alpha = 80.5, 2% of the top-ranked compounds of the all compounds were calculated; 2% represents the proportion of active compounds in the DUD-E database;
            alpha = 321.5, 0.5% of the top-ranked compounds of the all compounds  were calculated; 4 times smaller than 2% --> early recognition.
            alpha = 20.0(default), 8% of the top-ranked compounds of the all compounds were calculated; 4 times larger than 2% --> is interesting for the cases where relatively high-throughput experiments are available.

    returns
    -------
    (float) BEDROC score
    """

    score = list(zip(y_pred_proba[:, 1], y_true))
    score.sort(key=lambda x: x[0], reverse=True) # sort the list by the first element, i.e. # the predicted probability of the positive class (1), in descending order.
    bedroc_score = CalcBEDROC(score, 1, alpha) # 1 is the column index for the ground-truth values (y_true)

    return bedroc_score

def calc_bedroc_on_clip(y_true, y_score, alpha: float = 20.0):
    """ Calculates the bedroc score unsing rdkit.ML.Scoring.CalcBEDROC.
    The source code is available at https://github.com/rdkit/rdkit/blob/master/rdkit/ML/Scoring/Scoring.py#L103
    This function is defined as `def CalcBEDROC(score, col, alpha)`, 
        where `score` is ordered list with tuples of (pred_proba, true value), with pred_proba being descendingly sorted,
        'col' is the column index for true values, i.e. 1 for the positive class (1), 
        and `alpha` is the early recognition parameter.

    
    Params
    ------
    y_true: (lst/array) a list of true values for all compounds.
    y_p: (lst/array) a list of predicted probabilities for all compounds, i.e. the value of model.predict_proba(x_test). 
                   y_pred_proba[:, 1] is the probability of the positive class (1).
    alpha: (float)  early recognition parameter. 
            alpha = 80.5, 2% of the top-ranked compounds of the all compounds were calculated; 2% represents the proportion of active compounds in the DUD-E database;
            alpha = 321.5, 0.5% of the top-ranked compounds of the all compounds  were calculated; 4 times smaller than 2% --> early recognition.
            alpha = 20.0(default), 8% of the top-ranked compounds of the all compounds were calculated; 4 times larger than 2% --> is interesting for the cases where relatively high-throughput experiments are available.

    returns
    -------
    (float) BEDROC score
    """

    pair = list(zip(y_score, y_true)) # pair the predicted scores with the true values
    pair.sort(key=lambda x: x[0], reverse=True)
    bedroc_score= CalcBEDROC(pair, 1, alpha) # 1 is the column index for the ground-truth values (y_true)

    return bedroc_score

def swipe_threshold_sparse(targets, scores, bedroc_alpha = 20, verbose=True, ret_dict=False):
    """
    This function computes metrics per assay (i.e., column-wise):

    Compute ArgMaxJ, AUROC, AVGP, AUPRC and BEDROC (and more if ret_dict=True) metrics for the true binary values
    `targets` given the predictions `scores`.

    Params
    ---------
    targets: :class:`scipy.sparse.csc_matrix`, shape(N, M) # N refers to the number of compounds, M refers to the number of assays.
        True target values.
    scores: :class:`scipy.sparse.csc_matrix`, shape(N, M)
        Predicted values
    bedroc_alpha: float
        Early recognition parameter for BEDROC. Default is 20.0, which is interesting for the cases where relatively high-throughput experiments are available.
    verbose: bool
        Be verbose if True.
    

    Returns
    ---------
    tuple of dict
        - ArgMaxJ of each valid column keyed by the column index (assay index), # get the optimal threshold that maximizes the difference between true positive rate (TPR) and false positive rate (FPR).
        - AUROC of each valid column keyed by the column index (assay index) # AUROC
        - AVGP of each valid column keyed by the column index (assay index) # average precision score
        - NegAVGP of each valid column keyed by the column index (assay index) # average precision score for the negative class (1 - y_true)
        - dAVGP of each valid column keyed by the column index (assay index) # difference between average precision and the mean of y_true
        - dNegAVGP of each valid column keyed by the column index (assay index) # difference between average precision for the negative class and the mean of 1 - y_true
        - AUPRC of each valid column keyed by the column index (assay index) # area under the precision-recall curve
        - BEDROC of each valid column keyed by the column index (assay index) # early recognition.
    """

    assert targets.shape == scores.shape, '"targets" and "scores" must have the same shape.' # assert <condition>, <error message>
    
    # find non-empty columns
    # (https://mike.place/2015/sparse/ for CSR, but works for CSC, too)
    non_empty_idx = np.where(np.diff(targets.indptr) != 0)[0] # Return the compounds that have at least one assay with a non-zero value?

    counter_invalid = 0
    argmax_j, auroc, avgp, neg_avgp, davgp, dneg_avgp, auprc, bedroc = {}, {}, {}, {}, {}, {}, {}, {}

    for col_idx in non_empty_idx: # This function computes metrics per assay (i.e., column-wise):
        y_true = np.array(list(get_sparse_data(targets, col_idx)))
        if len(pd.unique(y_true)) == 1: # `pd.unique` is faster than `np.unique` and `set`.
            counter_invalid += 1
            continue
        y_score = np.array(list(get_sparse_data(scores, col_idx)))
        assert len(y_true) == len(y_score)

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        assert len(fpr) == len(tpr) == len(thresholds), 'Length mismatch between "fpr", "tpr", and "thresholds".'
        argmax_j[col_idx] = thresholds[np.argmax(tpr - fpr)] 

        auroc[col_idx] = metrics.roc_auc_score(y_true, y_score)
        avgp[col_idx] = metrics.average_precision_score(y_true, y_score)
        neg_avgp[col_idx] = metrics.average_precision_score(1 - y_true, 1 - y_score)
        davgp[col_idx] = avgp[col_idx] - y_true.mean()
        dneg_avgp[col_idx] = neg_avgp[col_idx] - (1 - y_true.mean())

        # check if the auprc is same as avgp.
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
        auprc[col_idx] = metrics.auc(recall, precision)
        
        bedroc[col_idx] = calc_bedroc_on_clip(y_true, y_score, alpha=bedroc_alpha)

    if verbose:
        logger.info(f'Found {len(auroc)} columns with both positive and negative samples.')
        logger.info(f'Found and skipped {counter_invalid} columns with only positive or negative samples.')

    if ret_dict:
        return {'argmax_j':argmax_j, 'auroc':auroc, 'avgp':avgp, 'neg_avgp':neg_avgp,
                'davgp':davgp, 'dneg_avgp':dneg_avgp, 'auprc':auprc, 'bedroc':bedroc}

    return argmax_j, auroc, avgp, neg_avgp, davgp, dneg_avgp, auprc, bedroc

def top_k_accuracy(y_true, y_pred, k=5, ret_arocc=False, ret_mrocc=False, verbose=False, count_equal_as_correct=False, eps_noise=0):
    """
    partly from http://stephantul.github.io/python/pytorch/2020/09/18/fast_topk/
    count_equal counts equal values as being a correct choice. e.g. all preds = 0 --> T1acc=1
    ret_mrocc ... also return median rank of correct choice
    eps_noise ... if > 0, and noise*eps to y_pred .. recommended e.g. 1e-10 #?Yu
    """
    if eps_noise > 0:
        if torch.is_tensor(y_pred):#?Yu
            y_pred = y_pred + torch.rand(y_pred.shape)*eps_noise
        else:
            y_pred = y_pred + np.random.rand(*y_pred.shape)*eps_noise
    if count_equal_as_correct:
        greater = (y_pred > y_pred[range(len(y_pred)), y_true][:,None]).sum(1) # how many are bigger
    else:
        greater = (y_pred >= y_pred[range(len(y_pred)), y_true][:,None]).sum(1) # how many are bigger or equal
    if torch.is_tensor(y_pred):
        greater = greater.long()
    if isinstance(k, int): k = [k] # pack it into a list
    tkaccs = []
    for ki in k:
        if count_equal_as_correct:
            tkacc = (greater<=(ki-1))
        else:
            tkacc = (greater<=(ki))

        if torch.is_tensor(y_pred):
            tkacc = tkacc.float().mean().detach().cpu().numpy()
        else:
            tkacc = tkacc.mean()
        tkaccs.append(tkacc)
        if verbose:
            print('Top', ki, 'acc:\t', str(tkacc)[:6])
    
    if ret_arocc:
        arocc = greater.float().mean()+1
        if torch.is_tensor(arocc):
            arocc = arocc.detach().cpu().numpy()
        return (tkaccs[0], arocc) if len(tkaccs) == 1 else (tkaccs, arocc)
    if ret_mrocc:
        mrocc = greater.median()+1
        if torch.is_tensor(mrocc):
            mrocc = mrocc.float().detach().cpu().numpy()
        return (tkaccs[0], mrocc) if len(tkaccs) == 1 else (tkaccs, mrocc)
    
    return tkaccs[0] if len(tkaccs) == 1 else tkaccs

#================================================================================================================
#                                       dataloader.py
#================================================================================================================
def get_sparse_indices_and_data(m, i):
    """Get the indices and data of a sparse matrix.
    This function will be called in `InMemoryClamp.getitem_meta_assay`
    
    Params:
    -------
    m: a sparse matrix.
    i: the index of the row for which to extract the non-zero elements.

    Returns:
    -------
    tuple: (indices, data)
        col_indices: the column indices of the non-zero data in row i. (in csr format, only the non-zero elements are stored)
        data: the values of the non-zero elements in row i.
    """
    # `m.data`: the non-zero values of the sparse matrix
    # `m.indices`: the column indices of the non-zero values
    # `m.indptr`: which maps the elements of `data` and `indices` to the rows of the sparse matrix. Explanation: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
    col_indices = m.indices[m.indptr[i]:m.indptr[i+1]] 
    data = m.data[m.indptr[i]:m.indptr[i+1]]
    return col_indices, data

class InMemoryClamp(Dataset):
    """
    Subclass of :class:`torch.utils.data.Dataset` holding activity data, 
    that is, activity triplets, and compound and assay feature vectors. 
    Notice that the compound and assay features are already precompouted and stored in the `encoded_compounds` and `encoded_assays` directories, respectively.
    The current implementation of this class doesn't support feature computation on the fly.

    :class:`InMemoryClamp` supports two different indexing (and iteration) styles. 
    The default style is to itreate over `(compound, assay, activity)` COO triplets, however they are sorted.
    The "meta-assays" style consists in interating over unique compounds using a CSR sparse structure,
    and averaging the feature vectors of the positive and negative assays of each compound. 

    By inheriting from :class:`torch.utils.data.Dataset`, this class must implement at least two methods:
    - :meth:`__len__` to return the size of the dataset.
    - :meth:`__getitem__` to retrieve a single data point from the dataset.
    """

    def __init__(
            self,
            root: Union[str, Path],
            assay_mode: str,
            assay_column_list: str = 'columns_short', 
            compound_mode: str = None, 
            train_size: float = 0.6, # -->self._chunk -->def _find_splits; The train_valid_test ratio would be 6:2:2  ?Yu: it's used yet in current workflow.
            verbose: bool = True
    ) -> None:
        """
        Instantiate the dataset class.

        - The data is loaded in memory with the :meth:`_load_dataset` method.
        - Splits are created separately along compounds and along assays with the :meth:`_find_splits` method. 
            Compound and assay splits can be interwoven with the :meth:`subset` method. 

        Params:
        root: str or :class:`pathlib.Path`
            Path to a directory of ready activity files.
        assay_mode: str
            Type of assay features ("clip", or "lsa").
        assay_column_list: str
            The column list to use for assay features. Default is 'columns_short'.
        train_size: float (between 0 and 1)
            Fraction of compounds and assays assigned to training data.
        verbose: bool
            Be verbose if True.
        """
        self.root = Path(root)
        self.assay_mode = assay_mode
        self.assay_column_list = assay_column_list
        self.compound_mode = compound_mode
        self.train_size = train_size
        self.verbose = verbose

        self._load_dataset()
        self._find_splits() 

        self.meta_assays = False #Yu: remove later if not used.

    def _load_dataset(self) -> None:
        """
        Load prepared dataset from the `root` directory:

        - `activity`: Parquet file containing `(compound, assay, activity)` triplets. Compounds and assays are represented by indices, 
        and thus the file is directly loaded into a :class:`scipy.sparse.coo_matrix` with rows corresponding to compounds and columns corresponding to assays.

        - `compound_names`: Parquet file containing the mapping between the compound index used in `activity` and the corresponding compound name.
        It is loaded into a :class:`pandas.DataFrame`.

        - `assay_names`: Parquet file containing the mapping between the assay index used in `activity` and the corresponding assay name. 
        It is loaded into a :class:`pandas.DataFrame`.

        - `compound_features`: npz file containing the compound features array, where the feature vector for the compound indexed by `idx` is stored in the `idx`-th row. 
        It is loaded into a :class:`scipy.sparse.csr_matrix`.

        - `assay_features`: npy file containing the assay features array, where the feature vector for the assay indexed by `idx` is stored in the `idx`-th row.
        It is loaded into a :class:`numpy.ndarray`.

        Compute the additional basic dataset attributes `num_compounds`, `num_assays`, `compound_features_size`, `assay_features_size`.
        """

        if self.verbose:
            logger.info(f'Load dataset from "{self.root} with {self.assay_mode}" assay features.')

        #======= Load compound data =======
        with open(self.root / 'compound_names.parquet', 'rb') as f:
            self.compound_names = pd.read_parquet(f)
        self.num_compounds = len(self.compound_names)

        #compound_modes = self.compound_mode.split('+') if self.compound_mode is not None else 1
        #if len(compound_modes) > 1:
        #    logger.info('Multiple compound modes are concatenated ')
        #    self.compound_features = np.concatenate([self._load_compound(cm) for cm in compound_modes], axis=1)
        #else:
        self.compound_features = self._load_compound(self.compound_mode)
        # compound_features_size
        if 'graph' in self.compound_mode and (not 'graphormer' in self.compound_mode):
            self.compound_features_size = self.compound_features[0].ndata['h'].shape[1] # in_edge_feats. #? Yu: could be removed if graph is not used finally.
        elif isinstance(self.compound_features, pd.DataFrame):
            self.compound_features_size = 40000 #?Yu: too large?
        else:
            if len(self.compound_features.shape)>1:
                self.compound_features_size = self.compound_features.shape[1]
            else: #In this case, self.compound_features is a 1D array, then it's treated as having only one features per compound.
                self.compound_features_size = 1 

        #======== Load assay data ========
        with open(self.root / 'assay_info.parquet', 'rb') as f:
            self.assay_names = pd.read_parquet(f)
        self.num_assays = len(self.assay_names)

        assay_modes = self.assay_mode.split('+')
        if len(assay_modes)>1:
            logger.info('Multiple assay modes are concatenated')
            self.assay_features = np.concatenate([self._load_assay(am, self.assay_column_list) for am in assay_modes], axis=1)
        else:
            self.assay_features = self._load_assay(self.assay_mode, self.assay_column_list)

        # assay_features_size
        if (self.assay_features is None):
            self.assay_features_size = 512 #wild guess also 512
        elif len(self.assay_features.shape)==1:
            # its only a list, so usually text
            self.assay_features_size = 768  # a common default for models like BERT.
        else: # usually a 2D array
            self.assay_features_size = self.assay_features.shape[1]

        #======= Load activity data =======
        with open(self.root / 'activity.parquet', 'rb') as f:
            activity_df = pd.read_parquet(f)
            self.activity_df = activity_df
        
        self.activity = sparse.coo_matrix(
            (
                activity_df['activity'],# activity is the value
                (activity_df['compound_idx'], activity_df['assay_idx']) # compound in row, assay in column.
            ),
            shape=(self.num_compounds, self.num_assays),
        )
    
    def _load_compound(self, compound_mode=None):
        cmpfn = f'encoded_compounds/compound_features{"_"+compound_mode if compound_mode else ""}'
        logger.info(f'cmpfn: {cmpfn}')
        #?Yu: if 'graph' is not used, remove the below code
        if 'graph' in compound_mode and (not 'graphormer' in compound_mode):
            logger.info(f'graph in compound mode: loading '+cmpfn)
            import dgl
            from dgl.data.utils import load_graphs
            compound_features = load_graphs(str(self.root/(cmpfn+".bin")))[0]
            compound_features = np.array(compound_features)
        elif compound_mode == 'smiles':
            compound_features = pd.read_parquet(self.root/('compound_smiles.parquet'))['CanonicalSMILES'].values
        else:
            try: #tries to open npz files else npy
                with open(self.root/(cmpfn+".npz"), 'rb') as f:
                    compound_features = sparse.load_npz(f)
            except:
                logger.info(f'loading '+cmpfn+'.npz failed, using .npy instead')
                try:
                    logger.info(f'loading '+cmpfn+'.npy')
                    compound_features = np.load(self.root/(cmpfn+".npy"))
                except:
                    logger.info(f'loading '+cmpfn+'.npy failed, trying to compute it on the fly')
                    compound_features = pd.read_parquet(self.root/('compound_smiles.parquet'))
        return compound_features

    def _load_assay(self, assay_mode='lsa', assay_column_list='columns_short') -> None:
        """ loads assay """
        if assay_mode =='':
            print('no assay features')
            return None
        
        #? Yu: if the below assay modes are not used, remove them.
        if assay_mode == 'biobert-last':
            with open(self.root/('assay_features_dmis-lab_biobert-large-cased-v1.1_last_layer.npy'), 'rb') as f:
                return np.load(f, allow_pickle=True)
        elif assay_mode == 'biobert-two-last':
            with open(self.root/('assay_features_dmis-lab_biobert-large-cased-v1.1_penultimate_and_last_layer.npy'), 'rb') as f:
                return  np.load(f, allow_pickle=True)
        
        # load the prepared assay features
        asyfn = f'encoded_assays/assay_features_{assay_mode}_{assay_column_list}'
        try: # tries to open npz file else npy
            with open(self.root/(asyfn+".npz"), 'rb') as f:
                return sparse.load_npz(f)
        except:
            with open(self.root/(asyfn+".npy"), 'rb') as f:
                return np.load(f, allow_pickle=True)
        
        return None

    def _find_splits(self) -> None:
        """
        # Yu: modify this to enable cross-validation splits.
        We assume that during the preparation of the PubChem data, compounds(assays) have been indexed 
        so that a larger compound(assay) index corresponds to a compound(assay) incorporated to PubChem later in time.
        This function finds the compound(assay) index cut-points to create three chronological disjoint splits.

        The oldest `train_size` fraction of compounds(assays) are assigned to training. 
        From the remaining compounds(assays), the oldest half are assigned to vailidation, and the newest half are assigned to test.
        Only the index cut points are stored.
        """
        if self.verbose:
            logger.info(f'Find split cut-points for compound and assay indices (train_size={self.train_size}).')

        first_cut, second_cut = self._chunk(self.num_compounds, self.train_size)
        self.compound_cut = {'train': first_cut, 'valid': second_cut}

        first_cut, second_cut = self._chunk(self.num_assays, self.train_size)
        self.assay_cut = {'train': first_cut, 'valid': second_cut}

    @staticmethod
    def _chunk(n:int, first_cut_ratio:float) -> Tuple[int, int]:
        """
        # Yu: modify this to enable cross-validation splits.
        Find the two cut points required to chunk a sequence of `n` items into three parts, 
        the first having `first_cut_ratio` of the items, 
        the second and the third having approximately the half of the remaining items.

        Params
        -------
        n: int
            Length of the sequence to chunk.
        first_cut_ratio: float
            Portion of items in the first chunk. This is the `train_size`
        
        Returns
        -------
        int, int
            Positions where the first and second cut occurs.
        """
        first_cut = int(round(first_cut_ratio * n))
        second_cut = first_cut + int(round((n - first_cut) / 2))

        return first_cut, second_cut

    def subset(
            self, 
            c_low: Optional[int] = None,
            c_high: Optional[int] = None,
            a_low: Optional[int] = None,
            a_high: Optional[int] = None,
    ) -> np.ndarray:
        """
        #Yu: will be used when the `split` is set as `time_a_c`, which requires that assay and compound ids are sorted by time.
        #Yu: Remove this function if the `split` is not set as `time_a_c` in tha later benchmarking run.
        """
        if c_low is None: # set the compound low index to 0
            c_low = 0
        if c_high is None: # set the compound high index to the number of compounds
            c_high = self.num_compounds
        if a_low is None: # set the assay low index to 0
            a_low = 0
        if a_high is None: # set the assay high index to the number of assays
            a_high = self.num_assays

        if self.verbose:
            logger.info(f'Find activity triplets where {c_low} <= compound_idx <= {c_high} and {a_low} <= assay_idx <= {a_high}.')
        
        activity_bool = np.logical_and.reduce( # take multiple Boolean conditions and combines them using logical AND across all conditions.
            (
                self.activity.row >= c_low,
                self.activity.row <= c_high,
                self.activity.col >= a_low,
                self.activity.col <= a_high
            )
        )

        # applies the logical condition to the COO matrix and returns the indices that satisfy the condition. `flatnonzero` returns the indices of the non-zero elements, where is True(1).
        return np.flatnonzero(activity_bool) # return a 1D array of indices of activity where all the four conditions are True.

    def get_unique_names(
            self, 
            activity_idx: Union[int, Iterable[int], slice]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        #Yu: This function in only used in `def test` when 'Multitask' is in hparams['model']. Remove this function if 'Multitask' is not used in the later benchmarking run.

        Get the unique compound and assay names within the `activity` triplets  indexed by `activity_idx` in default, COO style.

        Params:
        -------
        activity_idx: int, iterable of int, slice
            Index to one or multiple `activity` triplets.
        
        Returns:
        -------
        compound_names: :class:`pandas.DataFrame`
        assay_names: :class:`pandas.DataFrame`
        """

        compound_idx = self.activity.row[activity_idx]
        assay_idx = self.activity.col[activity_idx]

        if isinstance(compound_idx, np.ndarray) and isinstance(assay_idx, np.ndarray):
            compound_idx = pd.unique(compound_idx)
            assay_idx = pd.unique(assay_idx)
        
        elif isinstance(compound_idx, (int, np.integer)) and isinstance(assay_idx, (int, np.integer)):
            pass # a single index means a single compound and assay, so no need to do anything.

        else:
            raise ValueError('activity_idx must be an int, iterable of int, or slice.')

        compound_names = self.compound_names.iloc[compound_idx]
        assay_names = self.assay_names.iloc[assay_idx]

        return compound_names.sort_index(), assay_names.sort_index() # sort the names alphabetically

    def getitem(
            self,
            activity_idx: Union[int, Iterable[int], slice],
            ret_np=False
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:#Yu：check the number of return values.
        """
        Get the data for the activity triplets indexed by `activity_idx` in default, COO style.
        
        Params
        -------
        activity_idx: int, iterable of int, slice
            Specifies the indices of the activity triplets to retrieve.
        ret_np: bool
            Determines the format of the returned data. If True, returns numpy arrays; If False, returns PyTorch tensors.
        
        Returns
        -------
        tuple of :class:`torch.Tensor`
        - `activity_idx`: the original indices provided as input. This will enable to reconstruct the order in which the dataset has been visited.
        - `compound_features`: shape(len(activity_idx), compound_features_size)
        - `assay_features`: shape(len(activity_idx), assay_features_size)
        - `activity`: shape(len(activity_idx), ).
        """
        compound_idx = self.activity.row[activity_idx]
        assay_idx = self.activity.col[activity_idx]
        activity = self.activity.data[activity_idx]

        # ===== get compound_features =====
        if isinstance(self.compound_features, pd.DataFrame):
            #logger.info('Compound features are stored in a DataFrame, converting to numpy array.')
            compound_smiles = self.compound_features.iloc[compound_idx]['CanonicalSMILES'].values
            from datacat4ml.Scripts.data_prep.data_featurize.compound_featurize.encode_compound  import convert_smiles_to_fp
            if self.compound_mode == 'MxFP':
                fptype = 'maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+mhfp+rdkd'
            else:
                fptype = self.compound_mode
            # Todo: fp_size as input parameter
            fp_size = 40000 #
            compound_features = convert_smiles_to_fp(compound_smiles, fp_size=fp_size, which=fptype, radius=2, njobs=1).astype(np.float32)
        else:
            #logger.info('Compound features are stored in a sparse matrix, converting to numpy array.')
            compound_features = self.compound_features[compound_idx]
            if isinstance(compound_features, sparse.csr_matrix):
                compound_features = compound_features.toarray()
        

        # ===== get assay_features =====
        assay_features = self.assay_features[assay_idx]
        if isinstance(assay_features, sparse.csr_matrix):
            assay_features = assay_features.toarray()

        # ===== Handle single indices =====
        # If `activity_idx`is a single integer or a list with only one element, the retrieved feature vectors are reshaped into 1D arrays to maintain the consistency of the output format.
        if isinstance(activity_idx, (int, np.integer)):
            compound_features = compound_features.reshape(-1) 
            assay_features = assay_features.reshape(-1) 
            activity = [activity]
        elif isinstance(activity_idx, list):
            if len(activity_idx) == 1:
                compound_features = compound_features.reshape(-1)
                assay_features = assay_features.reshape(-1)
        activity = np.array(activity)

        # ===== Return =====
        # return the data as Numpy arrays.
        if ret_np:
            return(
                activity_idx,
                compound_features, #already float32
                assay_features if not isinstance(assay_features[0], str) else assay_features, # already float32
                (float(activity)) # torch.nn.BCEWithLogitsLoss needs this to be float.
            )

        # return the data as PyTorch tensors.
        if self.compound_mode == 'smiles':
            comp_feat = compound_features
        elif isinstance(compound_features, np.ndarray):
            #print(f"compound_features type: {type(compound_features)}")
            #print(f"compound_features dtype: {getattr(compound_features, 'dtype', 'not an ndarray')}")
            #print(f"compound_features[0]: {compound_features[0]}")
            comp_feat = torch.from_numpy(compound_features)
        elif not isinstance(compound_features[0], dgl.DGLGraph): #Yu: remove this if 'graph' is not used.
            comp_feat = dgl.batch(compound_features)
        else:
            comp_feat = compound_features

        return  (
            activity_idx, 
            comp_feat, # alread float32
            torch.from_numpy(assay_features) if not isinstance(assay_features[0], str) else assay_features, # already float32
            torch.from_numpy(activity).float() # torch.nn.BCEWithLogitsLoss needs this to be float too...
        )

    def getitem_meta_assay(
            self,
            compound_idx: Union[int, List[int], slice]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For a given compound (or list), retrieve the data in the `meta-assay` style, 
        which involves summarizing assay feature vectors (positive and negative) for each compound.

        #Yu: This function is neither used in the primary code or current workflow,thus remove it if not used in the later benchmarking run.
        
        Params
        -------
        compound_idx: int, iterable of int, slice
            Index to one or multiple compounds.
        
        Returns
        -------
        tuple of :class:`torch.Tensor`
        - `compound_features`: shape(N, compound_features_size)
        - `assay_features`: shape(N, assay_features_size)
        - `activity`: shape(N, )
        """
        
        # extract the data for the specified compounds
        activity_slice = self.activity.tocsr()[compound_idx] 

        # find non-empty rows
        # `activity_slic.indptr`: pointer to the start of each row in the sparse matrix.
        # `np.diff(activity_slice.indptr)`: measures the number of elements in each row.
        # `np.where(...!=0)`: finds rows that contain non-zero elements. (i.e., rows with at least one assay-related to the compound)`
        non_empty_row_idx = np.where(np.diff(activity_slice.indptr)!=0)[0] # [0] refer to the rows of the sparse matrix, i.e., compounds.

        # initialize containers for results
        compound_features_l = [] # list of compound features
        assay_positive_features_l, assay_negative_features_l = [], [] # averaged features of positive assays, and negative assays
        activity_l = [] # activity lables

        # process each non-empty row
        for row_idx in non_empty_row_idx:
            positive_l, negative_l  = [], []
            for col_idx, activity in get_sparse_indices_and_data(activity_slice, row_idx):
                if activity == 0:
                    negative_l.append(self.assay_features[col_idx]) 
                else:
                    positive_l.append(self.assay_features[col_idx])
            
            if len(negative_l) > 0:
                compound_features_l.append(self.compound_features[row_idx])
                negative = np.vstack(negative_l).mean(axis=0)
                assay_negative_features_l.append(negative)
                activity_l.append(0)
            
            if len(positive_l) > 0:
                compound_features_l.append(self.compound_features[row_idx])
                positive = np.vstack(positive_l).mean(axis=0)
                assay_positive_features_l.append(positive)
                activity_l.append(1)

        compound_features = sparse.vstack(compound_features_l).toarray()
        assay_features_l = np.vstack(
            assay_negative_features_l + assay_positive_features_l # '+' is used to concatenate the two lists
        )

        activity = np.array(activity_l)

        return (
            torch.from_numpy(compound_features), # already float32
            torch.from_numpy(assay_features_l), # already float32
            torch.from_numpy(activity).float() # torch.nn.BCEWithLogitsLoss needs this to be float too...
        )

    @staticmethod
    def collate(batch_as_list:list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Necessary for :meth:`getitem_meta_assay` if using a :class:`torch.utils.data.DataLoader`.
        Not necessary if using :class:`torch.utils.data.BatchSampler`.

        Params
        -------
        batch_as_list: list
            Result of :meth:`getitem_meta_assay` for a mini-batch.

        Returns
        -------
        tuple of :class:`torch.Tensor`
            Data for a mini-batch.
        """
        compound_features_t, assay_features_t, activity_t = zip(*batch_as_list)
        return(
            torch.cat(compound_features_t, dim=0),
            torch.cat(assay_features_t, dim=0),
            torch.cat(activity_t, dim=0)
        )
        

    def __getitem__(self, idx: Union[int, Iterable[int], slice]) -> Tuple:
        """
        Index or slice `activity` by `idx`. The indexing mode depends on the value of `meta_assays`. 
        If False(default), the indexing is over COO triplets.
        If True, the indexing is over unique compounds.
        """

        if self.meta_assays:
            return self.getitem_meta_assay(idx)
        else:
            return self.getitem(idx)
        
    def __len__(self) -> int:
        """
        Return the length of the dataset.

        - If `meta_assays` is False (default), length is defined as the number of `(compound, assay, activity)` COO triplets.
        - If `meta_assays` is True, length is defined as the number of the unique compounds.
        """

        if self.meta_assays:
            return self.num_compounds
        else:
            return self.activity.nnz # the number of non-zero elements in the sparse matrix
        
    def __repr__(self):
        return f'InMemoryClamp\n' \
               f'\troot="{self.root}"\n' \
               f'\tcompound_mode="{self.compound_mode}"\n' \
               f'\tassay_mode="{self.assay_mode}"\n' \
               f'\ttrain_size={self.train_size}\n' \
               f'\tactivity.shape={self.activity.shape}\n' \
               f'\tactivity.nnz={self.activity.nnz}\n' \
               f'\tmeta_assays={self.meta_assays}'
    
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
    'l2': float, #?Yu: L2 regularization?
    'dropout_input': float,
    'dropout_hidden': float,
    'loss_fun': str,
    'lr_factor': float,
    'batch_size': int,
    'assay_mode': str,
    'warmup_epochs': int,
    'multitask_temperature': float, #?Yu: if not used later, remove
    'epoch_max': int, 
    'nonlinearity': str,
    'pooling_mode': str,
    'attempts': int, # not used in public version #?Yu: don't understand, remove?
    'tokenizer': str,
    'transformer': str,
    'train_balanced': int,
    'beta': float,
    'norm': bool,
    'label_smoothing': float,
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
    'train_subsample': float, #?Yu:?
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
    # Early stopping is a regularization technique to prevent overfitting. 
    # During training, it monitors the validation loss and stops training when the validation loss does not improve for a specified number of epochs (patience).
    # This helps the model to generalize better rather than just memorizing the training data.
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
    model = getattr(model_def, hp['model'])( #?Yu model_def: class, hp['model']: the specified attribute. 
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
    print(f'Function signature and parameters:\n early_stopping={early_stopping}')
    # metrics
    bedroc_alpha = hparams.get('bedroc_alpha')

    # ================================= Model initialization =================================
    print(hparams)
    
    #?Yu: Regard different assays or targets as different tasks. Keep the below `Multitask` related code if used later, otherwise remove it.
    #?Yu: why `setup_assay_onehot` is used here?`
    if 'Multitask' in hparams.get('model'):

        _, train_assays = InMemory.get_unique_names(train_idx)
        InMemory.setup_assay_onehot(size=train_assays.index.max() + 1)
        train_assay_features = InMemory.assay_features[:train_assays.index.max() + 1] #?Yu: no `assay_features` defined neither in the primary code or 'InMemoryClamp` before.
        train_assay_features_norm = F.normalize(torch.from_numpy(train_assay_features), #?Yu: why set this here but use it quite later?
            p=2, dim=1 #Yu: p=2: the exponent value in the norm formulation; dim=1: the dimension to reduce.
        ).to(device)

        model = init_dp_model(
            compound_features_size=InMemory.compound_features_size,
            assay_features_size=InMemory.assay_onehot.size, #?Yu: `assay_onehot` has not been defined in the `InMemoryClamp` class before.
            hp=hparams,
            verbose=verbose
        )
    
    else:
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
            logger.infp('Load optimizer_state_dict from checkpoint into optimizer.')
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # ================================= Loss function initialization =================================
    # initialize loss function #Yu: the core of clamp.
    criterion = nn.BCEWithLogitsLoss() # default, allowing `loss_fun` to be optional.
    if 'loss_fun' in hparams:
        class CustomCE(nn.CrossEntropyLoss):
            """Cross entropy loss #?Yu"""
            def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                """
                param
                -------
                input: predicted unnormalized logits. This is the raw output (logits, i.e. preactivations, no softmax/sigmoid) from the model, typically of shape [batch_size, batch_size] in contrastive/self-supervised settings.
                target: ground truth class indices or class probabilities.

                return
                -------
                for `F.cross_entropy`:
                weight: a manual rescaling weight given to each class.
                """
                beta = 1/(input.shape[0]**(1/2)) # scaling factor, normalizes the logits so that their magnitude is independent of batch size, which can help stabilize training.
                input = input * (target*2-1) * beta # 
                target = torch.arange(0, len(input)).to(input.device)

                return F.cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
            
        class ConLoss(nn.CrossEntropyLoss):
            """Contrastive Loss"""
            def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                sigma = 1 # scaling factor. set to 1 here, so it does not affect the result.
                bs = target.shape[0] #?Yu
                #Yu: remove the below code if not used
                #only modify diag that is a negative
                # eg makes this from a target of [0, 1, 0]
                # tensor([[-1., 1., 1.],
                #         [1., 1., 1.],
                #         [1., 1., -1.]])
                modif = (1-torch.eye(bs)).to(target.device) + (torch.eye(bs).to(target.device)*(target*2-1)) # `torch.eye`: returns a 2-D tensor with ones on the diagonal and zeros elsewhere.`bs` is the number of rows.
                input = input*modif/sigma
                diag_idx = torch.arange(0, len(input)).to(input.device)

                label_smoothing = hparams.get('label_smoothing', 0.0)
                if label_smoothing is None:
                    label_smoothing = 0.0
                
                mol2txt = F.cross_entropy(input, diag_idx, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=label_smoothing)
                text2mol = F.cross_entropy(input.T, diag_idx, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, label_smoothing=label_smoothing)

                return mol2txt + text2mol
            
        str2loss_fun = {
            'BCE': nn.BCEWithLogitsLoss(),
            'CE': CustomCE(),
            'Con': ConLoss(),
        }
        assert hparams['loss_fun'] in str2loss_fun, "loss_fun not implemented"
        criterion = str2loss_fun[hparams['loss_fun']]

    criterion = criterion.to(device)
        
    # ================================= Learning rate Scheduler =================================
    # lambda function below returns `lr_factor` whatever the input to lambda is.
    if 'lr_factor' in hparams:
        lr_factor = hparams['lr_factor']
    else:
        lr_factor = 1 #?Yu: why
    scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda _: lr_factor)

    #lot_lr_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=1000, eta_min=0)
    num_steps_per_epoch = len(train_idx)/hparams['batch_size']
    class Linwarmup():
        def __init__(self, steps=10000):
            self.step = 0
            self.max_step = steps
            self.step_size = 1/steps
        def get_lr(self, lr):
            if self.step>self.max_step:
                return 1
            new_lr = self.step * self.step_size
            self.step += 1
            return new_lr
        
    #Todo Bug when set to 0
    if hparams.get('warmup_step'): #?Yu: why not `if hparams['warmup_step']`?
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                       lr_lambda=Linwarmup(steps=num_steps_per_epoch + hparams.get('warmup_epochs', 0)).get_lr)
    else:
        scheduler2 = None
    
    if lr_factor !=1:
        logger.info(f'Scheduler enabled with lr_factor={hparams["lr_factor"]}. Note that this makes different runs difficult to compare.')
    else:
        logger.info('Scheduler enabled with lr_factor=1. This keeps the interface but results in no reduction.')
    # ================================= Batch sampler =================================
    # The acutual dataset slicing is actually done manually.
    train_sampler = RandomSampler(data_source=train_idx) #?Yu: why not implement train_batcher like 'valid_batcher' and 'test_batcher'?

    valid_sampler = SequentialSampler(data_source=valid_idx)
    valid_batcher = BatchSampler(sampler=valid_sampler, batch_size=hparams['batch_size'], drop_last=False)

    test_sampler = SequentialSampler(data_source=test_idx)
    test_batcher = BatchSampler(sampler=test_sampler, batch_size=hparams['batch_size'], drop_last=False)

    epoch = checkpoint.get('epoch', 0) #?Yu: why 0
    new_train_idx = None #?Yu: why
    while epoch < checkpoint.get('epoch', 0) + hparams['epoch_max']:
        if hparams.get('train_balanced', False):
            logger.info('sampling balanced')
            num_pos = InMemory.activity.data[train_idx].sum() #Yu: why .sum()
            #?Yu: remove the below comment later
            # too large with WeightedRandomSampler
            # num_neg = (len(train_idx))-num_pos
            remove_those = train_idx[((InMemory.activity.data[train_idx]) == 0)]
            remove_those = np.random.choice(remove_those, size=int(len(remove_those)-num_pos)) #?Yu
            idx = np.in1d(train_idx, remove_those) #?Yu
            new_train_idx = train_idx[~idx] #?Yu
            if isinstance(hparams['train_balanced'], int):
                max_samples_per_epoch = hparams['train_balanced']
                if max_samples_per_epoch > 1:
                    logger.info(f'using only {max_samples_per_epoch} for one epoch')
                    new_train_idx = np.random.choice(new_train_idx, size=max_samples_per_epoch)
            train_sampler = RandomSampler(data_source=new_train_idx)
        if hparams.get('train_subsample', 0) > 0: #?Yu: why not `elif`
            if hparams['train_subsample']<1:
                logger.info(f'subsample training set to {hparams["train_subsample"]*100}%')
                hparams['train_subsample'] = int(hparams['train_subsample']*len(train_idx))
            logger.info(f'subsample training set to {hparams["train_subsample"]}')
            sub_train_idx = np.random.choice(train_idx if new_train_idx is None else new_train_idx, size=int(hparams['train_subsample']))
            train_sampler = RandomSampler(data_source=sub_train_idx)
        
        train_batcher = BatchSampler(sampler=train_sampler, batch_size=hparams['batch_size'], drop_last=False)
 
        # ================================= Training loop =================================
        print(f'============================\n Starting training: epoch {epoch} \n============================')
        loss_sum = 0.
        preactivations_l = []
        topk_l, arocc_l = [], []
        activity_idx_l = []
        for batch_num, batch_indices in enumerate(train_batcher):

            # get and unpack batch data
            batch_data = Subset(InMemory, indices=train_idx)[batch_indices]
            activity_idx, compound_features, assay_features, activity = batch_data #?Yu: what is no assay_onehot?

            # move data to device
            #?Yu: remove the below comments if not used
            # assignment is not necessary for modules but it is for tensors.
            # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
            if isinstance(compound_features, torch.Tensor):
                compound_features = compound_features.to(device)
            assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features #?Yu: why not using same method that is used for `compound_features`?
            #assay_onehot = assay_onehot.to(device).float() if not isinstance(assay_onehot[0], str) else assay_onehot #?Yu
            activity = activity.to(device)

            # forward
            #with torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float32): #?Yu: shall I keep this comment?
            if hparams.get('loss_fun') in ('CE', 'Con'): # why in the two cases, `forward_dense` is used?
                preactivations = model.forward_dense(compound_features, #?Yu: go to check the difference between 'forward' and 'forward_dense'
                                                     assay_features) #?Yu: consider whether to remove 'assay_onehot'
            else:
                preactivations = model(compound_features, assay_features)
            
            # loss
            beta = hparams.get('beta', 1)
            if beta is None: beta = 1
            preactivations = preactivations*1/beta #?Yu
            loss = criterion(preactivations, activity)

            # zero gradients, backpropagation, update
            optimizer.zero_grad()
            loss.backward()
            if hparams.get('optimizer') == 'SAM': #?Yu
                def closure():
                    preactivations = model(compound_features, assay_features) # why compute preactivation again?
                    loss = criterion(preactivations, activity)
                    loss.backward()
                    return loss 
                optimizer.step(closure)
            else:
                optimizer.step()
                scheduler.step()
                if scheduler2: scheduler2.step() #?Yu

            # accumulate loss 
            loss_sum += loss.item()

            if hparams.get('loss_fun')=='Con':
                ks = [1, 5, 10, 50] #?Yu
                tkaccs, arocc = top_k_accuracy(torch.arange(0, len(preactivations)), preactivations, k=[1, 5, 10, 50], ret_arocc=True)
                topk_l.append(tkaccs)
                arocc_l.append(arocc)
            if hparams.get('loss_fun') in ('CE', 'Con'):
                #preactivations = preactivations.sum(axis=1) #?Yu: why keep it here?
                preactivations =torch.diag(preactivations) # get only diag elements

            # accumulate preactivations
            # - need to detach; preactivations.requires_grad = True
            # - move it to cpu #?Yu
            preactivations_l.append(preactivations.detach().cpu())

            # accumulate_indices to track order in which the dataset is visited
            # - activity_idx is a np.array, not a torch.tensor
            activity_idx_l.append(activity_idx)

            if batch_num % EVERY == 0 and verbose: 
                logger.info(f'Epoch{epoch}: Training batch {batch_num} out of {len(train_batcher) - 1}.')

        # log mean loss over all minibatches
        mlflow.log_metric('train_loss', loss_sum / len(train_batcher), step=epoch)
        if wandb.run:
            wandb.log({
                'train/loss': loss_sum / len(train_batcher),
                'lr': scheduler2.get_last_lr()[0] if scheduler2 else scheduler.get_last_lr()[0]
            }, step=epoch)

        # compute metrics for each assay (on the cpu)
        preactivations = torch.cat(preactivations_l, dim=0)
        print(f'preactivations.shape: {preactivations.shape}; \n preactivations: {preactivations[:5]}')
        probabilities = torch.sigmoid(preactivations).numpy()
        print(f'probabilities.shape: {probabilities.shape}; \n probabilities: {probabilities[:5]}') 

        activity_idx = np.concatenate(activity_idx_l, axis=0)
        print(f'The length of activity_idx for training: {len(activity_idx)}, including {activity_idx}')
        # assert np.array_equal(np.sort(activity_idx), train_idx)
        # assert not np.array_equal(activity_idx, train_idx)

        targets = sparse.csc_matrix(
            (
                InMemory.activity.data[activity_idx],
                (
                    InMemory.activity.row[activity_idx],
                    InMemory.activity.col[activity_idx]
                )
            ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.bool_
        )

        scores = sparse.csc_matrix(
            (
                probabilities, #?Yu: why is probabilities used here?
                (
                    InMemory.activity.row[activity_idx],
                    InMemory.activity.col[activity_idx]
                )
            ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.float32
        )

        #?Yu: `metrics` should be changed according to my implementation.
        #md = metrics.swipe_threshold_sparse(targets=targets, scores=scores,verbose=verbose>=2, ret_dict=True) # returns dict for with metric per assay in the form of {metric: {assay_nr: value}} #?Yu: isn't `verbose=verbose>=2` syntax error?
        md = swipe_threshold_sparse(targets=targets, scores=scores, bedroc_alpha=bedroc_alpha, verbose=verbose>=2, ret_dict=True)

        if hparams.get('loss_fun') == 'Con':
            for ii, k in enumerate(ks):#?Yu: `ks` is not defined in this loop.
                md[f'top_{k}_acc'] = {0:np.vstack(topk_l)[:-1, ii].mean()} # drop last (might be not full) #?Yu why
            md['arocc'] = {0:np.hstack(arocc_l)[:-1].mean()} # drop last (might be not full) #?Yu why

        logdic = {f'train_mean_{k}': np.nanmean(list(v.values())) for k,v in md.items() if v}
        mlflow.log_metrics(logdic, step=epoch) #?Yu: go through the use of mlflow in `train_utils.py``
        if wandb.run: wandb.log({k.replace('_', '/'):v for k, v in logdic.items()}, step=epoch)
        # if verbose: logger.info(logdic) #?Yu: why not print the logdic？

        # ================================= Validation loop =================================
        print(f'============================\n Starting validation \n============================')
        with torch.no_grad():
            
            model.eval()

            loss_sum = 0.
            preactivations_l = [] #?Yu: will this overwrite the ones in the training loop?
            activity_idx_l = []
            for batch_num, batch_indices in enumerate(valid_batcher):

                # get and unpack batch data
                batch_data = Subset(InMemory, indices=valid_idx)[batch_indices]
                activity_idx, compound_features, assay_features, activity = batch_data #?Yu why isn't here `assay_onehot`?

                # move data to device
                # assignment is not necessary for modules but it is for tensors.
                # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
                if isinstance(compound_features, torch.Tensor):
                    compound_features = compound_features.to(device)
                assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features # why is the conditions different between 'assay_features' and 'compound_features'.
                activity = activity.to(device)

                # forward #?Yu: why the `Multitask` related code is here but not in the training loop?`
                if 'Multitask' in hparams['model']:
                    assay_features_norm = F.normalize(
                        assay_features, p=2, dim=1
                    )
                    sim_to_train = assay_features_norm @ train_assay_features_norm.T #?Yu
                    sim_to_train_weights = F.softmax(sim_to_train * hparams['multitask_temperature'], dim = 1) #?Yu: what does `F.softmax` do?
                    preactivations = model(compound_features, sim_to_train_weights)
                
                elif hparams.get('loss_fun') in ('CE', 'Con'):
                    preactivations = model.forward_dense(compound_features, assay_features) #?Yu: 'assay_onehot' is not defined in the validation loop.
                else:
                    preactivations = model(compound_features, assay_features)

                # loss
                preactivations = preactivations * 1 / hparams.get('beta', 1) #?Yu
                #?Yu Why is the below code block commented out in the primary code.
                if hparams.get('loss_fun') in ('CE', 'Con'):
                    loss = F.binary_cross_entropy_with_logits(preactivations, activity)
                else:
                    loss = criterion(preactivations, activity)
                
                # accumulate loss
                loss_sum += loss.item()

                if hparams.get('loss_fun') in ('CE', 'Con'): # how to calc the below metrics if `loss_fun` is neither 'CE' nor 'Con'?
                    ks = [1, 5, 10, 50]
                    tkaccs, arocc = top_k_accuracy(torch.arange(0, len(preactivations)), preactivations, k=[1, 5, 10, 50], ret_arocc=True)
                    topk_l.append(tkaccs) # already detached numpy #?Yu
                    arocc_l.append(arocc)
                
                # accumulate preactivations
                #Yu: remove the below comments if not used
                # - need to detach; preactivations.requires_grad is True
                # - move it to cpu
                if hparams.get('loss_fun') in ('CE', 'Con'): #?Yu: combine this condition with the one above? and similarily, how to calc preactivations if `loss_fun` is neither 'CE' nor 'Con'?
                    # preactivations = preactivations.sum(axis=1)
                    preactivations = torch.diag(preactivations) #?Yu: `torch.diag`

                preactivations_l.append(preactivations.detach().cpu())

                # accumulate indices just to double check.
                # - activity_idx is a np.array, not a torch.tensor
                activity_idx_l.append(activity_idx)

                if batch_num % EVERY == 0 and verbose:
                    logger.info(f'Epoch{epoch}: Validation batch {batch_num} out of {len(valid_batcher) -1}.')

            # log mean loss over all minibatches
            valid_loss = loss_sum / len(valid_batcher)
            mlflow.log_metric('valid_loss', valid_loss, step=epoch) #?Yu: sort the use of mlflow along the code.
            if wandb.run: wandb.log({'valid/loss': valid_loss}, step=epoch)
            
            # compute test auroc and avgp for each assay (on the cpu)
            preactivations = torch.cat(preactivations_l, dim=0)
            print(f'preactivations.shape: {preactivations.shape}') 
            probabilities = torch.sigmoid(preactivations).numpy()

            activity_idx = np.concatenate(activity_idx_l, axis=0)
            print(f'The length of activity_idx for validation: {len(activity_idx)}, including {activity_idx}')
            # assert np.array_equal(activity_idx, valid_idx) #?Yu: why is this line commented out in the primary code?

            targets = sparse.csc_matrix(
                (
                    InMemory.activity.data[valid_idx],
                    (
                        InMemory.activity.row[valid_idx],
                        InMemory.activity.col[valid_idx]
                    )
                ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.bool_
            )

            scores = sparse.csc_matrix(
                (
                    probabilities,
                    (
                        InMemory.activity.row[valid_idx],
                        InMemory.activity.col[valid_idx]
                    )
                ), shape=(InMemory.num_compounds, InMemory.num_assays), dtype=np.float32
            )

            #md = metrics.swipe_threshold_sparse(targets=targets, scores=scores, verbose=verbose>=2, ret_dict=True)
            md = swipe_threshold_sparse(targets=targets, scores=scores, bedroc_alpha=bedroc_alpha, verbose=verbose>=2, ret_dict=True)

            if hparams.get('loss_fun') == 'Con': #?Yu: what if 'loss_fun' is 'Con'
                #?Yu: how about other metrics?
                for ii, k in enumerate(ks): #?Yu: `ks` is not defined in this loop.
                    md[f'top_{k}_acc'] = {0:np.vstack(topk_l)[:-1, ii].mean()} # drop last (might be not full)
                md['arocc'] = {0:np.hstack(arocc_l)[:-1].mean()} # drop last (might be not full)
            
            # log metrics mean over assays #?Yu: modify here to calc metrics on OR datasets.

            logdic = {f'valid_mean_{k}': np.nanmean(list(v.values())) for k,v in md.items() if v} #?Yu: what is logdic?
            logdic['valid_loss'] =valid_loss

            mlflow.log_metrics(logdic, step=epoch)

            if wandb.run: wandb.log({k.replace('_', '/'):v for k, v in logdic.items()}, step=epoch)
            # if verbose: logger.info(logdic)

            # monitor metric
            evaluation_metric = 'valid_mean_davgp'

            if evaluation_metric not in logdic:
                logger.info('Using -valid_loss because valid_mean_avgp not in logdic.')
            log_value = logdic.get(evaluation_metric, -valid_loss) #?Yu: why -valid_loss?
            # metric_monitor(logdic['valid_mean_davgp'], epoch)
            do_early_stop = early_stopping(-log_value) # smaller is better #?Yu: why -log_value?
            print(f'Validation loop: \n do_early_stop={do_early_stop}')

            # log model checkpoint dir
            if wandb.run:
                wandb.run.config.update({'model_save_dir':checkpoint_file_path})
            
            if early_stopping.improved:
                logger.info(f'Epoch {epoch}: Save model and optimizer checkpoint with val-davgp: {log_value}.')
                torch.save({
                    'value': log_value,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_file_path)
            

            if do_early_stop: #?Yu is this set by me or during the training loop?
                logger.info(f'Epoch {epoch}: Out of patience. Early stop!')
                break

            model.train() #?Yu: how this code line connect with the code above.
        
        epoch +=1
    
    # ================================= Testing loop =================================
    print(f'============================\n Starting testing: epoch {epoch} \n============================')

    # test with best model
    with torch.no_grad():
        
        epoch -= 1 #?Yu how can this be the best model
        logger.info(f'Epoch {epoch}: Restore model from checkpoint.')
        # check if checkpoint exists
        if not os.path.exists(checkpoint_file_path):
            logger.warning(f'Checkpoint file {checkpoint_file_path} does not exist. Test with init model.')
        else:
            checkpoint = torch.load(checkpoint_file_path)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()

        loss_sum = 0.
        preactivations_l = []
        activity_idx_l = []
        for batch_num, batch_indices in enumerate(test_batcher):

            # get and unpack batch data
            batch_data = Subset(InMemory, indices=test_idx)[batch_indices]
            activity_idx, compound_features, assay_features, activity = batch_data #?Yu: why isn't here `assay_onehot`?

            # move data to device
            # assignment is not necessary for modules but it is for tensors.
            # https://discuss.pytorch.org/t/what-is-the-difference-between-doing-net-cuda-vs-net-to-device/69278/8
            if isinstance(compound_features, torch.Tensor):
                compound_features = compound_features.to(device)
            assay_features = assay_features.to(device) if not isinstance(assay_features[0], str) else assay_features
            activity = activity.to(device)

            # forward
            if 'Multitask' in hparams['model']:
                assay_features_norm = F.normalize(assay_features, p=2, dim=1) #
                sim_to_train = assay_features_norm @ train_assay_features_norm.T
                sim_to_train_weights = F.softmax(sim_to_train * hparams['multitask_temperature '], dim=1) #?Yu: what is `sim_to_train_weights`?
                preactivations = model(compound_features, sim_to_train_weights)
            else:
                preactivations = model(compound_features, assay_features) #?Yu: why not `assay_onehot`?
            
            # loss
            #?Yu: why the below code block is commented out in the primary code.
            #if hparams.get('loss_fun') in ('CE', 'Con'):
            #    loss = F.binary_cross_entropy_with_logits(preactivations, activity)
            #else:
            loss = criterion(preactivations, activity)

            # accumulate loss
            loss_sum += loss.item()

            # accumulate preactivations
            # - need to detach; preactivations.requires_grad is True
            # - move it to cpu
            preactivations_l.append(preactivations)

            # accumulate indices just to double check.
            # - activity_idx is a np.array, not a torch.tensor
            activity_idx_l.append(activity_idx) #?Yu: why not this code line closely after the definition of `activity_idx`.

            if batch_num % EVERY == 0 and verbose:
                logger.info(f'Epoch{epoch}: Test batch {batch_num} out of {len(test_batcher) -1}.')
        
        # log mean loss over all minibatches
        mlflow.log_metric('test_loss', loss_sum / len(test_batcher), step=epoch)
        if wandb.run: wandb.log({'test/loss': loss_sum / len(test_batcher)})

        # compute test auroc and avgp for each assay (on the cpu) 'WHY??? #?Yu: 'WHY' is in the primary code.
        preactivations = torch.cat(preactivations_l, dim=0)
        print(f'preactivations.shape: {preactivations.shape}') 
        probabilities = torch.sigmoid(preactivations) #?Yu: figure out `sigmoid` function, why use it here.

        activity_idx = np.concatenate(activity_idx_l, axis=0)
        print(f'The length of activity_idx for testing: {len(activity_idx)}, including {activity_idx}')
        # assert np.array_equal(activity_idx, test_idx) #Todo WHY??? #?Yu: 'WHY' is in the primary code.

        probabilities = probabilities.detach().cpu().numpy().astype(np.float32)

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
        print(f'Scores in test: scores.shape: {scores.shape},\n scores: {scores}')

        #md  = metrics.swipe_threshold_sparse(targets=targets, scores=scores, verbose=verbose>=2, ret_dict=True)
        md  = swipe_threshold_sparse(targets=targets, scores=scores, bedroc_alpha=bedroc_alpha, verbose=verbose>=2, ret_dict=True)

        if hparams.get('loss_fun') == 'Con':
            for ii, k in enumerate(ks): # what is `ii`
                md[f'top_{k}_acc'] = {0:np.vstack(topk_l)[:-1, ii].mean()} # drop last (might be not full)
            md['arocc'] = {0:np.hstack(arocc_l)[:-1].mean()} # drop last (might be not full)

        # log metrics mean over assays

        logdic = {f'test_mean_{k}': np.nanmean(list(v.values())) for k,v in md.items() if v} #?Yu: why `:` is not in f''
        mlflow.log_metrics(logdic, step=epoch)

        if wandb.run: wandb.log({k.replace('_','/'):v for k, v in logdic.items()}, step=epoch)
        if verbose:
            logger.info(pd.DataFrame.from_dict([logdic]).T) #?Yu: print a dataframe?
        
        # compute test activity counts and positives
        counts, positives = {}, {}
        for idx, col in enumerate(targets.T):
            if col.nnz == 0:
                continue
            counts[idx] = col.nnz
            positives[idx] = col.sum()

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
    
    if not keep: #?Yu: what is keep
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