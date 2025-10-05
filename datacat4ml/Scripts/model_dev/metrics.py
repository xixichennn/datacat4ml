# conda env: datacat (python=3.8.2)
import numpy as np
import pandas as pd
from loguru import logger

from sklearn import metrics
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import torch

from sklearn.metrics import r2_score

#====================================For CLIP-alike Model =========================================================

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


#====================================For ML classifiers =========================================================
def calc_auroc(y_true, y_pred_prob):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC) for a binary classification task.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred_prob (array-like): Predicted probabilities for the positive class (1).

    Returns:
        float: The ROC AUC score.
    """
    auroc = metrics.roc_auc_score(y_true, y_pred_prob[:, 1])
    return auroc

def calc_auprc(y_true, y_pred_prob):
    """
    Calculates the Area Under the Precision-Recall Curve (AUPRC) for a binary classification task.

    Args:
        y_true (array-like): True binary labels (0 or 1).
        y_pred_prob (array-like): Predicted probabilities for the positive class (1).

    Returns:
        float: The AUPRC score.
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred_prob[:, 1])
    auprc = metrics.auc(recall, precision)
    return auprc

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

#====================================For ML regressor =========================================================
def calc_rmse(y_true, y_pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    y_pred = np.array(y_pred) if type(y_pred) is not np.array else y_pred
    y_true = np.array(y_true) if type(y_true) is not np.array else y_true

    return np.sqrt(np.mean(np.square(y_true - y_pred)))

def calc_r2(y_true, y_pred):
    """ Calculates the R2 score

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) r2 score
    """
    # Convert to 1-D numpy array if it's not
    y_pred = np.array(y_pred) if type(y_pred) is not np.array else y_pred
    y_true = np.array(y_true) if type(y_true) is not np.array else y_true

    return r2_score(y_true, y_pred)