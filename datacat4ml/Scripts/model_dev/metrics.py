"""
The functions containing the activity cliffs are adapted from MoleculeACE
 https://github.com/molML/MoleculeACE.git
"""

from typing import List, Union
import sys
import numpy as np

# classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef
# regression metrics
from sklearn.metrics import r2_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC


# inner modules
from datacat4ml.Scripts.data_prep.data_split.split_utils.cliff import ActivityCliffs

#=================== For classification ===================
# calculate the accuracy
def calc_accuracy(true, pred):
    """ Calculates the accuracy

    Args:
        true: (1d array-like shape) true test values (int)
        pred: (1d array-like shape) predicted test values (int)

    Returns: (float) accuracy
    """
    # Convert to 1-D numpy array if it's not
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true

    return accuracy_score(true, pred)

# calculate the precision
def calc_precision(true, pred):
    """ Calculates the precision

    Args:
        true: (1d array-like shape) true test values (int)
        pred: (1d array-like shape) predicted test values (int)

    Returns: (float) precision
    """
    # Convert to 1-D numpy array if it's not
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true

    return precision_score(true, pred)

# calculate the recall
def calc_recall(true, pred):
    """ Calculates the recall

    Args:
        true: (1d array-like shape) true test values (int)
        pred: (1d array-like shape) predicted test values (int)

    Returns: (float) recall
    """
    # Convert to 1-D numpy array if it's not
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true

    return recall_score(true, pred)

# calculate the roc_auc score
def calc_roc_auc(true, pred):
    """ Calculates the roc_auc score

    Args:
        true: (1d array-like shape) true test values (int)
        pred: (1d array-like shape) predicted test values (int)

    Returns: (float) roc_auc score
    """
    # Convert to 1-D numpy array if it's not
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true

    return roc_auc_score(true, pred)

# calculate the f1 score
def calc_f1(true, pred):
    """ Calculates the f1 score

    Args:
        true: (1d array-like shape) true test values (int)
        pred: (1d array-like shape) predicted test values (int)

    Returns: (float) f1 score
    """
    # Convert to 1-D numpy array if it's not
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true

    return f1_score(true, pred)


# calculate the matthews correlation coefficient
def calc_mcc(true, pred):
    """ Calculates the matthews correlation coefficient"""

    # Convert to 1-D numpy array if it's not
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true

    return matthews_corrcoef(true, pred)

# calculate the bedroc score
def calc_bedroc(y_pred_proba, y_true, alpha: float = 80.5):
    """ Calculates the bedroc score unsing rdkit.ML.Scoring.CalcBEDROC
    
    :param y_pred_proba: (lst/array) predicted probabilities, i.e. the value of model.predict_proba(x_test)
    :param y_true: (lst/array) true values
    :param alpha: (float)  early recognition parameter

    :return: (float) BEDROC score
    """

    score = list(zip(y_pred_proba[:, 1], y_true))
    score.sort(key=lambda x: x[0], reverse=True)
    bedroc_score = CalcBEDROC(score, 1, alpha)

    return bedroc_score

def bedroc_score(y_true, y_pred, decreasing=True, alpha=20.0):
    
    """
    Adopted from CLAMP repository
    https://github.com/ml-jku/clamp.git
    
    BEDROC metric implemented according to Truchon and Bayley.
    The Boltzmann Enhanced Descrimination of the Receiver Operator
    Characteristic (BEDROC) score is a modification of the Receiver Operator
    Characteristic (ROC) score that allows for a factor of *early recognition*.
    References:
        The original paper by Truchon et al. is located at `10.1021/ci600426e
        <http://dx.doi.org/10.1021/ci600426e>`_.
    Args:
        y_true (array_like):
            Binary class labels. 1 for positive class, 0 otherwise.
        y_pred (array_like):
            Prediction values.
        decreasing (bool):
            True if high values of ``y_pred`` correlates to positive class.
        alpha (float):
            Early recognition parameter.
    Returns:
        float:
            Value in interval [0, 1] indicating degree to which the predictive
            technique employed detects (early) the positive class.
     """

    assert len(y_true) == len(y_pred), \
        'The number of scores must be equal to the number of labels'

    big_n = len(y_true)
    n = sum(y_true == 1)

    if decreasing:
        order = np.argsort(-y_pred)
    else:
        order = np.argsort(y_pred)

    m_rank = (y_true[order] == 1).nonzero()[0] + 1

    s = np.sum(np.exp(-alpha * m_rank / big_n))

    r_a = n / big_n

    rand_sum = r_a * (1 - np.exp(-alpha))/(np.exp(alpha/big_n) - 1)

    fac = r_a * np.sinh(alpha / 2) / (np.cosh(alpha / 2) -
                                      np.cosh(alpha/2 - alpha * r_a))

    cte = 1 / (1 - np.exp(alpha * (1 - r_a)))

    return s * fac / rand_sum + cte

#====================== For regression ======================
def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not    
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true


    return np.sqrt(np.mean(np.square(true - pred)))

def calc_r2(true, pred):
    """ Calculates the R2 score

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) r2 score
    """
    # Convert to 1-D numpy array if it's not    
    pred = np.array(pred) if type(pred) is not np.array else pred
    true = np.array(true) if type(true) is not np.array else true

    return r2_score(true, pred)

# calc rmse for activity cliff molecules
def calc_cliff_rmse(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculates the RMSE of the activity cliff molecules

    :param y_test_pred: (lst/array)predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES for test set
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES for train set
    :param kwargs: additional arguments for ActivityCliffs() 
    :return: (float) rmse on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError("if cliff_mols_test is None, then smiles_test, y_train, and smiles_train must be provided to compute activity cliffs")

    # Convert to numpy arrays if they are not none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # calculate the activity cliffs and 
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_mols=False, **kwargs)
        # take only the test cliffs
        cliff_mols_test = cliff_mols[len(y_train):] # assign the sublist of 'cliff_mols' that corresponds to the activity cliff compounds in the test set

    # Get the index of the activity cliff compounds
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]
    
    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)

# calc r2 for activity cliff molecules
def calc_cliff_r2(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
                    y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None, **kwargs):
    """ Calculates the R2 of the activity cliff molecules

    :param y_test_pred: (lst/array)predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES for test set
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES for train set
    :param kwargs: additional arguments for ActivityCliffs() 
    :return: (float) r2 on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError("if cliff_mols_test is None, then smiles_test, y_train, and smiles_train must be provided to compute activity cliffs")

    # Convert to numpy arrays if they are not none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # calculate the activity cliffs and 
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_mols=False, **kwargs)
        # take only the test cliffs
        cliff_mols_test = cliff_mols[len(y_train):] # assign the sublist of 'cliff_mols' that corresponds to the activity cliff compounds in the test set

    # Get the index of the activity cliff compounds
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]
    
    return calc_r2(y_pred_cliff_mols, y_test_cliff_mols)