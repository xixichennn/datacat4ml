import os
from typing import List
from tqdm import tqdm

import numpy as np

# for `def random_split`
import random
from sklearn.model_selection import StratifiedShuffleSplit
# for `def cluster_aware_split`
from rdkit.ML.Cluster import Butina 
# for molecular distance calculations
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from Levenshtein import distance as levenshtein

from datacat4ml.const import RANDOM_SEED

#===============================================================================
# Molecular distance 
#===============================================================================
"""Adopted from https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/benchmark/cliffs.py"""
#  Substructure distance based on morgan fingerprint
def get_substructure_dmat(smiles: List[str], radius: int = 2, nBits: int = 1024):

    """ 
    Calculates a matrix of Tanimoto distance scores for the whole molecules of a list of SMILES string.
    
    This method capture the “global” differences between molecules by considering the entire set of substructures they contain
    """

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        fps.append(fp)
    
    # generate the distance matrix based on the fingerprints:
    dmat = np.zeros((len(fps), len(fps)), float)
    for i, fp in enumerate(fps):
        if i == len(fps) - 1:
            break
        ds = np.array(
            DataStructs.BulkTanimotoSimilarity(fp,
                                               fps[i + 1:],
                                               returnDistance=1)) # Distance = 1 - Similarity. Set returnDistance=1 to get distance, 0 for similarity.
        dmat[i, i + 1:] = ds
        dmat[i + 1:, i] = ds
    
    return dmat
    
# scaffold distance based on morgan fingerprint
def get_scaffold_dmat(smiles: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto distance scores for the scaffolds of a list of SMILES string """
    scaf_fps = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        try:
            skeleton = GraphFramework(mol) # returns the generic scaffold graph, whcih represents the connectivity and topology of the molecule
        except Exception: # In the very rare case that the molecule cannot be processed, then use a normal scaffold
            print(f"Could not create a generic scaffold of {smi}, then used a normal scaffold instead")
            skeleton = GetScaffoldForMol(mol) # returns the Murcko scaffold, which is the result of removing side chains while retaining ring systems
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
        scaf_fps.append(skeleton_fp)

    dmat = np.zeros((len(smiles), len(smiles)), float)
    for i, scaf_fp in enumerate(scaf_fps):
        if i == len(scaf_fps) - 1:
            break
        ds = np.array(
            DataStructs.BulkTanimotoSimilarity(scaf_fp,
                                               scaf_fps[i + 1:],
                                               returnDistance=1)) # Distance = 1 - Similarity. Set returnDistance=1 to get distance, 0 for similarity.
        dmat[i, i + 1:] = ds
        dmat[i + 1:, i] = ds

    return dmat

# levenstein distance based on SMILES strings
def get_levenshtein_dmat(smiles: List[str], normalize: bool = True):
    """ Calculates a matrix of levenshtein similarity scores for a list of SMILES string
    Levenshtein similarity, i.e edit distance similarity, measures the number of single character edits (insertions, deletions or substitutions) required to change one string into the other.
    Ad SMILES is a text-based representation of a molecule, this similarity metric can be used to measure the similarity between two molecules.
    
    """
    smi_len = len(smiles)

    matrix = np.zeros([smi_len, smi_len])
    # calcultate the upper triangle of the matrix
    for i in tqdm(range(smi_len)):
        for j in range(i, smi_len):
            if normalize:
                matrix[i,j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j])) # calculate the distance and normalize it to [0,1]
            else:
                matrix[i,j] = levenshtein(smiles[i], smiles[j]) # calculate the distance without normalization
    # fill in the lower triangle without having to loop (saves ~50% of time)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    # get from a distance matrix to a similarity matrix
    matrix = 1 - matrix

    # fill the diagonal with 0's
    np.fill_diagonal(matrix, 0)

    return matrix

def molecule_similarity(smiles: List[str], similarity: float = 0.9,):
    """ Calculate which pairs of molecules have a high substructure, scaffold, or SMILES similarity """
    m_subs = get_substructure_dmat(smiles) <= (1 - similarity)
    m_scaff = get_scaffold_dmat(smiles) <= (1 - similarity)
    m_leve = get_levenshtein_dmat(smiles) <= (1 - similarity)

    return (m_subs + m_scaff + m_leve).astype(int)

def find_stereochemical_siblings(smiles: List[str]):
    """
    """
    lower = np.tril(get_substructure_dmat(smiles, radius=4, nBits=4096), k=0)
    identical = np.where(lower == 1) # identical[0] is the row indices, identical[1] is the column indices
    identical_pairs = [[smiles[identical[0][i]], smiles[identical[1][i]]] for i, j in enumerate(identical[0])] # a list of 2-element lists, each containing a pair of identical SMILES strings.
                       
    return list(set(sum(identical_pairs, []))) # a list of unique SMILES strings that have at least one stereochemical sibling in the input list `smiles`.

#===============================================================================
# Data splitting methods
#===============================================================================
# random split
def random_split(x, y, n_folds=5, test_size=0.2, random_seed=RANDOM_SEED):
    """
    randomly split the dataset into training and testing sets for n_folds times, stratified on y

    returns
    -----------
    - test_splits: List[np.ndarray]
        A list of lists, each containing the indices of the test set for each fold.
    """
    sss = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_seed)
    test_folds = []
    for train_idx, test_idx in sss.split(x, y):
        test_folds.append((test_idx.tolist()))
    return test_folds

"""Adopted from https://github.com/rinikerlab/molecular_time_series/blob/main/ga_lib_3.py#L1171"""
def clusterData(dmat, threshold, clusterSizeThreshold, combineRandom=False):
    """ 
    Cluster data based on a distance matrix using the Butina algorithm.

    Params
    ------
    - dmat: a distance matrix get from `get_substructure_matrix`, `get_scaffold_matrix`, or `get_levenshtein_matrix`
    - threshold: float, the distance threshold for clustering. E.g. 0.4 means mols with a similarity above 0.6 will be grouped into the same cluster.
    - clusterSizeThreshold: int, the minimum size for a cluster to be considered "large". Clusters smaller than this size will be handled according to the `combineRandom` parameter.
    - combineRandom: bool, if True, small clusters will be combined randomly to form larger clusters. If False, points from small clusters will be added to the nearest larger cluster based on the distance matrix.

    Returns
    -------
    - largeClusters: List of clusters, where each cluster is represented as a list of indices
    """
    nfps = len(dmat)
    symmDmat = []
    for i in range(1, nfps):
        symmDmat.extend(dmat[i, :i]) # convert a square distance matrix to a 1D array representing the upper triangle of the matrix
    cs = Butina.ClusterData(symmDmat, nfps, threshold, isDistData=True, reordering=True)
    cs = sorted(cs, key=lambda x: len(x), reverse=True) # sort clusters by size in descending order

    # start with the large clusters:
    largeClusters = [list(c) for c in cs if len(c) >= clusterSizeThreshold]
    if not largeClusters:
        raise ValueError("no clusters found")
    # now combine the small clusters to make larger ones:
    if combineRandom:
        tmpCluster = []
        for c in cs:
            if len(c) >= clusterSizeThreshold:
                continue
            tmpCluster.extend(c)
            if len(tmpCluster) >= clusterSizeThreshold:
                random.shuffle(tmpCluster)
                largeClusters.append(tmpCluster)
                tmpCluster = []
        if tmpCluster:
            largeClusters.append(tmpCluster)
    else:
        # add points from small clusters to the nearest larger cluster
        #  nearest is defined by the nearest neighbor in that cluster
        for c in cs:
            if len(c) >= clusterSizeThreshold:
                continue
            for idx in c:
                closest = -1
                minD = 1e5
                for cidx, clust in enumerate(largeClusters):
                    for elem in clust:
                        d = dmat[idx, elem]
                        if d < minD:
                            closest = cidx
                            minD = d
                assert closest > -1
                largeClusters[closest].append(idx)
    return largeClusters

def assignUsingClusters(dmat, threshold, clusterSizeThreshold=5, combineRandom=False, 
                        random_seed=RANDOM_SEED, test_size=0.2,n_folds=5, selectionStrategy='clust_holdout', ):
    """
    Assigns data points to training and testing sets based on selection strategies using clustering.

    Params
    ------
    - dmat: a distance matrix get from `get_substructure_matrix`, `get_scaffold_matrix`, or `get_levenshtein_matrix`
    - threshold: float, the distance threshold for clustering. Molecules with distances below this threshold will be grouped into the same cluster.
    - clusterSizeThreshold: int, the minimum size for a cluster to be considered "large". Clusters smaller than this size will be handled according to the `combineRandom` parameter.
    - combineRandom: bool, if True, small clusters will be combined randomly to form larger clusters. If False, points from small clusters will be added to the nearest larger cluster based on the distance matrix.
    
    - randomSeed: int, seed for random number generator to ensure reproducibility.
    - test_size: float, the proportion of the dataset to include in the test split.
    - n_folds: int, the number of different train-test splits to generate.
    - selectionStrategy: SelectionStrategy, the strategy to use for selecting test samples. Options are 'clust_stratified' and 'clust_holdout'.
    
    Returns
    -------
    - assignments: List[List[int]], a list containing n_folds lists, each representing the indices of the test samples.
    """

    largeClusters = clusterData(dmat, threshold, clusterSizeThreshold, combineRandom)
    
    random.seed(random_seed) # set the random seed for reproducibility
    nTest= round(len(dmat)*test_size)
    assignments = [] # list of lists, each containing the indices of the test samples for each split
    for i in range(n_folds): 
        # ensure distributional overlap between train and test splits -> easier task
        if selectionStrategy == 'clust_stratified': 
            ordered = []
            for c in largeClusters:
                random.shuffle(c) # shuffle the 
                ordered.extend((i / len(c), x) for i, x in enumerate(c))
            ordered = [y for x, y in sorted(ordered)]
            test=ordered[:nTest]
        # ensure cluster disjointness - train and test cover different regions of chemical space -> harder task
        elif selectionStrategy == 'clust_holdout': 
            random.shuffle(largeClusters)
            test = []
            for c in largeClusters:
                nRequired = nTest - len(test)
                test.extend(c[:nRequired])
                if len(test) >= nTest:
                    break
        assignments.append(test)

    return assignments

def cluster_aware_split(dist_type, selectionStrategy, x, 
                        threshold=0.65, combineRandom=False, 
                        random_seed=RANDOM_SEED, test_size=0.2, n_folds=5):
    """
    params
    ------
    - dist_type: str, the type of distance metric to use. Options are 'substruct', 'scaf', and 'levenshtein'.
    - selectionStrategy: SelectionStrategy, the strategy to use for selecting test samples. Options are 'clust_stratified' and 'clust_holdout'.
         according to my observation,
          - 'cluster_stratified' ensures each fold has different data points for all kinds of datasets (hhd, mhd, lhd, small or large)
          - 'cluster_holdout' only ensure each fold for mhd and hhd has different data points, but for lhd or very small datasets, some folds may have identical data points


    returns
    -----------
    - test_folds: List[np.ndarray]
        A list of lists, each containing the indices of the test set for each sampling.
    """
    if dist_type == 'substruct':
        dmat = get_substructure_dmat(x)
    elif dist_type == 'scaf':
        dmat = get_scaffold_dmat(x)
    elif dist_type == 'levenshtein':
        dmat = get_levenshtein_dmat(x)
    
    clusterSizeThreshold=max(5, len(x)/50) # set a minimum cluster size based on the dataset size

    test_folds = assignUsingClusters(dmat, threshold, clusterSizeThreshold, combineRandom,
                                     random_seed, test_size, n_folds, selectionStrategy)
    return test_folds