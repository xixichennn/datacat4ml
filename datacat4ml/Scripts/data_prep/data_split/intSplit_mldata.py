"""
Split datasets internally into train-test folds using different splitting strategies:
- random split
- cluster-aware split (cluster k-fold and cluster repeated holdout)
"""

import os
from typing import List
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd

# for `def random_split`
import random
from sklearn.model_selection import StratifiedKFold
# for `def cluster_kfold_split` and `def cluster_repeated_holdout_split`
from rdkit.ML.Cluster import Butina 
# for molecular distance calculations
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from Levenshtein import distance as levenshtein

from datacat4ml.const import RANDOM_SEED
from datacat4ml.const import CURA_LHD_OR_DIR, CURA_MHD_OR_DIR, CURA_MHD_effect_OR_DIR, CURA_HHD_OR_DIR
from datacat4ml.const import SPL_DATA_DIR, SPL_LHD_OR_DIR, SPL_MHD_OR_DIR, SPL_MHD_effect_OR_DIR, SPL_HHD_OR_DIR

#===============================================================================
# Molecular distance 
#===============================================================================
"""Adopted from https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/benchmark/cliffs.py"""
#  Substructure distance or similarity based on morgan fingerprint
def get_substructure_mat(smiles: List[str], radius: int = 2, nBits: int = 1024, distance: bool=True):

    """ 
    Calculates a matrix of Tanimoto distance or similarity scores for the whole molecules of a list of SMILES string.
    
    This method capture the “global” differences or similarity between molecules by considering the entire set of substructures they contain
    """

    if distance:
        returnDistance = 1 # return distance matrix. Distance = 1 - Similarity.
    else:
        returnDistance = 0 # return similarity matrix

    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        fps.append(fp)

    # generate the distance/similarity matrix based on the fingerprints:
    mat = np.zeros((len(fps), len(fps)), float)
    for i, fp in enumerate(fps):
        if i == len(fps) - 1:
            break
        ds = np.array(
            DataStructs.BulkTanimotoSimilarity(fp,
                                               fps[i + 1:],
                                               returnDistance=returnDistance))
        mat[i, i + 1:] = ds
        mat[i + 1:, i] = ds

    return mat

# scaffold distance/similarity based on morgan fingerprint
def get_scaffold_mat(smiles: List[str], radius: int = 2, nBits: int = 1024, distance: bool=True):
    """ Calculates a matrix of Tanimoto distance/similarity scores for the scaffolds of a list of SMILES string """
    
    if distance:
        returnDistance = 1 # return distance matrix. Distance = 1 - Similarity.
    else:
        returnDistance = 0 # return similarity matrix
    
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

    mat = np.zeros((len(smiles), len(smiles)), float)
    for i, scaf_fp in enumerate(scaf_fps):
        if i == len(scaf_fps) - 1:
            break
        ds = np.array(
            DataStructs.BulkTanimotoSimilarity(scaf_fp,
                                               scaf_fps[i + 1:],
                                               returnDistance=returnDistance))
        mat[i, i + 1:] = ds
        mat[i + 1:, i] = ds

    return mat

# levenstein distance/similarity based on SMILES strings
def get_levenshtein_mat(smiles: List[str], normalize: bool = True, distance: bool=True):
    """ Calculates a matrix of levenshtein distance/similarity scores for a list of SMILES string
    Levenshtein distance/similarity, i.e edit distance/similarity, measures the number of single character edits (insertions, deletions or substitutions) required to change one string into the other.
    As SMILES is a text-based representation of a molecule, this similarity metric can be used to measure the similarity between two molecules.
    
    """
    smi_len = len(smiles)

    mat = np.zeros([smi_len, smi_len])
    # calcultate the upper triangle of the matrix
    for i in tqdm(range(smi_len)):
        for j in range(i, smi_len):
            # normalized to [0,1] or not
            if normalize: 
                dist = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
                sim = 1 - dist
            else:
                dist = levenshtein(smiles[i], smiles[j])
                sim = 1.0 / (1 + dist)
            
            # return distance or similarity matrix
            if distance:
                mat[i, j] = dist
            else:
                mat[i, j] = sim
    # fill in the lower triangle without having to loop (saves ~50% of time)
    mat = mat + mat.T - np.diag(np.diag(mat))
    # get from a distance matrix to a similarity matrix
    mat = 1 - mat

    # fill the diagonal with 0's
    np.fill_diagonal(mat, 0)

    return mat

def molecule_similarity(smiles: List[str], similarity: float = 0.9,):
    """ Calculate which pairs of molecules have a high substructure, scaffold, or SMILES similarity """
    m_subs = get_substructure_mat(smiles) <= (1 - similarity)
    m_scaff = get_scaffold_mat(smiles) <= (1 - similarity)
    m_leve = get_levenshtein_mat(smiles) <= (1 - similarity)

    return (m_subs + m_scaff + m_leve).astype(int)

def find_stereochemical_siblings(smiles: List[str]):
    """
    Detects molecules that have different SMILES strings, but encode for the same molecule with different stereochemistry. 
    For racemic mixtures it is often unclear which one is measured/active

    returns:
        pair_smis: List of pairs of SMILES strings that are stereochemical siblings
        pair_idx: List of pairs of indices corresponding to the SMILES strings in the input list
    """
    smat_lower = np.tril(get_substructure_mat(smiles, radius=4, nBits=4096, distance=False), k=0)
    identical = np.where(smat_lower == 1) # identical[0] is the row indices, identical[1] is the column indices

    pair_idx = []
    pair_smis = []
    for i, j in zip(identical[0], identical[1]):
        pair_idx.append([i, j])
        pair_smis.append([smiles[i], smiles[j]])
    #print(f'pair_idx is \n{pair_idx}')
    #print(f'len(pair_smis): {len(pair_smis)}')

    return pair_smis, pair_idx

#===============================================================================
# Data splitting methods
#===============================================================================
# random split
def random_split(x, y, n_folds=5, random_seed=RANDOM_SEED):
    """
    randomly split the dataset into training and testing sets for n_folds times, stratified on y

    params
    ------
    - x: list or np.array
        input features
    - y: list or np.array
        target values # StratifiedKFold will stratifiy based on y
    - n_folds: int
        number of folds for cross-validation. The test_size will be 1/n_folds automatically.

    returns
    -----------
    - test_splits: List[np.ndarray]
        A list of lists, each containing the indices of the test set for each fold.
    """
    # adjust folds if class imbalance prevents stratification
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = min(counts)
    print(f'min_class_count: {min_class_count}')

    if min_class_count < n_folds:
        print(f'resetting n_folds {n_folds} → {min_class_count} due to class imbalance.')
        n_folds = min_class_count
    if n_folds < 2:
        print(f"skipped — k-fold CV not applicable due to n_folds < 2).")
        return None, None

    else:
        skf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
        train_folds, test_folds = [], []
        for train_idx, test_idx in skf.split(x, y):
            train_folds.append((train_idx.tolist()))
            test_folds.append((test_idx.tolist()))

        return train_folds, test_folds

# cluster-aware split
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

def cluster_repeated_holdout_split(x, dist_type='substruct', clusterSizeThreshold=5, threshold=0.65, combineRandom=False, 
                        random_seed=RANDOM_SEED, test_size=0.2,n_splits=5, selectionStrategy='clust_holdout'):
    """
    Assigns data points to training and testing sets based on selection strategies using clustering.

    Params
    ------
    - x: list or np.array
        input features (e.g., list of SMILES strings)
    - dist_type: str, the type of distance metric to use. Options are 'substruct', 'scaf', and 'levenshtein'.
    - clusterSizeThreshold: int, the minimum size for a cluster to be considered "large". Clusters smaller than this size will be handled according to the `combineRandom` parameter.
    - threshold: float, the distance threshold for clustering. Molecules with distances below this threshold will be grouped into the same cluster.
    - combineRandom: bool, if True, small clusters will be combined randomly to form larger clusters. If False, points from small clusters will be added to the nearest larger cluster based on the distance matrix.
    
    - randomSeed: int, seed for random number generator to ensure reproducibility.
    - test_size: float, the proportion of the dataset to include in the test split.
    - n_splits: int, the number of different train-test splits to generate.
        Here, n_splits is different from n_folds in `random_split`. 
        - n_splits is the number of repeated samplings of train-test splits, each split is independent from each other.
        - n_folds is the number of folds in cross-validation, each fold won't have identical data points.
    - selectionStrategy: SelectionStrategy, the strategy to use for selecting test samples.
        - 'cluster_stratified' ensures each fold has different data points for all kinds of datasets (hhd, mhd, lhd, small or large)
        - 'cluster_holdout' only ensure each fold for mhd and hhd has different data points, but for lhd or very small datasets, some folds may have identical data points
    
    Returns
    -------
    - test_folds: List[np.ndarray]
        A list of lists, each containing the indices of the test set for each sampling.
    - train_folds: List[np.ndarray]
        A list of lists, each containing the indices of the train set for each sampling.

    """

    # get distance matrix
    if dist_type == 'substruct':
        dmat = get_substructure_mat(x)
    elif dist_type == 'scaf':
        dmat = get_scaffold_mat(x)
    elif dist_type == 'levenshtein':
        dmat = get_levenshtein_mat(x)

    # cluster the data
    clusterSizeThreshold=max(5, len(x)/50) # set a minimum cluster size based on the dataset size
    largeClusters = clusterData(dmat, threshold, clusterSizeThreshold, combineRandom)

    # assign data into train and test sets
    nTest= round(len(dmat)*test_size)

    test_folds = [] # list of lists, each containing the indices of the test samples for each split
    train_folds = []

    for n in range(n_splits): 

        random.seed(random_seed + n) # set different random seed for each split to get distinct splits.

        # ensure distributional overlap between train and test splits -> easier task
        if selectionStrategy == 'clust_stratified': 
            ordered = []
            for c in largeClusters:
                random.shuffle(c) # shuffle the cluster
                ordered.extend((i / len(c), x) for i, x in enumerate(c))
            ordered = [y for x, y in sorted(ordered)]
            test=ordered[:nTest]
            train=ordered[nTest:]

        # ensure cluster disjointness - train and test cover different regions of chemical space -> harder task
        elif selectionStrategy == 'clust_holdout': 
            random.shuffle(largeClusters)
            test = []
            train = []
            for c in largeClusters:
                if len(test) + len(c) <= nTest:
                    test.extend(c) # add entire cluster to test set
                #if len(test) < nTest:
                #    nRequired = nTest - len(test)
                #    test.extend(c[:nRequired]) # this way may sometimes split a cluster across train and test sets, which defeats the purpose of 'cluster disjointness'.
                #    train.extend(c[nRequired:])
                else: 
                    train.extend(c) # all remaining clusters go to train set
                    
        test_folds.append(test)
        train_folds.append(train)

    return train_folds,test_folds

def cluster_kfold_split(x, dist_type='substruct', clusterSizeThreshold=5, threshold=0.65, combineRandom=False,
                        random_seed=RANDOM_SEED, n_folds=5, selectionStrategy='clust_holdout'):
    """
    Assigns data points to training and testing sets for k-fold cross-validation based on selection strategies using clustering.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)

    # get distance matrix
    if dist_type == 'substruct':
        dmat = get_substructure_mat(x)
    elif dist_type == 'scaf':
        dmat = get_scaffold_mat(x)
    elif dist_type == 'levenshtein':
        dmat = get_levenshtein_mat(x)

    # cluster the data
    clusterSizeThreshold=max(5, len(x)/50) # set a minimum cluster size based on the dataset size
    largeClusters = clusterData(dmat, threshold, clusterSizeThreshold, combineRandom)
    n_mols = len(x)

    test_folds = []
    train_folds = []

    if selectionStrategy == 'clust_stratified':
        
        fold_assignments = [[] for _ in range(n_folds)] # Each cluster contributes some members to each fold
        for cluster in largeClusters:
            random.shuffle(cluster)
            # Split cluster into roughly equal parts across folds
            splits = np.array_split(cluster, n_folds)
            for i in range(n_folds):
                fold_assignments[i].extend(splits[i])

        # Construct train/test splits
        for i in range(n_folds):
            test_idx = fold_assignments[i]
            train_idx = [idx for j, f in enumerate(fold_assignments) if j != i for idx in f] # when j == i, the idx belongs to test set
            test_folds.append(np.array(test_idx))
            train_folds.append(np.array(train_idx))

    elif selectionStrategy == 'clust_holdout':
        n_clusters = len(largeClusters)
        if n_clusters < n_folds:
            # Adjust n_folds down (Safest)
            print(f"Warning: Reducing n_folds from {n_folds} to {n_clusters} due to low cluster count.")
            n_folds = n_clusters

        if n_folds < 2:
            print(f'n_folds < 2 after adjustment, cannot perform cluster k-fold split.')
        else:
            try:
                # Shuffle clusters (use local Random for reproducibility without changing global RNG)
                rng = random.Random(random_seed)
                rng.shuffle(largeClusters)

                # Greedy assignment: assign each cluster to the fold with the smallest current size
                fold_assignments = [[] for _ in range(n_folds)]
                fold_sizes = [0] * n_folds  # number of molecules in each fold

                for cluster in largeClusters:
                    # choose fold with minimum size (tie-breaker: lowest index)
                    smallestFold_idx = int(np.argmin(fold_sizes)) # get the indices of the fold with the smallest size
                    fold_assignments[smallestFold_idx].extend(cluster)
                    fold_sizes[smallestFold_idx] += len(cluster)

                # Construct train/test splits (ensure integer dtype)
                for i in range(n_folds):
                    test_idx = np.array(fold_assignments[i], dtype=int)
                    train_idx = np.array([idx for j, f in enumerate(fold_assignments) if j != i for idx in f], dtype=int)
                    test_folds.append(test_idx)
                    train_folds.append(train_idx)
            except Exception as e:
                print(f"Cluster k-fold split failed due to: {e}")
    return train_folds, test_folds

#===============================================================================
# Internal splitting
#===============================================================================
Cura_Spl_Dic = {CURA_HHD_OR_DIR: SPL_HHD_OR_DIR,
                 CURA_MHD_OR_DIR: SPL_MHD_OR_DIR,
                 CURA_LHD_OR_DIR: SPL_LHD_OR_DIR,
                 CURA_MHD_effect_OR_DIR: SPL_MHD_effect_OR_DIR
                 }

def add_fold_columns(df, prefix, train_folds, test_folds):
    """
    Add 'train/test' columns to df for each fold using prefix.
    """
    df = df.copy()
    for i, (train_idx, test_idx) in enumerate(zip(train_folds, test_folds)):
        fold_col = f"{prefix}_fold{i}"
        df[fold_col] = np.nan # initialize the column with NaN

        # mark train/test according to the fold indices
        df.loc[train_idx, fold_col] = 'train'
        df.loc[test_idx, fold_col] = 'test'
    return df

def random_splitter(df, n_folds, aim):
    """
    Apply stratified random split to the dataset for two stereo modes:
    - rmvS0: full dataset
    - rmvS1: dataset without stereochemical siblings

    params
    ------
    - df: pd.DataFrame.
    - n_folds: int, number of folds for cross-validation. The test_size will be 1/n_folds automatically.
    - aim: str, the aim of the model build for. Options are 'lo' and 'vs'. It is used to name the split columns.

    returns
    -----------
    - df: pd.DataFrame
        a new df with additional columns for the random splits
    """
    df_result = df.copy()
    activity_col = 'lo_activity' if aim == 'lo' else 'vs_activity'

    # Internal helper
    def _safe_split(sub_df, prefix, n_folds, random_seed=RANDOM_SEED):
        """Perform safe stratified split and return updated df or None."""
        if len(sub_df) == 0:
            print(f"{prefix}: skipped — no data available.")
            return None
        
        x = sub_df['canonical_smiles_by_Std'].tolist()
        y = sub_df[activity_col].tolist()

        try:
            train_folds, test_folds = random_split(x, y, n_folds, RANDOM_SEED)
            if train_folds is None or test_folds is None:
                return None
            else:
                df_split = add_fold_columns(sub_df, prefix, train_folds, test_folds)
                return df_split
        except ValueError as e:
            print(f"{prefix}: skipped due to {e}")
            return None

    # --- Perform both splits ---
    df_rmvS0 = _safe_split(df, f"rmvS0_rs-{aim}", n_folds)
    df_rmvS1_sub = _safe_split(df[df['stereoSiblings'] == False].reset_index(drop=True),
                                    f"rmvS1_rs-{aim}", n_folds)

    # --- Merge results ---
    if df_rmvS0 is not None:
        df_result = df_rmvS0.copy()

    if df_rmvS1_sub is not None:
        merge_cols = ['activity_id'] + [col for col in df_rmvS1_sub.columns if col.startswith(f"rmvS1_rs-{aim}")]
        df_result = df_result.merge(df_rmvS1_sub[merge_cols], on='activity_id', how='left')

    print("Random splitting completed.")

    return df_result

def cluster_kfold_splitter(df, selectionStrategy):
    """
    Apply cluster-aware split and add new columns for train/test folds.

    params
    ------
    - df: pd.DataFrame.
    - selectionStrategy: SelectionStrategy, the strategy to use for selecting test samples. Options are 'clust_stratified' and 'clust_holdout'.

    returns
    -----------
    - df: pd.DataFrame
        a new df with additional columns for the cluster-aware splits
    """

    sS = 'cs' if selectionStrategy == 'clust_stratified' else 'ch'
    df_result = df.copy()

    def _safe_cluster_split(sub_df, prefix, selectionStrategy, random_seed=RANDOM_SEED):
        """Perform a safe cluster-aware split and return df with new columns or None."""
        if len(sub_df) == 0:
            print(f"{prefix}: skipped — empty subset.")
            return None

        else:
            x = sub_df['canonical_smiles_by_Std'].tolist()

            try:
                train_folds, test_folds = cluster_kfold_split(
                    x=x,
                    selectionStrategy=selectionStrategy,
                )

                # Check and deduplicate folds if necessary
                tupled_test_folds = [tuple(sorted(fold)) for fold in test_folds]
                if len(tupled_test_folds) != len(set(tupled_test_folds)):
                    print(f"{prefix}: duplicate test folds detected. Keeping unique ones.")
                    test_folds = list(set(tupled_test_folds))

                # Add fold columns
                return add_fold_columns(sub_df, prefix, train_folds, test_folds)

            except ValueError as e:
                print(f"{prefix}: cluster-aware split skipped due to {e}")
                return None

    # --- Run for both rmvS0 and rmvS1 ---
    df_rmvS0 = _safe_cluster_split(df, f"rmvS0_{sS}", selectionStrategy)
    df_rmvS1_sub = _safe_cluster_split(
        df[df['stereoSiblings'] == False].reset_index(drop=True),
        f"rmvS1_{sS}",
        selectionStrategy
    )

    # --- Merge results ---
    if df_rmvS0 is not None:
        df_result = df_rmvS0.copy()

    if df_rmvS1_sub is not None:
        merge_cols = ['activity_id'] + [col for col in df_rmvS1_sub.columns if col.startswith(f"rmvS1_{sS}")]
        df_result = df_result.merge(df_rmvS1_sub[merge_cols], on='activity_id', how='left')

    print(f"Cluster-aware split completed for strategy '{selectionStrategy}'.")

    return df_result

def internal_split(in_dir: str = CURA_HHD_OR_DIR, rmvD: int = 1):
    """
    Split data into train-test folds for file(s) in the input directory.

    params
    ------
    - in_dir: str, input directory containing files to be split.
    """
    # input directory
    print(f"in_dir is: {in_dir}\n")
    in_file_dir = os.path.join(in_dir, f'rmvD{str(rmvD)}')
    files = os.listdir(in_file_dir)

    # output directory
    out_dir = os.path.join(Cura_Spl_Dic[in_dir], f'rmvD{str(rmvD)}')
    print(f"out_dir is: {out_dir}\n")
    os.makedirs(out_dir, exist_ok=True)

    # split data
    for f in files:

        print (f"\ninput_file is: {f}\n")
        df = pd.read_csv(os.path.join(in_file_dir, f))
        
        # skip files with less than 40 data points
        #if len(df) < 40:
        #    print(f"Skip {f}, because it has less than 40 data points, not enough for building ML models")
        #else:

        # apply split
        print("random split...")
        df = random_splitter(df, n_folds=5, aim='lo')
        df = random_splitter(df, n_folds=5, aim='vs')

        print("cluster-aware split...")
        df = cluster_kfold_splitter(df, selectionStrategy='clust_stratified')
        df = cluster_kfold_splitter(df, selectionStrategy='clust_holdout')

        # save the new df
        out_file = os.path.join(out_dir, f[:-12] + f"_split.csv")
        df.to_csv(out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into test/train internally for ML model training and evaluation")
    parser.add_argument('--in_dir', type=str, required=True, help= 'Input directory containing files to be split')
    parser.add_argument('--rmvD', type=int, required=True, help='Remove duplicate molecules')

    args = parser.parse_args()

    internal_split(in_dir=args.in_dir, rmvD=args.rmvD)
