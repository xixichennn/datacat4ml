import os
from typing import List
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd

# for `def random_split`
import random
from sklearn.model_selection import StratifiedKFold
# for `def cluster_aware_split`
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
    skf = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
    test_folds = []
    for train_idx, test_idx in skf.split(x, y):
        test_folds.append((test_idx.tolist()))

    return test_folds

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

def assignUsingClusters(dmat, threshold, clusterSizeThreshold=5, combineRandom=False, 
                        random_seed=RANDOM_SEED, test_size=0.2,n_samples=5, selectionStrategy='clust_holdout', ):
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
    - n_samples: int, the number of different train-test splits to generate.
        Here, n_samples is different from n_folds in `random_split`. 
        - n_samples is the number of repeated samplings of train-test splits, each split is independent from each other.
        - n_folds is the number of folds in cross-validation, each fold won't have identical data points.
    - selectionStrategy: SelectionStrategy, the strategy to use for selecting test samples. Options are 'clust_stratified' and 'clust_holdout'.
    
    Returns
    -------
    - assignments: List[List[int]], a list containing n_samples lists, each representing the indices of the test samples.
    """

    largeClusters = clusterData(dmat, threshold, clusterSizeThreshold, combineRandom)
    
    random.seed(random_seed) # set the random seed for reproducibility
    nTest= round(len(dmat)*test_size)
    assignments = [] # list of lists, each containing the indices of the test samples for each split
    for i in range(n_samples): 
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
                        random_seed=RANDOM_SEED, test_size=0.2, n_samples=5):
    """
    Splits the dataset into training and testing sets using a clustering-aware strategy based on molecular distance.

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
        dmat = get_substructure_mat(x)
    elif dist_type == 'scaf':
        dmat = get_scaffold_mat(x)
    elif dist_type == 'levenshtein':
        dmat = get_levenshtein_mat(x)

    clusterSizeThreshold=max(5, len(x)/50) # set a minimum cluster size based on the dataset size

    test_folds = assignUsingClusters(dmat, threshold, clusterSizeThreshold, combineRandom,
                                     random_seed, test_size, n_samples, selectionStrategy)
    return test_folds

#===============================================================================
# main
#===============================================================================
Cura_Spl_Dic = {CURA_HHD_OR_DIR: SPL_HHD_OR_DIR,
                 CURA_MHD_OR_DIR: SPL_MHD_OR_DIR,
                 CURA_LHD_OR_DIR: SPL_LHD_OR_DIR,
                 CURA_MHD_effect_OR_DIR: SPL_MHD_effect_OR_DIR
                 }

def random_splitter(df, x, y, rmv_stereo, n_folds, aim):
    """
    Apply stratified random split. Add new columns about the split info.

    params
    ------
    - x, should be df['canonical_smiles_by_Std'].tolist()
    - y, should be df['lo_activity'].tolist() or df['vs_activity'].tolist()
    - n_folds: int, number of folds for cross-validation. The test_size will be 1/n_folds automatically.
    - aim: str, the aim of the model build for. Options are 'lo' and 'vs'. It is used to name the split columns.
    - rmv_stereo: int, whether stereochemical siblings have been removed. It is used to name the split columns.

    returns
    -----------
    - df: pd.DataFrame
        a new df with additional columns for the random splits
    """
    try:
    
        # check the minimum number of samples for each class in y
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = min(counts)

        # reduce n_folds to ensure stratification
        if min_class_count < n_folds: 
            print(f'Reset the n_folds from {n_folds} to {min_class_count} to ensure stratification')
            n_folds = min_class_count
        
        if n_folds >=2:

            # get random splits
            test_folds = random_split(x, y, n_folds=n_folds, random_seed=RANDOM_SEED)

            # assign split to df
            rmv_stereo = int(rmv_stereo)
            for i, fold in enumerate(test_folds):
                col = f'rmvStereo{rmv_stereo}_rs_{aim}_fold{i}'
                df[col] = ['test' if idx in fold else 'train' for idx in range(len(x))]
        else:
            print(f'Random split skipped, because k-fold CV is not applicable for this dataset')

    except ValueError as e:
        print(f'Random split skipped due to: {e}')

    return df

def cluster_aware_splitter(df, x, rmv_stereo, selectionStrategy):
    """
    Apply cluster-aware split. Add new columns about the split info.

    params
    ------
    - x, should be df['canonical_smiles_by_Std'].tolist()
    - selectionStrategy: SelectionStrategy, the strategy to use for selecting test samples. Options are 'clust_stratified' and 'clust_holdout'.
    """
    # get cluster-aware splits
    try:
        test_folds= cluster_aware_split(dist_type= 'substruct', selectionStrategy=selectionStrategy, x=x,
                                    threshold=0.65, combineRandom=False, random_seed=RANDOM_SEED, test_size=0.2, n_samples=5)

        # shorthand for the new column names
        if selectionStrategy == 'clust_stratified':
            sS = 'cs'
        elif selectionStrategy == 'clust_holdout':
            sS = 'ch'

        ## check whether there are identical fold in 'test_folds'
        tupled_test_folds = [tuple(sorted(fold)) for fold in test_folds]
        has_duplicates = len(tupled_test_folds) != len(set(tupled_test_folds))
        print(f'has_duplicates in {sS}_test_folds: {has_duplicates}')

        if has_duplicates:
            # get the unique folds
            test_folds = list(set(tupled_test_folds))
            print(f'Unique folds retained: {len(test_folds)}')

        # assign split to df
        rmv_stereo = int(rmv_stereo)
        for i, fold in enumerate(test_folds):
            col = f'rmvStereo{rmv_stereo}_{sS}_fold{i}'
            df[col] = ['test' if idx in fold else 'train' for idx in range(len(x))]

    except ValueError as e:
        print(f'Cluster-aware split skipped due to: {e}')
    
    return df

def split_data(in_dir: str = CURA_HHD_OR_DIR, rmv_stereo: int = 1, rmv_dupMol: int = 1):
    """
    Split data into train-test folds for file(s) in the input directory.
    Each `rmv_stereo` setting writes its own file:
        *_split_rmv0.csv
        *_split_rmv1.csv

    params
    ------
    - in_dir: str, input directory containing files to be split.
    """
    # input directory
    print(f"in_path is: {in_dir}\n")
    in_file_dir = os.path.join(in_dir, 'rmvDupMol' + str(rmv_dupMol))
    files = os.listdir(in_file_dir)

    # output directory
    out_dir = os.path.join(Cura_Spl_Dic[in_dir], 'rmvDupMol' + str(rmv_dupMol))
    print(f"out_dir is: {out_dir}\n")
    os.makedirs(out_dir, exist_ok=True)

    # split data
    for f in files:

        print (f"\ninput_file is: {f}\n")
        df = pd.read_csv(os.path.join(in_file_dir, f))
        
        # skip files with less than 40 data points
        if len(df) < 40:
            print(f"Skip {f}, because it has less than 40 data points, not enough for building ML models")
        else:

            # remove stereochemical siblings
            if rmv_stereo == 1:
                print("remove stereochemical siblings...")
                df = df[df['stereoSiblings'] == False].reset_index(drop=True)

            smiles = df['canonical_smiles_by_Std'].tolist() # x
            lo_activity = df['lo_activity'].tolist() # y
            vs_activity = df['vs_activity'].tolist() # y
            activity_id = df['activity_id'].tolist() # identifier for cross-file splitting

            # apply split
            print("random split...")
            df = random_splitter(df, smiles, lo_activity, rmv_stereo, n_folds=5, aim='lo')
            df = random_splitter(df, smiles, vs_activity, rmv_stereo, n_folds=5, aim='vs')

            print("cluster-aware split...")
            df = cluster_aware_splitter(df, smiles, rmv_stereo, selectionStrategy='clust_stratified')
            df = cluster_aware_splitter(df, smiles, rmv_stereo, selectionStrategy='clust_holdout')

            # save the new df
            out_file = os.path.join(out_dir, f[:-12] + f"_split_rmvStereo{rmv_stereo}.csv")
            df.to_csv(out_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data for ML model training and evaluation")
    parser.add_argument('--in_dir', type=str, required=True, help= 'Input directory containing files to be split')
    parser.add_argument('--rmv_stereo', type=int, required=True, help='Remove stereochemical siblings')
    parser.add_argument('--rmv_dupMol', type=int, required=True, help='Remove duplicate molecules')

    args = parser.parse_args()

    split_data(in_dir=args.in_dir, rmv_stereo=args.rmv_stereo, rmv_dupMol=args.rmv_dupMol)

