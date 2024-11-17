"""
The concept and cofe for molecular similarity and acitivity cliff analysis is adapted from MoleculeACE

https://github.com/molML/MoleculeACE.git

"""

from typing import List, Union
from tqdm import tqdm

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from Levenshtein import distance as levenshtein

#===================1. Molecular similarity  ====================
# tanimoto similarity based on morgan fingerprint
def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for the whole molecules of a list of SMILES string"""

    # make a fingerprint database
    db_fp = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        db_fp[smi] = fp

    smi_len = len(smiles)
    matrix = np.zeros([smi_len, smi_len])
    # calcultate the upper triangle of the matrix
    for i in tqdm(range(smi_len)):
        for j in range(i, smi_len):
            matrix[i,j] = DataStructs.TanimotoSimilarity(db_fp[smiles[i]], 
                                                        db_fp[smiles[j]]) # calculate tanimoto similarity based on morgan fingerprint between two SMILES strings at position i and j in the list `smiles`
    # fill in the lower triangle without having to loop (saves ~50% of time)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix)) 
    # fill the diagonal whth 0'set
    np.fill_diagonal(matrix, 0)

    return matrix

# scaffold similarity based on morgan fingerprint
def get_scaffold_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for the scaffolds of a list of SMILES string """
    db_scaf = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        try:
            skeleton = GraphFramework(mol) # returns the generic scaffold graph, whcih represents the connectivity and topology of the molecule
        except Exception: # In the very rare case that the molecule cannot be processed, then use a normal scaffold
            print(f"Could not create a generic scaffold of {smi}, then used a normal scaffold instead")
            skeleton = GetScaffoldForMol(mol) # returns the Murcko scaffold, which is the result of removing side chains while retaining ring systems
        skeleton_fp = AllChem.GetMorganFingerprintAsBitVect(skeleton, radius=radius, nBits=nBits)
        db_scaf[smi] = skeleton_fp

    smi_len = len(smiles)
    matrix = np.zeros([smi_len, smi_len])
    # calcultate the upper triangle of the matrix
    for i in tqdm(range(smi_len)):
        for j in range(i, smi_len):
            matrix[i,j] = DataStructs.TanimotoSimilarity(db_scaf[smiles[i]], 
                                                        db_scaf[smiles[j]]) # calculate tanimoto similarity based on morgan fingerprint between two SMILES strings at position i and j in the list `smiles`
    # fill in the lower triangle without having to loop (saves ~50% of time)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    # fill the diagonal whth 0'set
    np.fill_diagonal(matrix, 0)

    return matrix

# levenstein similarity based on SMILES strings
def get_levenshtein_matrix(smiles: List[str], normalize: bool = True):
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
                matrix[i,j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
            else:
                matrix[i,j] = levenshtein(smiles[i], smiles[j])
    # fill in the lower triangle without having to loop (saves ~50% of time)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    # get from a distance matrix to a similarity matrix
    matrix = 1 - matrix

    # fill the diagonal whth 0'set
    np.fill_diagonal(matrix, 0)

    return matrix

def molecule_similarity(smiles: List[str], similarity: float = 0.9):
    """ Calculate which pairs of molecules have a high tanimoto, scaffold, or SMILES similarity """
    m_tani = get_tanimoto_matrix(smiles) >= similarity
    m_scaff = get_scaffold_matrix(smiles) >= similarity
    m_leve = get_levenshtein_matrix(smiles) >= similarity

    return (m_tani + m_scaff + m_leve).astype(int)

#===================2. Molecular activity cliff  ====================
#def find_fc(a: float, b: float):
#    """Get the fold change of to bioactivities (deconvert from log10 if needed)"""
#
#    return max([a, b]) / min([a, b])

def find_fc(a: float, b: float):
    """Get the fold change of to bioactivities, where a and b are the pStandard_value of two compounds"""

    return max([a, b]) - min([a, b])

def get_fc(pStandard_value: List[float]):
    """ Calculates the pairwise fold difference in compound activity given a list of activities"""

    act_len = len(pStandard_value)
    matrix = np.zeros((act_len, act_len))
    # calcultate the upper triangle of the matrix
    for i in tqdm(range(act_len)):
        for j in range(i, act_len):
            matrix[i,j] = find_fc(pStandard_value[i], pStandard_value[j])
    # fill in the lower triangle without having to loop (saves ~50% of time)
    matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    # fill the diagonal whth 0'set
    np.fill_diagonal(matrix, 0)

    return matrix

class ActivityCliffs:
    """ Activity cliff class that computes cliff compounds and returns a list of SMILES strings that are activity cliffs """
    def __init__(self, smiles: List[str], pStandard_value: Union[List[float], np.array]):
        self.smiles = smiles
        self.pStandard_value = pStandard_value if isinstance(pStandard_value, list) else list(pStandard_value)
        self.cliffs = None

    def find_cliffs(self, similarity: float = 0.9, potency_fold: float = 1):
        """ Compute activity cliffs

                :param similarity: (float) threshold value to determine structural similarity
                :param potency_fold: (float) threshold value to determine difference in bioactivity
        """

        sim = molecule_similarity(self.smiles, similarity)

        # get a matrix indicating which pairs of compounds have a difference in pStandard_value greater than the specified fold change
        fc = (get_fc(self.pStandard_value) > potency_fold).astype(int)

        # computes the activity cliffy by taking the logical AND of the structurral similarity and pStandard_value difference matrices
        self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)

        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        """
        get the molecules that are involved in the activity cliffs

        :param return_smiles: (bool) return activity cliff molecules as a list of SMILES strings
        :param kwargs: arguments for ActivityCliffs.find_cliffs()
        :return: (List[int]) returns a binary list where 1 means activity cliff compounds

        """
        if self.cliffs is None:
            self.find_cliffs(**kwargs)

        if return_smiles:
            # sum(cliffs) returns the sum of each column in the matrix, here is a binary list indicating which molecules have activity cliffs
            # np.where(sum(matrix) >0 ) returns the indices of the columns that where the molecules have activity cliffs
            # Thus smiles[i] for i in np.where(sum(matrix) >0 ) returns the SMILES strings of the molecules that have activity cliffs 
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        else:
            # return a binary list indicating the indices of the molecules that have activity cliffs
            return list((sum(self.cliffs) > 0).astype(int))
    
    def __repr__(self):
        return "Activity cliffs"