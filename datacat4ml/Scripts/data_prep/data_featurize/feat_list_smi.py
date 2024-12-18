import sys
from typing import List
import os
import argparse

import numpy as np
import pandas as pd

from datacat4ml.const import *
from datacat4ml.utils import mkdirs

#===================== Utility functions =====================#
def mol_from_smi(smi: str):
    """ Create a list of RDkit mol objects from a list of SMILES strings """
    from rdkit.Chem import MolFromSmiles
    mol = MolFromSmiles(smi) 
    return mol

def rdkit_numpy_convert(fp):
    """ Convert the RDKit fingerprints to a numpy arrays"""
    from rdkit.DataStructs import ConvertToNumpyArray
    output = []
    for f in fp:
        arr = np.zeros((1,)) # initialize array of zeros of length 1
        ConvertToNumpyArray(f, arr) # convert RDKit fingerprint to numpy array
        output.append(arr)
    
    return np.asarray(output)

def embed_mol(smi: str):
    """
    This code block is adoped from MoleculeACE.
    Try to slove problem of bad conformer ID
    """

    from rdkit.Chem import AllChem, AddHs, MolFromSmiles
    
    # Construct mol object add hydrogens
    m = MolFromSmiles(smi)
    mh = AddHs(m)

    # Use distance geometry to obtain initial coordinates for a molecule
    embed = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=True, randomSeed=0xf00d,
                                    maxAttempts=5)

    if embed == -1:
        print(f"failed first attempt for molecule {smi}, trying for more embedding attempts")
        embed = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=True, randomSeed=0xf00d,
                                        clearConfs=False, enforceChirality=False, maxAttempts=45)

    if embed == -1:
        print(f"failed second attempt for molecule {smi}, trying embedding w/o using basic knowledge")
        embed = AllChem.EmbedMolecule(mh, useRandomCoords=True, useBasicKnowledge=False, randomSeed=0xf00d,
                                        clearConfs=True, enforceChirality=False, maxAttempts=1000)

    if embed == -1:
        raise RuntimeError(f"FAILED embedding {smi}")

    AllChem.MMFFOptimizeMolecule(mh, maxIters=1000, mmffVariant='MMFF94')

    return mh

#===================== Class Featurizer =====================#
class Featurizer:
    def __init__(self):
        pass
    
    # Fingerprints
    @staticmethod
    def ecfp4(smi: List[str], radius: int = 2, nbits: int = 1024):
        """ Convert a list of SMILES to ECFP fingerprints """
        from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
        mols = [ mol_from_smi(m) for m in smi]
        fp = [ GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits) for mol in mols]
        return rdkit_numpy_convert(fp)

    @staticmethod
    def ecfp6(smi: List[str], radius: int = 3, nbits: int = 1024):
            """ Convert a list of SMILES to ECFP fingerprints """
            from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
            mols = [mol_from_smi(m) for m in smi]
            fp = [GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits) for mol in mols]
            return rdkit_numpy_convert(fp)
    
    @staticmethod
    def maccs(smi: List[str]):
        """ Convert a list of SMILES strings to MACCs fingerprints"""
        from rdkit.Chem import MACCSkeys
        mols = [mol_from_smi(m) for m in smi]
        fp = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        return rdkit_numpy_convert(fp)
    
    @staticmethod
    def rdkit_fp(smi: List[str]):
        """ Convert a list of SMILES strings to RDKit(topological) fingerprints"""
        from rdkit.Chem import RDKFingerprint
        mols = [mol_from_smi(m) for m in smi]
        fp = [RDKFingerprint(mol) for mol in mols]

        return rdkit_numpy_convert(fp)
    
    @staticmethod
    def pharm2d(smi: List[str]):
        """ Convert a list of SMILES strings to RDKit 2D fingerprints"""
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
        from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
        from rdkit.DataStructs import ExplicitBitVect

        mols = [mol_from_smi(m) for m in smi]
        fp = [Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory) for mol in mols]

        # convert to bit string
        fp_bit_str = [f.ToBitString() for f in fp]
        # convert to explicit bit vector
        explicit_bit_vect = [ExplicitBitVect(len(str)) for str in fp_bit_str] 
        return rdkit_numpy_convert(explicit_bit_vect)
    
    @staticmethod
    def erg(smi: List[str]):
        """ Convert a list of SMILES strings to RDKit extended reduced graph fingerprints"""
        from rdkit.Chem import rdReducedGraphs
        
        mols = [mol_from_smi(m) for m in smi]
        fp = [rdReducedGraphs.GetErGFingerprint(mol) for mol in mols]

        return fp
    
    # Physicochemical properties
    @staticmethod
    def calc_physchem(smi: List[str]):
        """ Compute the physicochemical properties of a list of SMILES strings """

        from rdkit.Chem import Descriptors
        from rdkit import Chem


        mols = [mol_from_smi(m) for m in smi]
        X = []
        for mol in mols:
            weight = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_bond_donor = Descriptors.NumHDonors(mol)
            h_bond_acceptors = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            atoms = Chem.rdchem.Mol.GetNumAtoms(mol)
            heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)
            molar_refractivity = Chem.Crippen.MolMR(mol)
            topological_polar_surface_area = Chem.QED.properties(mol).PSA
            formal_charge = Chem.rdmolops.GetFormalCharge(mol)
            rings = Chem.rdMolDescriptors.CalcNumRings(mol)

            X.append(np.array([weight, logp, h_bond_donor, h_bond_acceptors, rotatable_bonds, atoms, heavy_atoms,
                                molar_refractivity, topological_polar_surface_area, formal_charge, rings]))

        return np.array(X)
    
    # 3D descriptors
    @staticmethod
    def calc_shape3d(smi: List[str]):
        """ Calculate 3D descriptors that are related to the shape and size of the molecules"""

        from rdkit.Chem import rdMolDescriptors
        
        mols = [embed_mol(m) for m in smi]

        X = []
        for mol in mols:
            x = []
            x.append(rdMolDescriptors.CalcPBF(mol))
            x.append(rdMolDescriptors.CalcPMI1(mol))
            x.append(rdMolDescriptors.CalcPMI2(mol))
            x.append(rdMolDescriptors.CalcPMI3(mol))
            x.append(rdMolDescriptors.CalcNPR1(mol))
            x.append(rdMolDescriptors.CalcNPR2(mol))
            x.append(rdMolDescriptors.CalcRadiusOfGyration(mol))
            x.append(rdMolDescriptors.CalcInertialShapeFactor(mol))
            x.append(rdMolDescriptors.CalcEccentricity(mol))
            x.append(rdMolDescriptors.CalcAsphericity(mol))
            x.append(rdMolDescriptors.CalcSpherocityIndex(mol))

            X.append(np.array(x))

        return np.array(X)
    
    @staticmethod
    def calc_autocorr3d(smi: List[str]):
        """ Calculate 3D autocorrelation descriptor, which represents the spatial distribution of atom properties in the molecules"""

        from rdkit.Chem import rdMolDescriptors
        
        mols = [embed_mol(m) for m in smi]

        X = []
        for mol in mols:
            x = rdMolDescriptors.CalcAUTOCORR3D(mol)
            X.append(x)

        return np.array(X)
    
    @staticmethod
    def calc_rdf(smi: List[str]):
        """ Calculate the radial distribution function descriptor, which represents the distribution of atoms in the molecule relative to a reference point"""

        from rdkit.Chem import rdMolDescriptors
        
        mols = [embed_mol(m) for m in smi]

        X = []
        for mol in mols:
            x= rdMolDescriptors.CalcRDF(mol)
            X.append(x)

        return np.array(X)

    @staticmethod
    def calc_morse(smi: List[str]):
        """ Calculate the MORSE descriptor, which represents the electronic and geometric properties of the molecules"""

        from rdkit.Chem import rdMolDescriptors

        mols = [embed_mol(m) for m in smi]
        X = []
        for mol in mols:
            x = rdMolDescriptors.CalcMORSE(mol)
            X.append(x)

        return np.array(X)
    
    @staticmethod
    def calc_whim(smi: List[str]):
        """ Calculate the weighted holistic invariant molecular descriptor, 
        which represents a combination of geometric and electronic properties of the molecule"""

        from rdkit.Chem import rdMolDescriptors

        mols = [embed_mol(m)for m in smi]
        X = []
        for mol in mols:
            x = rdMolDescriptors.CalcWHIM(mol)
            X.append(x)

        return np.array(X)
    
    @staticmethod
    def calc_getaway(smi: List[str]):
        """ Calculate the GETAWAY descriptor, which represents the electronic and geometric properties of the molecule"""

        from rdkit.Chem import rdMolDescriptors

        mols = [embed_mol(m)for m in smi]
        X = []
        for mol in mols:
            x = rdMolDescriptors.CalcGETAWAY(mol)
            X.append(x)

        return np.array(X)
    
    # ChemBERTa tokenization
    @staticmethod
    def chemberta_tokens(smi: List[str], max_len: int = 250, padding: bool = True, truncation: bool = True, auto_tokenizer: str = "seyonec/PubChem10M_SMILES_BPE_450k"):
        """ Tokenize a list of SMILES strings using ChemBERTa.

        :param smi: a list of SMILES strings
        :param max_len: maximum length of the tokenized SMILES string
        :param padding: whether to pad the tokenized SMILES string
        :param truncation: whether to truncate the tokenized SMILES string
        :param auto_tokenizer: name of the auto tokenizer from HuggingFace
        :return: Dictionary of tokenized SMILES string. The keys are 'input_ids', 'attention_mask', the values are the corresponding tensors.
        """

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(auto_tokenizer)
        tokens = tokenizer(smi, return_tensors="pt", padding=padding, truncation=truncation, max_length=max_len)

        return tokens
    
    # One-hot encoding
    @staticmethod
    def onehot(smi: List[str]):
        """ Convert a list of SMILES strings to one-hot encoding. OneHotFeaturizer from DeepChem is used.

        :param smi: a list of SMILES strings   
        """

        import deepchem as dc
        featurizer = dc.feat.OneHotFeaturizer()
        encodings = [featurizer.featurize(m) for m in smi]

        return np.array(encodings)
    
    # Graph convolutional featurization
    @staticmethod
    def graph_conv(smi: List[str], use_edges: bool = True, use_chirality: bool = True):
        """ Convert a list of SMILES strings to graph convolutional featurization. MolGraphConvFeaturizer from DeepChem is used.

        :param smi: a list of SMILES strings   
        """
        import deepchem as dc
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=use_edges, use_chirality=use_chirality)
        graphs = [featurizer.featurize(m) for m in smi]
        
        return np.array(graphs)

    def __call__(self, descriptor, **kwargs):
        if descriptor == 'ECFP4':
            return self.ecfp4(**kwargs)
        if descriptor == 'ECFP6':
            return self.ecfp6(**kwargs)
        if descriptor == 'MACCS':
            return self.maccs(**kwargs)
        if descriptor == 'RDKIT_FP':
            return self.rdkit_fp(**kwargs)
        if descriptor == 'PHARM2D':
            return self.pharm2d(**kwargs)
        if descriptor == 'ERG':
            return self.erg(**kwargs)
        if descriptor == 'PHYSICOCHEM':
            return self.calc_physchem(**kwargs)
        if descriptor == 'SHAPE3D':
            return self.calc_shape3d(**kwargs)
        if descriptor == 'AUTOCORR3D':
            return self.calc_autocorr3d(**kwargs)
        if descriptor == 'RDF':
            return self.calc_rdf(**kwargs)
        if descriptor == 'MORSE':
            return self.calc_morse(**kwargs)
        if descriptor == 'WHIM':
            return self.calc_whim(**kwargs)
        if descriptor == 'GETAWAY':
            return self.calc_getaway(**kwargs)
        if descriptor == 'TOKENS':
            return self.chemberta_tokens(**kwargs)
        if descriptor == 'ONEHOT':
            return self.onehot(**kwargs)
        if descriptor == 'GRAPH':
            return self.graph_conv(**kwargs)
        


# ===================== Featurize data =====================#
def featurize_data(filepath, descriptor_cat: str = 'FP', task:str = 'cls', confidence_score:int = 8, 
                   use_clustering:str = 'True'):
    
    """
    Instead of using 'class Featurizer' to get the featurized data as intermediate data, 
    this function serves to save the featurized data as a pickle file for later usage.
    
    save the featurized data as a pickle file
    """

    print(f"filepath is: {filepath}\n") # path to curated data
    print(f"task is: {task}\n")
    print(f"confidence_score is: {confidence_score}\n")
    print(f"use_clustering is: {use_clustering}\n")
    print(f"descriptor_class is: {descriptor_cat}\n")
    
    featurizer = Featurizer()    
    descriptors = Descriptors[descriptor_cat]

    # access the final csv files obtained from data curation
    folder_path = os.path.join(filepath, 'splited', task, 'confidence_score'+'_'+str(confidence_score), 
                               'use_clustering'+'_'+str(use_clustering))
    files = os.listdir(folder_path)
    final_files = [file for file in files if file.endswith('_final.csv')]

    # make new directory to store the featurized data
    new_path = os.path.join(filepath, 'featurized', task, 'confidence_score'+'_'+str(confidence_score), 
                            'use_clustering'+'_'+str(use_clustering), descriptor_cat)
    mkdirs(new_path)


    for final_file in final_files:
        print (f"final_file is: {final_file}\n")

        df = pd.read_csv(os.path.join(folder_path, final_file))
        df = df.drop(columns=['Unnamed: 0'])
    
        for descriptor in descriptors:
            print (f"descriptor is: {descriptor}\n")

            df[descriptor] = df['CuratedSmiles'].apply(lambda x: featurizer(descriptor, smi=x))

        # save the featurized data as a pickle file
        df.to_pickle(os.path.join(new_path, final_file[:-4]+'.pkl'))

# ===================== Main =====================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Featurize data with different combinations of parameters.")
    parser.add_argument("--descriptor_cat", choices=Descriptor_cats, required=True, help="Descriptor category (FP or Morgan)")
    parser.add_argument("--task", choices=Tasks, required=True, help="Task name")
    parser.add_argument("--confidence", type=int, choices=Confidence_scores, required=True, help="Confidence score")
    parser.add_argument("--use_clustering", required=True, help="Whether to use clustering")
    parser.add_argument("--filepath", required=True, help="Path to datasets directory")
    
    args = parser.parse_args()

    featurize_data(args.filepath, args.descriptor_cat, args.task, args.confidence, args.use_clustering)