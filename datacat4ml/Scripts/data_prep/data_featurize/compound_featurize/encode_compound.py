import os
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
import pandas as pd
import argparse
from loguru import logger

from joblib import Parallel, delayed
from tqdm import tqdm # not `import tqdm` because it will mask potential mistakes with `delayed`
from tqdm.contrib.concurrent import process_map
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder

from datacat4ml.const import SPLIT_DATA_DIR

# ========================= SparseMorganEncoder =========================
class SparseMorganEncoder:
    """
    Adopted from github repo ml-jku/clamp
    """
    def __init__(self, radius=2, fp_size=1024, njobs=1):
        self.radius = radius
        self.fp_size = fp_size
        self.njobs = njobs
        if fp_size > 65535: # the `unit16`can store values in the range of [0, 65535]
            raise ValueError('fp_size must be <= 65535 (unit16) for sparse matrix representation')
    
    def encode(self, list_of_smiles):
        fps = Parallel(n_jobs=self.njobs)(
            delayed(self._get_morgan_fingerprint)(smiles) for smiles in tqdm(list_of_smiles)
        ) # distributed parallel processing for efficiency
        return self._sparse_matrix_from_fps(fps)

    def _get_morgan_fingerprint(self, smiles):
        mol=Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.fp_size)
            return np.array(fp.GetOnBits(), dtype=np.uint16) # `GetOnBits` provides a compact representation by returning only the indices of the bits that are `1`.
        else:
            return None
    
    def _sparse_matrix_from_fps(self, on_bits):
        n_samples = len(on_bits)
        sparse_matrix = sparse.lil_matrix((n_samples, self.fp_size), dtype=bool) # `lil_matrix`, list of list, is efficient for row-wise addition.
        for i, fp in enumerate(on_bits):
            if fp is not None:
                sparse_matrix[np.array([i]*len(fp)), fp] = True
        print('convert to csr for efficient savings')
        return sparse_matrix.tocsr() # `csr_matrix` is efficient for storage and numerical operations.

# ========================= FpEncoder utils =========================
#def disable_rdkit_logging():
#    """
#    Disables RDKit logging to avoid cluttering the output with warnings and messages.
#    """
#    import rdkit.rdBase as rkrb
#    import rdkit.RDLogger as rkl
#    logger.setLevel(rkl.ERROR)
#    rkrb.DisableLog('rdApp.error')

def ebv2np(ebv):
    """Converts an explicit bit vector (EBV) returned by RDKit into a numpy array of integers(0s and 1s)."""
    return np.frombuffer(bytes(ebv.ToBitString(), 'utf-8'), 'u1') - ord('0') # `ord('0')` returns the ASCII value of '0', which is '48'.

def getFingerprint(smiles, fp_size=4096, which='morgan', radius=2, sanitize=True):
    """
    maccs +  morganc + topologicaltorsion + erg + atompair + pattern + rdkc
    """

    if isinstance(smiles, list):
        return np.array([getFingerprint(smi, fp_size, which, radius) for smi in smiles]).max(0) # max pooling if it's list of lists #?Yu: what is max pooling?
    
    mol = Chem.MolFromSmiles(str(smiles), sanitize=False)

    if mol is None:
        logger.warning(f"{smiles} couldn't be converted to a fingerprint using 0's instead")
        return np.zeros(fp_size).astype(np.bool) #?Yu: remove this line if can't figure out its usage
    
    if sanitize:
        faild_op = Chem.SanitizeMol(mol, catchErrors=True)
        FastFindRings(mol) # Providing ring info

    mol.UpdatePropertyCache(strict=False) # Correcting valence info # important operation

    def mol2np(mol, fp_size, which):
        is_dict = False
        if which =='morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_size, useFeatures=False, useChirality=True)
        elif which == 'rdk':
            fp = Chem.RDKFingerprint(mol, fsSize=fp_size, maxPath=6)
        elif which == 'rdkc':
            # https://greglandrum.github.io/rdkit-blog/similarity/reference/2021/05/26/similarity-threshold-observations1.html
            # -- maxPath 6 found to be better for retrieval in databases
            fp = AllChem.UnfoldedRDkFingerprintCountBased(mol, maxPath=6).GetNonzeroElements()
            is_dict = True
        elif which == 'morganc':
            fp = AllChem.GetMorganFingerprint(mol, radius, useChirality=True, useBondTypes=True, useFeatures=True, useCounts=True).GetNonzeroElements()
            is_dict = True
        elif which == 'topologicaltorsion':
            fp = AllChem.GetTopologicalTorsionFingerprint(mol).GetNonzeroElements()
            is_dict = True
        elif which == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif which == 'erg':
            v = AllChem.GetErGFingerprint(mol)
            fp = {idx:v[idx] for idx in np.nonzero(v)[0]}
            is_dict = True
        elif which == 'atompair':
            fp = AllChem.GetAtomPairFingerprint(mol).GetNonzeroElements()
            is_dict = True
        elif which == 'pattern':
            fp = Chem.PatternFingerprint(mol, fpSize=fp_size)
        elif which == 'ecfp4':
            # roughly equivalent to Morgan with radius=2
            fp = AllChem.GetMorganFingerprintAsBitvect(mol, radius=2, nBits=fp_size, useFeatures=False, useChirality=True)
        elif which == 'layered':
            fp = AllChem.LayeredFingerprint(mol, fpSize=fp_size, maxPath=7)
        elif which == 'mhfp':
            #Todo check if one can avoid instantiating the MHFP encoder
            fp = MHFPEncoder().EncodeMol(mol, radius=radius, rings=True, isomeric=False, kekulize=False, min_radius=1)
            fp = {f:1 for f in fp}
            is_dict = True
        elif not (type(which)==str):
            fp = which(mol)
        
        if is_dict:
            nd = np.zeros(fp_size)
            for k in fp:
                nk = k%fp_size #remainder
                # print(nk, k, fp_size)
                # 3160 36322170 3730
                # print(nd[nk], fp[k])
                if nd[nk]!=0:
                    #print('c', end='')
                    nd[nk] = nd[nk] + fp[k] # pooling colisions
                nd[nk] = fp[k]

            return nd #np.log(1+nd) 

        return ebv2np(fp)

    """ + for folding, * for concat"""
    cc_symb= '*'
    if ('+' in which) or (cc_symb in which):
        concat = False
        split_sym = '+'
        if cc_symb in which:
            concat = True
            split_sym = '*'
        
        np_fp = np.zeros(fp_size)
        remaining_fps = (which.count(split_sym)+1)
        fp_length_remain = fp_size

        for fp_type in which.split(split_sym):
            if concat:
                fpp = mol2np(mol, fp_length_remain//remaining_fps, fp_type)
                np_fp[(fp_size-fp_length_remain):(fp_size-fp_length_remain+len(fpp))] += fpp
                fp_length_remain -= len(fpp)
                remaining_fps -= 1
            else:
                try:
                    fpp = mol2np(mol, fp_size, fp_type)
                    np_fp[:len(fpp)] += fpp
                except:
                    pass
        return np.log(1 + np_fp)
    else:
        return mol2np(mol, fp_size, which)

def _getFingerprint(input):
  return getFingerprint(input[0], input[1], input[2], input[3])

def convert_smiles_to_fp(list_of_smiles, fp_size=2048, which='ecfp4', radius=2, njobs=1, verbose=False): #?Yu why fp_size=2048?; Todo: remove is_smarts
    """
    list of smiles can be list of lists, then the resulting array will be padded to the max_list_length.
    which: morgan, rdk, ecfp4, or object;
    Note: 
    """

    input = [(smi, fp_size, which, radius) for smi in list_of_smiles]
    if verbose: 
        print(f'starting pool with {njobs} workers')
    if njobs > 1:
        fps = process_map(_getFingerprint, input, max_workers=njobs, chunksize=1, mininterval=1)
    else:
        fps = [getFingerprint(smi, fp_size=fp_size, which=which, radius=radius) for smi in list_of_smiles]
    
    return np.array(fps)

# ========================= FpEncoder =========================
class FpEncoder:
    def __init__(self, fp_size=8192, fp_type='morganc+rdkc', radius=2, njobs=32, disable_logging=True): #?Yu fp_size=8192=1024*8
        self.fp_size = fp_size
        self.fp_type = fp_type
        self.radius = radius
        self.njobs = njobs
        self.disable_logging = disable_logging

        self.convert_smiles_to_fp = convert_smiles_to_fp

        #if self.disable_logging:
        #    disable_rdkit_logging()

        if self.fp_type == 'MxFP':
            self.fp_type = 'maccs+morganc+topologicaltorsion+erg+atompair+pattern+rdkc+mhfp+rdkd'
    
    def encode(self, list_of_smiles):
        return self.convert_smiles_to_fp(list_of_smiles, 
                                         fp_size=self.fp_size,
                                         which=self.fp_type,
                                         radius=self.radius,
                                         njobs=self.njobs,
                                         verbose=False)

# ========================== Main Function =========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute RDKit sparse features for the GPCR compounds from ChEMBL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--compounds2smiles', help='Path to a Parquet file mapping CIDs to SMILES strings.')
    parser.add_argument('--fp_type', help='Fingerprint type, e.g. sprsFP, morganc+rdkc, MxFP, cddd, mlruns', default='morganc+rdkc', type=str)
    parser.add_argument('--fp_size', help='Fingerprint size', default=8192, type=int) 
    parser.add_argument('--njobs', help='Number of jobs to run in parallel', default=32, type=int)
    parser.add_argument('--smiles_column', help='Column name for SMILES strings', default='CanonicalSMILES', type=str)

    args = parser.parse_args()
    compound2smiles_df = pd.read_parquet(args.compounds2smiles)
    path = os.path.join(os.path.dirname(args.compounds2smiles), 'encoded_compounds')
    os.makedirs(path, exist_ok=True)

    compound2smiles = compound2smiles_df.set_index('CID')[args.smiles_column].to_dict()

    compounds = compound2smiles_df['CID'].squeeze().tolist()

    logger.info(f'converting {len(compounds)} smiles to features')

    list_of_smiles = [compound2smiles[c] for c in compounds]

    #Yu?
    #if 'mlruns' in args.fp_type:
    #    encoder = MLRUNEncoder(args.fp_type) #Yu todo: set fp_size and assay_feature_size
    #    args.fp_type = args.fp_type.split('/')[-1]

    if args.fp_type == 'sprsFP':
        encoder = SparseMorganEncoder(radius=2, fp_size=1024, njobs=args.njobs)
    
    #elif args.fp_type == 'clamp':
    #    logger.info('Using CLAMP encoder')
    #    encoder = ClampEncoder() #Todo: path to model
    else:
        encoder = FpEncoder(fp_type=args.fp_type, fp_size=args.fp_size, njobs=args.njobs)

    x = encoder.encode(list_of_smiles)

    p= os.path.join(path, f'compound_features_{args.fp_type}.npy')
    logger.info(f'Save compound features with shape {x.shape} to {p}')
    np.save(p, x) if args.fp_type!='sprsFP' else sparse.save_npz(p, x)