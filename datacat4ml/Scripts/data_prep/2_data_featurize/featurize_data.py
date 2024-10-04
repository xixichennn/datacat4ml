# inner module import
import sys
sys.path.append("/storage/homefs/yc24j783/datacat4ml/datacat4ml")
from const import FETCH_DATA_DIR, FEATURIZE_DATA_DIR

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as pl

from rdkit import Chem
from rdkit.Chem import AllChem
from mapchiral.mapchiral import encode

#============================ Functions ============================
def calc_ecfp4(smi:str, radius=2, nbits=2048):
    '''
    Cacluate the ECFP4 fingerprint of a 'canonical_smiles' string.
    '''
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    
    return fp

def calc_map4c(smi:str,max_radius=2, n_permutations=2048):
    '''
    Cacluate the map4c fingerprint of a 'canonical_smiles' string.
    ''' 
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = encode(mol, max_radius=max_radius, n_permutations=n_permutations)

    return fp

def get_activity(pchembl_value:float):
    ''' active is defined as pchembl_value > 7 '''
    if pchembl_value > 7:
        return 'active'
    elif 5 < pchembl_value <= 7:
        return 'intermediate'
    else:
        return 'inactive'
    
def process_df(inputfolder:str, inputfile:str, outputfolder, outputfile:str):
    '''
    Process the dataframe and save the processed dataframe to a new csv file.
    '''
    
    # load the data
    df = pd.read_csv(os.path.join(inputfolder, inputfile))
    print(f'The shape of the {inputfile[:-8]}df is {df.shape}')

    # calculate the ECFP4 fingerprint
    df['ecfp4'] = df['canonical_smiles'].apply(calc_ecfp4)
    # calculate the MAP4C fingerprint
    df['map4c'] = df['canonical_smiles'].apply(calc_map4c)
    # calculate the activity (active, intermediate, inactive)
    df['activity'] = df['pchembl_value'].apply(get_activity)

    # save the processed dataframe to a pickle file (not csv file, otherwise the fingerprint will be saved as string)
    df.to_pickle(os.path.join(outputfolder, outputfile))

#================================ Main =================================
def main():
    process_df(FETCH_DATA_DIR, 'ic50_mincur_8_data.csv', FEATURIZE_DATA_DIR, 'ic50_mincur_8_fp.pkl')
    process_df(FETCH_DATA_DIR, 'ic50_maxcur_8_data.csv', FEATURIZE_DATA_DIR, 'ic50_maxcur_8_fp.pkl')
    process_df(FETCH_DATA_DIR, 'ki_mincur_8_data.csv', FEATURIZE_DATA_DIR, 'ki_mincur_8_fp.pkl')
    process_df(FETCH_DATA_DIR, 'ki_maxcur_8_data.csv', FEATURIZE_DATA_DIR, 'ki_maxcur_8_fp.pkl')
    process_df(FETCH_DATA_DIR, 'ec50_mincur_8_data.csv', FEATURIZE_DATA_DIR, 'ec50_mincur_8_fp.pkl')
    process_df(FETCH_DATA_DIR, 'ec50_maxcur_8_data.csv', FEATURIZE_DATA_DIR, 'ec50_maxcur_8_fp.pkl')

    folders = ['ic50_mincur', 'ki_mincur', 'ec50_mincur']
    for folder in folders:
        print(f'Processing {folder}...')
        for file in os.listdir(os.path.join(FETCH_DATA_DIR, folder)):
            if file[-4:] == '.csv':
                print(f'Processing {file}...')
                process_df(os.path.join(FETCH_DATA_DIR, folder), file, os.path.join(FEATURIZE_DATA_DIR, folder), file[:-4]+'_fp.pkl')


if __name__ == "__main__":
    main()