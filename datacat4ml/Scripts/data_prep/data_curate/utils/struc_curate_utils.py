from rdkit import Chem
from rdkit.Chem import AllChem

""" 
Code adopted from MoleculeACE (https://github.com/molML/MoleculeACE.git)

Curating the molecules by removing salts, sanitizing and neutralizing the charges.
"""

def curate_struct(smiles, remove_salts=True, sanitize=True, neutralize=True):
    # initialize the outputs
    salt = []
    failed_sanit = []
    neutralized = []

    # performs the molecule preparation based on the flags
    if remove_salts and smiles is not None:
        if "." in smiles:  # checks if salts
            salt = True
        else:
            salt = False

    if sanitize is True:
        smiles, failed_sanit = sanitize_mol(smiles)

    if neutralize is True and failed_sanit is False:
        smiles, neutralized = neutralize_mol(smiles)
    
    return smiles, salt, failed_sanit, neutralized

def sanitize_mol(smiles):
    """ Sanitizes a molecule using rdkit """

    failed_sanit = False

    # == basic checks on SMILES validity
    mol = Chem.MolFromSmiles(smiles)

    # flags: Kekulize, check valencies, set aromaticity, conjugation and hybridization
    san_opt = Chem.SanitizeFlags.SANITIZE_ALL

    # check if the conversion to mol was successful, return otherwise
    if mol is None:
        failed_sanit = True
    # sanitization based on the flags (san_opt)
    else:
        sanitize_fail = Chem.SanitizeMol(mol, catchErrors=True, sanitizeOps=san_opt)  #  the sanitization fails, the SanitizeMol() function returns an error message
        if sanitize_fail: # If sanitize_fail is not None, it means that sanitization failed, 
            failed_sanit = True
            raise ValueError(sanitize_fail)  # returns if failed

    return smiles, failed_sanit

# ====== neutralizes charges based on the patterns specified in _InitialiseNeutralisationReactions

def _InitialiseNeutralisationReactions():
    """ adapted from the rdkit contribution of Hans de Winter """
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[nH]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    # use list comprehension to returns a list of tuples where each tuple contains two mol objects.
    # The first mol in each tuple is a SMARTS pattern that defines the substructures to be replaced,
    # while the second mol is the corresponding replacement structure.
    return [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

def neutralize_mol(smiles):
    neutralized = False
    mol = Chem.MolFromSmiles(smiles)

    # retrieves the transformations
    transfm = _InitialiseNeutralisationReactions()  # set of transformations

    # applies the transformations
    for i, (reactant, product) in enumerate(transfm):
        while mol.HasSubstructMatch(reactant):
            neutralized = True
            rms = AllChem.ReplaceSubstructs(mol, reactant, product) # returns a list of modified molecules
            mol = rms[0] # in this case, there is only one molecule in the list, so we take the first one
    # converts back the molecule to smiles
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) # encode sterochemistry information using the SMILES syntax, which includes a stereochemistry desciptor for each chiral center in the molecule.

    return smiles, neutralized

def struct_curator(df):
    """
    Curates the structures of the molecules in the dataset, including removing salts, sanitizing and neutralizing molecules.

    Parameters
    ----------
    df : pandas.DataFrame. Obtained from data_categorize

    returns
    -------
    curated_struct_df : pandas.DataFrame. Contains the curated structures of the molecules.
    """
    curated_struct_df = df.copy()  # copy for further editing

    # structure curation
    for index, row in curated_struct_df.iterrows():
        smiles = row['canonical_smiles']
        # if the smiles is not None as well as the data type is string
        if type(smiles) == str:
            smiles, salt, failed_sanit, neutralized = curate_struct(smiles, remove_salts=True, sanitize=True, neutralize=True)
        else:
            smiles = 'missing'
            salt = False
            failed_sanit = True
            neutralized = True
            
    
        curated_struct_df.loc[index, 'IsSalt'] = salt
        curated_struct_df.loc[index, 'FailedSanit'] = failed_sanit
        curated_struct_df.loc[index, 'Neutralized'] = neutralized
        curated_struct_df.loc[index, 'StandardizedSMILES'] = smiles

    # Clean the new df (salts & failures in sanitization)
    # remove salts
    curated_struct_df.drop(curated_struct_df.loc[curated_struct_df['IsSalt'] == True].index, inplace=True)
    # remove failed sanitization
    curated_struct_df.drop(curated_struct_df.loc[curated_struct_df['FailedSanit'] == True].index, inplace=True)
    # reset the index
    curated_struct_df.reset_index(drop=True, inplace=True)

    return curated_struct_df