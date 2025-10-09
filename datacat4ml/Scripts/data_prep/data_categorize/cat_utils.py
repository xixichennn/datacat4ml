import os
import pandas as pd

from typing import List

from datacat4ml.utils import get_df_name, mkdirs
from datacat4ml.const import EFFECT_TYPE_LOOKUP, CAT_MHD_OR_DIR, CAT_MHD_GPCR_DIR


###################### hhd ######################
def hhd(targets_list, GPCR_dfs, hhds_dir, ds_type='gpcr'):
    """
    
    Generate HHD (highly heterogeneous data) datasets for given targets.

    params:
    ----------
    targets_list: List[str]: List of target chembl IDs. e.g. GPCR_chemblids or OR_chemblids.
    GPCR_dfs: dict: Dictionary of dataframes for each target_chembl_id. Default, GPCR_dfs.
    hhds_dir: str: The directory to save the HHD datasets.

    return:
    -------
    None
    
    """
    type_dfs = {}
    hhd_dfs_len = {}

    for target_chembl_id in targets_list:
        target_df = GPCR_dfs[target_chembl_id]

        for std_type in ['Ki', 'IC50', 'EC50']:

            type_df = target_df[target_df['standard_type'] == std_type]
            if len(type_df) == 0:
                print(f"No data for {target_chembl_id} with standard_type {std_type}\n")
            else:
                type_df_name = f"{target_chembl_id}_{std_type}_hhd_df"
                file_path = os.path.join(hhds_dir, target_chembl_id, std_type)
                mkdirs(file_path)
                type_df.to_csv(os.path.join(file_path, f'{type_df_name}.csv' ), index=False)

                type_dfs[type_df_name] = type_df

                len_record = {
                    "ds_cat_level": "hhd",
                    "ds_type": ds_type,
                    "use_lookup": 'None',
                    "target_chembl_id": target_chembl_id,
                    "effect": 'None',
                    "assay": 'None',
                    "standard_type": std_type,
                    "assay_chembl_id": 'None',
                    "hhd_df": len(type_df),
                    "effect_type_df": 'None',
                    "plus_df": 'None',
                    "exclude_df": 'None',
                    "mhd_df": 'None',
                    "lhd_df": 'None'
                }

                hhd_df_len_name = f"{target_chembl_id}_{std_type}_len_df"
                hhd_dfs_len[hhd_df_len_name] = pd.DataFrame(len_record, index=[0])

    return type_dfs, hhd_dfs_len

######################## mhd ######################
def print_df_info(df: pd.DataFrame) -> None:
    """
    Print information about a given dataframe
    """
    df_name = get_df_name(df)
    print(f"The shape of {df_name} is {df.shape}")
    print(f"#assay_desc:\n{df['assay_desc'].describe()}\n")
    print(f"#canonical_smiles:\n{df['canonical_smiles'].describe()}\n")

class DataCategorizer:
    """

    Categorize the data for each target_chembl_id based on the effect, assay, and standard_type

    Attributes
    ----------
    dfs: dict
        dictionary of dataframes for each target_chembl_id. Default, GPCR_dfs.
    use_lookup: bool
        whether to use the lookup table (EFFECT_TYPE_LOOKUP)
    target_chembl_id: string
        target_chembl_id, e.g. CHEMBL233, CHEMBL237, CHEMBL236, CHEMBL2014
    effect: string
        tested pharmacological effect, e.g. binding affinity, agonism, antagonism, etc.
    assay: string
        assay type, e.g. RBA etc.
    std_type: string
        'standard_type', e.g. Ki, IC50, etc.
    pattern: string
        pattern to match
    pattern_ex: string
        pattern to exclude
    
    Methods
    -------
    generate_type_df()
        obtain the dataframe for each 'standard_type', e.g. Ki, IC50, etc.
    
    """
    def __init__(self, dfs: dict, use_lookup = True,
                 target_chembl_id='CHEMBL233', effect='bind', assay='RBA', std_type='Ki', 
                 pattern:str="", pattern_ex: str= ""):
        self.dfs = dfs
        self.use_lookup = use_lookup
        self.target_chembl_id = target_chembl_id
        self.effect = effect
        self.assay = assay
        self.std_type = std_type
        self.pattern = pattern
        self.pattern_ex = pattern_ex

    def generate_type_df(self):
        """
        obtain the dataframe for each 'standard_type', e.g. Ki, IC50, etc.
        """
        act_df = self.dfs[self.target_chembl_id]
        type_df = act_df[act_df['standard_type'] == self.std_type]

        return type_df
    
    def save_assay_desc(self, assay_df:pd.DataFrame, file_suffix:str = "",
                        saveFile:bool =True, saveFileOut:bool = True):

        """
        Save the assay_desc and its counts for entries that match the given pattern

        parameters
        ----------
        assay_df: pd.DataFrame
            dataframe for entries that match the given pattern
        file_suffix: str
            suffix for the saved file name
        saveFile: bool
            whether to save the matching records (default: True)
        saveFileOut: bool
            whether to save the non-matching records (default: True)

        return
        ------
        None

        """

        type_df = self.generate_type_df()
        if len(type_df) == 0:
            print(f"No data for {self.target_chembl_id} with standard_type {self.std_type}\n")
        else:
            # Subtract entries that match the given pattern from 'type_df'
            assay_out_df = pd.merge(type_df, assay_df, how='left', indicator=True)
            assay_out_df = assay_out_df[assay_out_df['_merge'] == 'left_only'].drop(columns='_merge')
            
            if self.use_lookup == True:
                file_path = os.path.join(CAT_MHD_OR_DIR, self.target_chembl_id, self.effect, self.assay, self.std_type)
            elif self.use_lookup == False:
                file_path = os.path.join(CAT_MHD_GPCR_DIR, self.target_chembl_id, self.effect, self.assay, self.std_type)

            mkdirs(file_path)
            if saveFile:
                # save assay_desc and its counts for entries that match the given pattern
                VC = assay_df['assay_desc'].value_counts()
                VC_df = pd.DataFrame({'count': VC.values, 'assay_desc': VC.index})

                if len(VC_df) > 0:
                    VC_df.to_csv(os.path.join(file_path, f'assay_{self.target_chembl_id}_{self.effect}_{self.assay}_{self.std_type}{file_suffix}.csv'), index=False)

            if saveFileOut:
                nVC = assay_out_df['assay_desc'].value_counts()
                nVC_df = pd.DataFrame({'count': nVC.values, 'assay_desc': nVC.index})

                if len(nVC_df) > 0:
                    nVC_df.to_csv(os.path.join(file_path, f'assay_{self.target_chembl_id}_{self.effect}_{self.assay}_{self.std_type}{file_suffix}_out.csv'), index=False)


    def match_effect_type(self):
        """
        Retrieve activity records for each target, type, and effect.

        Returns:
        - effect_type_df: dataframe for a type matching the specified effect for a target
        """

        type_df = self.generate_type_df()
        if len(type_df) == 0:
            print(f"No data for {self.target_chembl_id} with standard_type {self.std_type}\n")
        else:
            # Create boolean masks to filter matching and non-matching records
            mask = type_df['assay_desc'].str.contains(self.pattern)
            mask_ex = ~type_df['assay_desc'].str.contains(self.pattern_ex)

            effect_type_df = type_df[mask & mask_ex]
            self.save_assay_desc(effect_type_df, saveFile=True, saveFileOut=True)
            return effect_type_df

    def generate_plus_df(self, plus_chembl_id:list=[]):
        """
        Retrieve additional activity records for each target, type, and effect.

        Parameters:
        - plus_chembl_id: list of assay_chembl_id for additional entries (default: [])
        
        Returns:
        - plus_df: dataframe containing additional entries
        """

        type_df = self.generate_type_df()

        # Through manually checking, the assays that reported Ki values for binding affinity but don't match the above pattern
        plus_df = type_df[type_df['assay_chembl_id'].isin(plus_chembl_id)]
        self.save_assay_desc(plus_df, file_suffix='Plus', saveFile=True, saveFileOut=False)

        return plus_df
    
    def generate_exclude_df(self, exclude_chembl_id:list=[]):
        """
        exclude activity records for each target, type, and effect.

        Parameters:
        - exclude_chembl_id: list of assay_chembl_id for entries should be excluded (default: [])

        Returns:
        - exclude_df: dataframe containing entries should be excluded

        """

        type_df = self.generate_type_df()
        
        # Through manually checking, the assays that reported Ki values not for binding affinity but  match the above pattern
        exclude_df = type_df[type_df['assay_chembl_id'].isin(exclude_chembl_id)]
        self.save_assay_desc(exclude_df, file_suffix='Exclude', saveFile=True, saveFileOut=False)

        return exclude_df

    def generate_mhd_df(self):
        """
        only for target in 'ORs'
        
        Merge the matching entries and additional entries, 
        and then save the entries that out of the pattern for manual checking.
        
        """
        if self.use_lookup == False:
            raise ValueError("This method is only for targets in 'ORs' with use_lookup=True")
        
        elif self.use_lookup == True:

            type_df = self.generate_type_df()

            effect_type_df = self.match_effect_type()

            plus_df = self.generate_plus_df(plus_chembl_id=EFFECT_TYPE_LOOKUP[self.target_chembl_id][self.effect][self.assay][self.std_type]['plus'])
            exclude_df = self.generate_exclude_df(exclude_chembl_id=EFFECT_TYPE_LOOKUP[self.target_chembl_id][self.effect][self.assay][self.std_type]['exclude'])

            # Add plus_df to original_df
            combined_df = pd.concat([effect_type_df, plus_df], ignore_index=True)

            # Subtract exclude_df from combined_df
            mhd_df = pd.merge(combined_df, exclude_df, how='left', indicator=True)
            mhd_df = mhd_df[mhd_df['_merge'] == 'left_only'].drop(columns='_merge')

            mhd_out_df = pd.merge(type_df, mhd_df, how='left', indicator=True)
            mhd_out_df = mhd_out_df[mhd_out_df['_merge'] == 'left_only'].drop(columns='_merge')

            self.save_assay_desc(mhd_df, file_suffix='mhd', saveFile=False, saveFileOut=True)

            return mhd_df, mhd_out_df
        
def mhd_lhd(
    dfs: dict, 
    targets_list: List[str],
    use_lookup: bool = True, # True for ORs, False for GPCRs
    effect: str = "bind",
    assay: str = "RBA",
    std_types: list = ["Ki", "IC50"],
    pattern: str = "",
    pattern_ex: str = "",
    ds_type: str = 'gpcr' # 'or' for ORs, 'gpcr' for GPCRs
):
    """
    Generate MHD (median heterogeneous data) by categorizing GPCRs(use_lookup=False) or ORs(use_lookup=True).
    For each MHD, extract LHD (low heterogeneous data) based on each assay_chembl_id if the num of activities >= 50.

    Parameters
    ----------
    dfs: dict. 
        Dictionary of dataframes for each target_chembl_id. Default, GPCR_dfs. 
    targets_list : List[str]
        List of target chembl IDs. e.g. GPCR_chemblids or OR_chemblids.
    use_lookup : bool
        Whether to use the lookup table. True for ORs, False for GPCRs.
    effect : str
        Tested pharmacological effect, e.g. binding affinity.
    assay : str
        Assay type, e.g. RBA.
    std_types : list
        Standard types, e.g. ["Ki", "IC50"].
    pattern : str
        Regex pattern to match.
    pattern_ex : str
        Regex pattern to exclude.

    Returns
    -------
    dict
        Dictionaries with key as the categorized dataframes, and values as the length of each dataframe. 
        For ORs this includes type_dfs, effect_type_dfs, plus_dfs, exclude_dfs, mhd_dfs, len_dfs.
        For GPCRs only type_dfs, effect_type_dfs, len_dfs.
    """
    # Select which targets and base directory to use
    if use_lookup == True:
        base_dir = CAT_MHD_OR_DIR
    elif use_lookup == False:
        base_dir = CAT_MHD_GPCR_DIR

    # Initialize outputs
    type_dfs = {}
    effect_type_dfs = {}
    plus_dfs = {}
    exclude_dfs = {}
    mhd_dfs = {}
    mhd_dfs_len = {}
    lhd_dfs = {}
    lhd_dfs_len = {}

    for target_chembl_id in targets_list:
        print(f"Target: {target_chembl_id}\n")
        target_dir = os.path.join(base_dir, target_chembl_id)
        mkdirs(target_dir)

        print(f"Effect: {effect}\n")
        effect_dir = os.path.join(target_dir, effect)
        mkdirs(effect_dir)

        print(f"Assay: {assay}\n")
        assay_dir = os.path.join(effect_dir, assay)
        mkdirs(assay_dir)

        print(f"Pattern: {pattern}\n")
        print(f"Pattern_ex: {pattern_ex}\n")

        for std_type in std_types:
            print(f"Standard type: {std_type}\n")
            std_type_dir = os.path.join(assay_dir, std_type)
            mkdirs(std_type_dir)
            
            # ================================ generate mhd datasets ================================
            categorizer = DataCategorizer(
                dfs = dfs,
                use_lookup=use_lookup,
                target_chembl_id=target_chembl_id,
                effect=effect,
                assay=assay,
                std_type=std_type,
                pattern=pattern,
                pattern_ex=pattern_ex,
            )

            # Generate type_df
            type_df_name = f"{target_chembl_id}_{std_type}_df"
            type_df = categorizer.generate_type_df()
            if len(type_df) == 0:
                print(f"No data for {target_chembl_id} with standard_type {std_type}\n")
            else:
                type_dfs[type_df_name] = type_df
                print(f"The shape of type_df is {type_df.shape}\n")

                # Generate effect_type_df
                effect_type_df_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_df"
                effect_type_df = categorizer.match_effect_type()

                if len(effect_type_df) == 0:
                    print(f"No data for {target_chembl_id} with standard_type {std_type} and effect {effect}\n")
                else:
                    effect_type_df.to_csv(os.path.join(std_type_dir, f"{effect_type_df_name}.csv")) # the final mhd_df for gpcr datasets.
                    effect_type_dfs[effect_type_df_name] = effect_type_df
                    print_df_info(effect_type_df)

                    mhd_df_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_mhd_df"
                    # for GPCRs
                    if use_lookup == False:
                        plus_df = pd.DataFrame() # empty dataframe
                        exclude_df = pd.DataFrame() # empty dataframe
                        mhd_df = effect_type_df
                        
                    # For ORs: handle plus, exclude, mhd
                    if use_lookup == True:
                        plus_chembl_id = EFFECT_TYPE_LOOKUP[target_chembl_id][effect][assay][std_type]['plus']
                        exclude_chembl_id = EFFECT_TYPE_LOOKUP[target_chembl_id][effect][assay][std_type]['exclude']

                        plus_df_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_plus_df"
                        plus_df = categorizer.generate_plus_df(plus_chembl_id=plus_chembl_id)
                        plus_dfs[plus_df_name] = plus_df
                        print(f"The shape of plus_df is {plus_df.shape}\n")

                        exclude_df_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_exclude_df"
                        exclude_df = categorizer.generate_exclude_df(exclude_chembl_id=exclude_chembl_id)
                        exclude_dfs[exclude_df_name] = exclude_df
                        print(f"The shape of exclude_df is {exclude_df.shape}\n")

                        mhd_df, _ = categorizer.generate_mhd_df()
                    
                    if len(mhd_df) == 0:
                        print(f"No data for {target_chembl_id}_{effect}_{assay}_{std_type} \n")
                    else:
                        mhd_df.to_csv(os.path.join(std_type_dir, f"{mhd_df_name}.csv"), index=False)
                        mhd_dfs[mhd_df_name] = mhd_df
                        print(f"The shape of mhd_df is {mhd_df.shape}\n")
                        
                        # Track lengths
                        mhd_len_record = {
                                "ds_cat_level": "mhd",
                                "ds_type": ds_type,
                                "use_lookup": use_lookup,
                                "target_chembl_id": target_chembl_id,
                                "effect": effect,
                                "assay": assay,
                                "standard_type": std_type,
                                "assay_chembl_id": 'None',
                                "hhd_df": len(type_df),
                                "effect_type_df": len(effect_type_df),
                                "plus_df": len(plus_df),
                                "exclude_df": len(exclude_df),
                                "mhd_df": len(mhd_df),
                                "lhd_df": 'None'}
                        
                        mhd_df_len_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_len_df"
                        mhd_dfs_len[mhd_df_len_name] = pd.DataFrame(mhd_len_record, index=[0])
                        # ================================ generate lhd datasets ================================
                        # Get counts and filter valid IDs
                        id_counts = mhd_df['assay_chembl_id'].value_counts()

                        # the number of data points in a single assay should,on the one hand, be at least 50 to ensure the model can be trained;
                        # on the other hand, should not exceed 5000 to avoid high-throughput screens, as these are generally considered noisy
                        assay_chembl_ids = id_counts[(id_counts >= 50) & (id_counts <= 5000)].index.tolist()

                        if not assay_chembl_ids:
                            print(f"No valid IDs found for {std_type}. Skipping...")
                            continue

                        for assay_chembl_id in assay_chembl_ids:

                            print(f"assay_chembl_id: {assay_chembl_id}\n")
                            
                            lhd_df = mhd_df[mhd_df['assay_chembl_id'] == assay_chembl_id]

                            if len(lhd_df) == 0:
                                print(f"No data for {target_chembl_id}_{effect}_{assay}_{std_type}_{assay_chembl_id}\n")
                            else:
                                lhd_df_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_{assay_chembl_id}_lhd_df"
                                lhd_dir = os.path.join(std_type_dir, 'lhd')
                                mkdirs(lhd_dir)
                                lhd_df.to_csv(os.path.join(lhd_dir, f"{lhd_df_name}.csv"), index=False)

                                lhd_dfs[lhd_df_name] = lhd_df
                                print(f"The shape of lhd_df is {lhd_df.shape}\n")

                                lhd_len_record = {
                                    "ds_cat_level": "lhd",
                                    "ds_type": ds_type,
                                    "use_lookup": use_lookup,
                                    "target_chembl_id": target_chembl_id,
                                    "effect": effect,
                                    "assay": assay,
                                    "standard_type": std_type,
                                    "assay_chembl_id": assay_chembl_id,
                                    "hhd_df": len(type_df),
                                    "effect_type_df": len(effect_type_df),
                                    "plus_df": len(plus_df),
                                    "exclude_df": len(exclude_df),
                                    "mhd_df": len(mhd_df),
                                    "lhd_df": len(lhd_df)
                                }

                                lhd_df_len_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_{assay_chembl_id}_len_df"
                                lhd_dfs_len[lhd_df_len_name] = pd.DataFrame(lhd_len_record, index=[0])

                                print('##########################')


    print('================================================================')

    if use_lookup == True: # ors
        return type_dfs, plus_dfs, exclude_dfs, mhd_dfs, mhd_dfs_len, lhd_dfs, lhd_dfs_len
    elif use_lookup == False: # gpcrs
        return type_dfs, mhd_dfs, mhd_dfs_len, lhd_dfs, lhd_dfs_len


