import os
import pandas as pd

from datacat4ml.utils import get_df_name, mkdirs
from datacat4ml.const import FETCH_DATA_DIR, ASSAY_CHEMBL_IDS, CAT_DATASETS_DIR, CAT_GPCR_DIR
from datacat4ml.const import OR_chemblids, OR_names, OR_name_chemblids

############################# Read data #############################
ki_gpcr_df = pd.read_csv(os.path.join(FETCH_DATA_DIR, 'Ki_gpcr_maxcur_8_data.csv'))
ic50_gpcr_df = pd.read_csv(os.path.join(FETCH_DATA_DIR, 'IC50_gpcr_maxcur_8_data.csv'))
ec50_gpcr_df = pd.read_csv(os.path.join(FETCH_DATA_DIR, 'EC50_gpcr_maxcur_8_data.csv'))

# Filter data for each target
# ORs ##
OR_dfs = {}
for target_chembl_id, name in zip(OR_chemblids, OR_names):
    ki_df = ki_gpcr_df[ki_gpcr_df['target_chembl_id'] == target_chembl_id]
    ic50_df = ic50_gpcr_df[ic50_gpcr_df['target_chembl_id'] == target_chembl_id]
    ec50_df = ec50_gpcr_df[ec50_gpcr_df['target_chembl_id'] == target_chembl_id]
    
    act_df = pd.concat([ki_df, ic50_df, ec50_df], ignore_index=True)
    OR_dfs[name] = act_df
    
#    print(f'The shape of {name}_df is \n ki: {ki_df.shape}, ic50: {ic50_df.shape}, ec50: {ec50_df.shape}')

## GPCRs ##
GPCR_dfs = {}
union_targets = set(ki_gpcr_df['target_chembl_id']) | set(ic50_gpcr_df['target_chembl_id']) | set(ec50_gpcr_df['target_chembl_id'])
for target_chembl_id in union_targets:
    ki_df = ki_gpcr_df[ki_gpcr_df['target_chembl_id'] == target_chembl_id]
    ic50_df = ic50_gpcr_df[ic50_gpcr_df['target_chembl_id'] == target_chembl_id]
    ec50_df = ec50_gpcr_df[ec50_gpcr_df['target_chembl_id'] == target_chembl_id]

    act_df = pd.concat([ki_df, ic50_df, ec50_df], ignore_index=True)
    GPCR_dfs[target_chembl_id] = act_df

#    print(f'The shape of {target_chembl_id}_df is \n ki: {ki_df.shape}, ic50: {ic50_df.shape}, ec50: {ec50_df.shape}')
    

###################### Functions ######################
def print_df_info(df: pd.DataFrame) -> None:
    """
    Print information about a given dataframe
    """
    df_name = get_df_name(df)
    print(f"The shape of {df_name} is {df.shape}")
    print(f"#assay_desc:\n{df['assay_desc'].describe()}\n")
    print(f"#canonical_smiles:\n{df['canonical_smiles'].describe()}\n")

class DataCategoizer:
    """ 
    a class to analyze the data for each target, effect, assay, and standard_type

    Attributes
    ----------
    target: string
        target name, e.g. mor, kor, dor, nor
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
    def __init__(self, targets='ORs', target='mor', target_chembl_id='CHEMBL233',
                 effect='bind', assay='RBA', std_type='Ki', 
                 pattern:str="", pattern_ex: str= ""):
        self.targets = targets
        self.target = target
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
        act_df = GPCR_dfs[self.target_chembl_id]
        type_df = act_df[act_df['standard_type'] == self.std_type]

        return type_df
    
    def save_assay_desc(self, assay_df:pd.DataFrame, file_suffix:str = "",
                        saveFile:bool =True, saveFileOut:bool = True):

        """
        Save the assay_desc and its counts for entries that match the given pattern

        parameters
        ----------
        type_df: dataframe
            dataframe for each 'standard_type', e.g. Ki, IC50, etc.
        effect_type_df: dataframe
            dataframe for each 'standard_type' and tested pharmacological effect, e.g. Ki_ago
        target: string
            target name, e.g. mor, kor, dor, nor
        type: string
            'standard_type', e.g. Ki, IC50, etc.
        effect: string
            tested pharmacological effect, e.g. binding affinity, agonism, antagonism, etc.

        return
        ------
        None

        """

        type_df = self.generate_type_df()
        # Subtract entries that match the given pattern from 'type_df'
        assay_out_df = pd.merge(type_df, assay_df, how='left', indicator=True)
        assay_out_df = assay_out_df[assay_out_df['_merge'] == 'left_only'].drop(columns='_merge')
        
        if self.targets == 'ORs':
            file_path = os.path.join(CAT_DATASETS_DIR, self.target_chembl_id, self.effect, self.assay, self.std_type)
        elif self.targets == 'GPCRs':
            file_path = os.path.join(CAT_GPCR_DIR, self.target_chembl_id, self.effect, self.assay, self.std_type)
            
        mkdirs(file_path)
        if saveFile:
            # save assay_desc and its counts for entries that match the given pattern
            VC = assay_df['assay_desc'].value_counts()
            VC_df = pd.DataFrame({'count': VC.values, 'assay_desc': VC.index})

            VC_df.to_excel(os.path.join(file_path, f'{self.target_chembl_id}_{self.effect}_{self.assay}_{self.std_type}{file_suffix}.xlsx'), index=False)
        
        if saveFileOut:
            nVC = assay_out_df['assay_desc'].value_counts()
            nVC_df = pd.DataFrame({'count': nVC.values, 'assay_desc': nVC.index})

            nVC_df.to_excel(os.path.join(file_path, f'{self.target_chembl_id}_{self.effect}_{self.assay}_{self.std_type}{file_suffix}_out.xlsx'), index=False)

    def match_effect_type(self):
        """
        Retrieve activity records for each target, type, and effect.

        Parameters:
        - pattern: pattern to match
        - pattern_ex: pattern to exclude
        - target: target name (default: 'mor')
        - type: standard_type (default: 'Ki')
        - effect: tested pharmacological effect (default: 'bind')
        - save_file: whether to save the matching records (default: True)
        - save_file_out: whether to save the non-matching records (default: True)
        
        Returns:
        - effect_type_df: dataframe for a type matching the specified effect for a target
        """

        type_df = self.generate_type_df()

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

    def generate_final_df(self):
        """
        only for target in 'ORs'
        
        Merge the matching entries and additional entries, 
        and then save the entries that out of the pattern for manual checking.
        
        """

        type_df = self.generate_type_df()

        effect_type_df = self.match_effect_type()

        plus_df = self.generate_plus_df(plus_chembl_id=ASSAY_CHEMBL_IDS[self.target][self.effect][self.assay][self.std_type]['plus'])
        exclude_df = self.generate_exclude_df(exclude_chembl_id=ASSAY_CHEMBL_IDS[self.target][self.effect][self.assay][self.std_type]['exclude'])
        
        # Add plus_df to original_df
        combined_df = pd.concat([effect_type_df, plus_df], ignore_index=True)

        # Subtract exclude_df from combined_df
        final_df = pd.merge(combined_df, exclude_df, how='left', indicator=True)
        final_df = final_df[final_df['_merge'] == 'left_only'].drop(columns='_merge')

        final_out_df = pd.merge(type_df, final_df, how='left', indicator=True)
        final_out_df = final_out_df[final_out_df['_merge'] == 'left_only'].drop(columns='_merge')

        self.save_assay_desc(final_df, file_suffix='Final', saveFile=False, saveFileOut=True)

        return final_df, final_out_df

def categorize_GPCRs(targets='GPCRs', effect='bind', assay='RBA', std_types=['Ki', 'IC50'], 
                  pattern:str="", pattern_ex:str=""):
      
    """
    categorize the data for each target, effect, assay, and standard_type
  
    parameters
    ----------
    effect: string.
        tested pharmacological effect, e.g. binding affinity, agonism, antagonism, etc.
    assay: string.
        assay type, e.g. RBA etc.
    std_types: list of strings.
        'standard_type', e.g. Ki, IC50, etc.
    pattern: string.
        pattern to match
    pattern_ex: string.
        pattern to exclude
      
    return
    ------
    type_dfs, effect_type_dfs, len_dfs: dictionaries
    """

    # binding affinity in Ki and IC50 data
    type_dfs = {}
    effect_type_dfs = {}
    len_dfs = {}
    
    for target_chembl_id in union_targets:
        print(f"Target: {target_chembl_id}\n")
        target_dir = os.path.join(CAT_GPCR_DIR, target_chembl_id)
        mkdirs(target_dir)
        
        print(f"Effect: {effect}\n")
        effect_dir = os.path.join(target_dir, effect)

        print(f"Assay: {assay}\n")
        assay_dir = os.path.join(effect_dir, assay)
        mkdirs(assay_dir)

        print(f"Pattern: {pattern}\n")
        print(f"Pattern_ex: {pattern_ex}\n")

        for std_type in std_types:
            print(f"Standard type: {std_type}\n")
            std_type_dir = os.path.join(assay_dir, std_type)
            mkdirs(std_type_dir)

            categorizer = DataCategoizer(targets=targets, target='', target_chembl_id=target_chembl_id,
                                          effect=effect, assay=assay, std_type=std_type, pattern=pattern, pattern_ex=pattern_ex)

            # Generate type_df: e.g. mor_Ki_df, mor_IC50_df
            type_df_name = f"{target_chembl_id}_{std_type}_df"
            type_df =categorizer.generate_type_df()
            type_df.to_csv(os.path.join(std_type_dir, f"{type_df_name}.csv"))
            print(f"The shape of type_df is {type_df.shape}\n")
            type_dfs[type_df_name] = type_df

            # Generate effect_type_df based on regex pattern: e.g. mor_bind_Ki_df, mor_bind_IC50_df
            effect_type_df_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}Final_df"
            effect_type_df =categorizer.match_effect_type()
            effect_type_df.to_csv(os.path.join(std_type_dir, f"{effect_type_df_name}.csv"))
            print_df_info(effect_type_df)
            effect_type_dfs[effect_type_df_name] = effect_type_df
            
            # Get length of each dataframe
            len_type_df = len(type_df)
            len_effect_type_df = len(effect_type_df)
            
            # make a pandas dataframe that contains length of each dataframe as a row
            len_df_name = f"{target_chembl_id}_{effect}_{assay}_{std_type}_len_df"
            len_df = pd.DataFrame({'target': target_chembl_id, 'effect': effect, 'assay': assay, 'std_type': std_type,
                                    'type_df': len_type_df, 'effect_type_df': len_effect_type_df}, index=[0])
            len_dfs[len_df_name] = len_df


            print('##########################')
        print('================================================================')
      
    return type_dfs, effect_type_dfs, len_dfs

def categorize_ORs(targets='ORs', effect='bind', assay='RBA', std_types=['Ki', 'IC50'], 
                  pattern:str="", pattern_ex:str=""):

      # binding affinity in Ki and IC50 data
      type_dfs = {}
      effect_type_dfs = {}
      plus_dfs = {}
      exclude_dfs = {}
      final_dfs = {}
      final_out_dfs = {}
      len_dfs = {}

      for target in OR_names:
            target_chembl_id = OR_name_chemblids[target]
            print(f'target_chembl_id: {target_chembl_id}\n')

            print(f"Target: {target_chembl_id}\n")
            target_dir = os.path.join(CAT_DATASETS_DIR, target_chembl_id)
            mkdirs(target_dir)
            
            print(f"Effect: {effect}\n")
            effect_dir = os.path.join(target_dir, effect)

            print(f"Assay: {assay}\n")
            assay_dir = os.path.join(effect_dir, assay)
            mkdirs(assay_dir)

            print(f"Pattern: {pattern}\n")
            print(f"Pattern_ex: {pattern_ex}\n")

            for std_type in std_types:
                print(f"Standard type: {std_type}\n")
                std_type_dir = os.path.join(assay_dir, std_type)
                mkdirs(std_type_dir)

                plus_chembl_id = ASSAY_CHEMBL_IDS[target][effect][assay][std_type]['plus']
                exclude_chembl_id = ASSAY_CHEMBL_IDS[target][effect][assay][std_type]['exclude']
                

                categorizer = DataCategoizer(targets=targets, target=target, target_chembl_id= target_chembl_id, 
                                             effect=effect, assay=assay, std_type=std_type, 
                                             pattern=pattern, pattern_ex=pattern_ex)

                # Generate type_df: e.g. mor_Ki_df, mor_IC50_df
                type_df_name = f"{target}_{std_type}_df"
                type_df =categorizer.generate_type_df()
                type_df.to_csv(os.path.join(std_type_dir, f"{type_df_name}.csv"))
                print(f"The shape of type_df is {type_df.shape}\n")
                type_dfs[type_df_name] = type_df

                # Generate effect_type_df: e.g. mor_bind_Ki_df, mor_bind_IC50_df
                effect_type_df_name = f"{target}_{effect}_{assay}_{std_type}_df"
                effect_type_df =categorizer.match_effect_type()
                effect_type_df.to_csv(os.path.join(std_type_dir, f"{effect_type_df_name}.csv"))
                print_df_info(effect_type_df)
                effect_type_dfs[effect_type_df_name] = effect_type_df

                # Generate plus_df: e.g. mor_bind_KiPlus_df, mor_bind_IC50Plus_df
                plus_df_name = f"{target}_{effect}_{assay}_{std_type}Plus_df"
                plus_df =categorizer.generate_plus_df(plus_chembl_id=plus_chembl_id)
                plus_df.to_csv(os.path.join(std_type_dir, f"{plus_df_name}.csv"))
                print(f"The shape of plus_df is {plus_df.shape}\n")
                plus_dfs[plus_df_name] = plus_df  

                # Generate exclude_df: e.g. mor_bind_KiExclude_df, mor_bind_IC50Exclude_df
                exclude_df_name = f"{target}_{effect}_{assay}_{std_type}Exclude_df"
                exclude_df =categorizer.generate_exclude_df(exclude_chembl_id=exclude_chembl_id)
                exclude_df.to_csv(os.path.join(std_type_dir, f"{exclude_df_name}.csv"))
                print(f"The shape of exclude_df is {exclude_df.shape}\n")
                exclude_dfs[exclude_df_name] = exclude_df

                # Generate final_df: e.g. mor_bind_KiFinal_df, mor_bind_IC50Final_df
                final_df_name = f"{target}_{effect}_{assay}_{std_type}Final_df"
                final_out_df_name = f"{target}_{effect}_{assay}_{std_type}FinalOut_df"
                final_df, final_out_df =categorizer.generate_final_df()
                final_df.to_csv(os.path.join(std_type_dir, f"{final_df_name}.csv"))
                print(f"The shape of final_df is {final_df.shape}\n")
                final_dfs[final_df_name] = final_df
                final_out_dfs[final_out_df_name] = final_out_df
                
                # Get length of each dataframe
                len_type_df = len(type_df)
                len_effect_type_df = len(effect_type_df)
                len_plus_df = len(plus_df)
                len_exclude_df = len(exclude_df)
                len_final_df = len(final_df)
                len_final_out_df = len(final_out_df)

                # make a pandas dataframe that contains length of each dataframe as a row
                len_df_name = f"{target}_{effect}_{assay}_{std_type}_len_df"
                len_df = pd.DataFrame({'target': target, 'effect': effect, 'assay': assay, 'std_type': std_type,
                                        'type_df': len_type_df, 'effect_type_df': len_effect_type_df,
                                        'plus_df': len_plus_df, 'exclude_df': len_exclude_df,
                                        'final_df': len_final_df, 'final_out_df': len_final_out_df}, index=[0])
                len_dfs[len_df_name] = len_df


                print('##########################')
            print('================================================================')
      
      return type_dfs, effect_type_dfs, plus_dfs, exclude_dfs, final_dfs, len_dfs
