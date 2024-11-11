
import pandas as pd

def remove_dup_mols(df):
    
    """
    Entries with multiple  annotations were included once with the arithmetic mean when the standard deviation of pstardard_value annotations was within 1 log unit;
    Otherwise, the entry was excluded.
    
    """
    # group by 'molecule_chembl_id' and calculate the mean and standard deviation of 'pstandard_value'
    df_group = df.groupby('compound_chembl_id')['pchembl_value'].agg(['mean', 'std'])
    # find where the standard deviation is greater than 1, and drop these rows. Then keep the first row of the rows with the same 'molecule_chembl_id'
    df = df[~df['compound_chembl_id'].isin(df_group[df_group['std'] > 1].index)].drop_duplicates(subset='compound_chembl_id', keep='first').copy()
    # map the mean of 'pstandard_value' to the 'molecule_chembl_id' in the original dataframe
    df['pchembl_value'] = df['compound_chembl_id'].map(df_group['mean'])
    # reset the index
    df = df.reset_index(drop=True)

    return df

