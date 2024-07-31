#====================== import libraries and modules======================#
from datetime import date
import os
import sys
import pandas as pd

# inner modules
from datacat4ml.const import DATA_DIR

# chembl_api
from chembl_webresource_client import new_client
from chembl_webresource_client.new_client import new_client
target_api = new_client.target
assay_api = new_client.assay
activity_api = new_client.activity
molecule_api = new_client.molecule

#====================== fetch target data ======================#
def fetch_single_protein(uniprot_id):
    # fetch target data
    targets = target_api.get(target_components__accession=uniprot_id)
    # dowload target data
    targets = pd.DataFrame.from_records(targets)
    # save 'single protein' data
    target = targets.iloc[0]
    target_chembl_id = target.target_chembl_id
    print(f"The target CHEMBL ID for single protein is {target_chembl_id}")

    return target_chembl_id

#====================== fetch activity data ======================#
def download_activity_data(target_chembl_id):
    activity = activity_api.filter(target_chembl_id=target_chembl_id)
    activity_df = pd.DataFrame.from_records(activity)
    print(f"The shape of {activity_df} is {activity_df.shape}")
    ## add a column 'assay_confidence_score' to the dataframe
    activity_df['confidence_score'] = activity_df['assay_chembl_id'].apply(lambda x: assay_api.get(x)['confidence_score'])
    print(f"{str(len(activity_df))} molecules for {target_chembl_id} collected")

    return activity_df

def check_data_validity(activity_df):
    validated_activity_df = activity_df[~activity_df['data_validity_comment'].isin(['Potential missing data', 
                                                                                    'Potential author error',
                                                                                    'Author confirmed error',
                                                                                    'Non standard unit for type', 
                                                                                    'Outside typical range',
                                                                                    'Potential transcription error'])]
    print(f"The shape of {validated_activity_df} is {validated_activity_df.shape}")

    return validated_activity_df

#====================== execute above functions ======================#
def main():

    # Define the target UniProt IDs
    target_ids = {
        'mor': 'P35372',
        'kor': 'P41145',
        'dor': 'P41143',
        'nor': 'P41146'
    }

    # Fetch target data and download activity data
    act_data = {}
    
    # create a folder to store data
    today = date.today()
    today_str = str(today)
    today_dir = os.path.join(DATA_DIR, 'data_prep', 'data_fetch', today_str)
    if not os.path.exists(today_dir):
        os.makedirs(today_dir)
        
    for target_name, uniprot_id in target_ids.items():
        target_chembl_id = fetch_single_protein(uniprot_id)
        act_df = download_activity_data(target_chembl_id)
        act_data[target_name] = act_df
        
        # Store activity data as pickle file and csv file
        file_path = os.path.join(today_dir, f'{target_name}_act.pkl')
        act_df.to_pickle(file_path)
        file_path = os.path.join(today_dir, f'{target_name}_act.csv')
        act_df.to_csv(file_path)

        # Check data validity and store validated activity data as pickle file and csv file
        validated_act_df = check_data_validity(act_df)
        validated_file_path = os.path.join(today_dir, f'{target_name}_act_validated.pkl')
        validated_act_df.to_pickle(validated_file_path)
        validated_file_path = os.path.join(today_dir, f'{target_name}_act_validated.csv')
        validated_act_df.to_csv(validated_file_path)

if __name__ == "__main__":
    main()
