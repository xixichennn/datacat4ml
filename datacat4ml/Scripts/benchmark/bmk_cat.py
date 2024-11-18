# conda env: pyg (Python 3.9.16)
"""
build a benchmarking pipeline for machine learning models and save the results to result.csv
"""
import os
import sys
import warnings
from tqdm import tqdm
import joblib
import argparse
import pandas as pd

# inner modules
from datacat4ml.const import *
from datacat4ml.utils import mkdirs
from datacat4ml.Scripts.model_dev.data_process import Data
from datacat4ml.Scripts.model_dev.ml import RFC, RFR
from datacat4ml.Scripts.model_dev.metrics import *
from datacat4ml.Scripts.model_dev.tune_alpha_low import get_config

#=========================Benchmarking==========================
algo4reg = [RFR]
algo4cls = [RFC]

def write_results(result_file, data, 
                  threshold:float, n_actives:int, n_inactives:int, active_ratio:float, n_active_train:int, n_inactive_train:int, active_ratio_train:float, n_active_test:int, n_inactive_test:int, active_ratio_test:float,
                  accuracy="None", precision="None",recall="None", mcc="None", bedroc_dec5="None", bedroc_2="None", bedroc_8="None", 
                  rmse="None", cliff_rmse="None", r2="None", cliff_r2="None",
                  file_path = SPLIT_CAT_DATASETS_DIR, task: str='cls', use_clustering: bool=True, use_smote: bool=True, 
                  descriptor: str='ECFP4', algoname=RFC):
    
    """
    Write benchmarking results to a file
    """

    output_dir = BMK_CAT_DIR
    mkdirs(output_dir)
    result_path= os.path.join(output_dir, result_file)

    if file_path == SPLIT_CAT_DATASETS_DIR:
        file_path_name = 'CAT_ORs'
    elif file_path == SPLIT_HET_DATASETS_DIR:
        file_path_name = 'HET_ORs'

    n_compounds = len(data.y_train)+len(data.y_test)
    n_compounds_train = len(data.y_train)
    n_compounds_test = len(data.y_test)

    if task == 'cls':
        n_cliff_compounds = 'NA'
        n_cliff_compounds_train = 'NA'
        n_cliff_compounds_test = 'NA'
    elif task == 'reg':
        n_cliff_compounds = sum(data.cliff_mols_train)+sum(data.cliff_mols_test)
        n_cliff_compounds_train = sum(data.cliff_mols_train)
        n_cliff_compounds_test = sum(data.cliff_mols_test)

    # Create output file if it doesn't exist already
    if not os.path.isfile(result_path):
        with open(result_path, 'w') as f:
            f.write('file_path,task,use_clustering,use_smote,'
                    'target,effect,assay,std_type,descriptor,algo,'
                    'n_compounds,n_cliff_compounds,n_compounds_train,n_cliff_compounds_train,n_compounds_test,n_cliff_compounds_test,'
                    'threshold,n_actives, n_inactives,active_ratio, n_active_train, n_inactive_train, active_ratio_train, n_active_test, n_inactive_test, active_ratio_test,' 
                    'accuracy, precision, recall, mcc, bedroc_dec5, bedroc_2, bedroc_8,'
                    'rmse, cliff_rmse, r2, cliff_r2\n')
            
    with open(result_path, 'a') as f:
        f.write(f'{file_path_name},{task},{use_clustering},{use_smote},'
                f'{data.target},{data.effect},{data.assay},{data.std_type},{descriptor},{algoname},'
                f'{n_compounds},{n_cliff_compounds},{n_compounds_train},{n_cliff_compounds_train},{n_compounds_test},{n_cliff_compounds_test},'
                f'{threshold},{n_actives},{n_inactives},{active_ratio},{n_active_train},{n_inactive_train},{active_ratio_train},{n_active_test},{n_inactive_test},{active_ratio_test},'
                f'{accuracy},{precision},{recall},{mcc},{bedroc_dec5},{bedroc_2},{bedroc_8},'
                f'{rmse},{cliff_rmse},{r2},{cliff_r2} \n')
        

def benchmark_result(result_file: str = "results_ml.csv", file_path=SPLIT_CAT_DATASETS_DIR, task: str = 'cls',
                     use_clustering: int=1, use_smote: int=1, descriptor: str='ECFP4'):
    
    use_clustering = bool(use_clustering)
    use_smote = bool(use_smote)

    print(f"file_path: {file_path}\n")
    print(f"task: {task}\n")
    print(f"use_clustering: {use_clustering}\n")

    file_folder = os.path.join(file_path, task, 'use_clustering' +'_'+str(use_clustering))
    print(f"file_folder is {file_folder}")

    filenames = os.listdir(file_folder)
    print(f"filenames is {filenames}")
    for filename in tqdm(filenames):
        print(f"file: {filename}\n")
        df = pd.read_csv(os.path.join(file_folder, filename))

        threshold = df['threshold'].iloc[0]
        n_actives = df['activity'].sum()
        n_inactives = len(df) - n_actives
        active_ratio = n_actives / len(df)
        n_active_train = df[df['split'] == 'train']['activity'].sum()
        n_inactive_train = len(df[df['split'] == 'train']) - n_active_train
        active_ratio_train = n_active_train / len(df[df['split'] == 'train'])
        n_active_test = df[df['split'] == 'test']['activity'].sum()
        n_inactive_test = len(df[df['split'] == 'test']) - n_active_test
        active_ratio_test = n_active_test / len(df[df['split'] == 'test'])
        
        print(f"use_smote: {use_smote}\n")
        if task == 'reg':
            use_smote = False
            algos = algo4reg
        elif task == 'cls':
            use_smote = use_smote
            algos = algo4cls
        
        # create a Data object
        try:
            data = Data(file_folder, filename, task, use_smote)


            print(f"descriptor: {descriptor}\n")

            # Featurize SMILES strings with the given descriptor
            data.featurize_data(descriptor)            
            
            if task == 'cls' and use_smote:
                data.balance_data()
                data.shuffle()
            else:
                data.shuffle()

            for algo in algos:
                print(f"algo: {algo.__name__}\n")
                config_path = os.path.join(BEST_CONFIG_CAT_DIR, task, 'use_clustering' +'_'+str(use_clustering), 
                                        'use_smote'+'_'+str(use_smote), filename[:-10], f"{algo.__name__}_{descriptor}.yml")
                model_path = os.path.join(MODELS_CAT_DIR, task, 'use_clustering' +'_'+str(use_clustering), 
                                            'use_smote'+'_'+str(use_smote), filename[:-10], f"{algo.__name__}_{descriptor}.joblib")
                if not os.path.isdir(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))

                try:
                    # Get the best hyperparmeters stored in the config file
                    print(f"read best config ...")
                    best_config = get_config(config_path)
                    print('Done')

                    # Train the model with the best hyperparameters
                    print(f"train model ...")
                    f = algo(task, **best_config)

                    if data.x_smote_train is not None:
                        print(f"smote is used")
                        f.train(data.x_smote_train, data.y_smote_train)
                    else:
                        print(f"smote is not used") 
                        f.train(data.x_train, data.y_train)
                    print('Done')

                    # Save the model
                    print(f"save model ...")
                    with open(model_path, 'wb') as handle:
                        joblib.dump(f, handle)
                    print('Done')

                    # Evaluate the model
                    print(f"evaluate model ...")
                    y_pred = f.predict(data.x_test)
                    if task == 'cls':
                        accuracy = calc_accuracy(data.y_test, y_pred)
                        precision = calc_precision(data.y_test, y_pred)
                        recall = calc_recall(data.y_test, y_pred)
                        mcc = calc_mcc(data.y_test, y_pred)
                        y_pred_proba = f.predict_proba(data.x_test)
                        bedroc_dec5 = calc_bedroc(y_pred_proba=y_pred_proba, y_true=data.y_test, alpha=321.9)
                        bedroc_2 = calc_bedroc(y_pred_proba=y_pred_proba, y_true=data.y_test, alpha=80.5)
                        bedroc_8 = calc_bedroc(y_pred_proba=y_pred_proba, y_true=data.y_test, alpha=20.0)

                        r2, cliff_r2, rmse, cliff_rmse = None, None, None, None

                    elif task == 'reg':
                        r2 = calc_r2(data.y_test, y_pred)
                        cliff_r2 = calc_cliff_r2(y_test_pred=y_pred, y_test=data.y_test,
                                                cliff_mols_test=data.cliff_mols_test)
                        rmse = calc_rmse(data.y_test, y_pred)
                        cliff_rmse = calc_cliff_rmse(y_test_pred=y_pred, y_test=data.y_test,
                                                        cliff_mols_test=data.cliff_mols_test)
                        accuracy, precision, recall, mcc, bedroc_dec5, bedroc_2, bedroc_8 = None, None, None, None
                    print('Done')
                    
                    # Write the results to a csv file
                    print(f"write results ...")
                    write_results(result_file, data, 
                                  threshold, n_actives, n_inactives, active_ratio, n_active_train, n_inactive_train, active_ratio_train, n_active_test, n_inactive_test, active_ratio_test,
                                  accuracy, precision,recall,mcc, bedroc_dec5, bedroc_2,bedroc_8, 
                                  rmse, cliff_rmse, r2, cliff_r2,
                                  file_path, task, use_clustering, use_smote, 
                                  descriptor, algo.__name__)
                    
                    print("Done")
                    # check the results by loading it as a pandas dataframe
                
                    print('######################')

                    
                except:
                        warnings.warn(f" -- FAILED {filename}, {task}, use_smote_{use_smote}, {algo.__name__}-{descriptor}")
                    
        except Exception as e:
            warnings.warn(f" -- FAILED to create Data object for {filename}: {e} --")

# ============================== main ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model building and benchmarking')
    parser.add_argument('--result_file', type=str, required=True, help='path to the result file')
    parser.add_argument('--file_path', type=str, required=True, help='path to the data folder')
    parser.add_argument('--task', type=str, required=True, help='task: cls or reg')
    parser.add_argument('--use_clustering', type=int, required=True, help='use clustering or not')
    parser.add_argument('--use_smote', type=int, required=True, help='use smote or not')
    parser.add_argument('--descriptor', type=str, required=True, help='descriptor')

    args = parser.parse_args()

    benchmark_result(result_file=args.result_file,
                     file_path=args.file_path,
                     task=args.task,
                     use_clustering=args.use_clustering,
                     use_smote=args.use_smote,
                     descriptor=args.descriptor)