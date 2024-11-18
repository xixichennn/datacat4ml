"""
Conducts hyperparameter tuning for the models in the pipeline.
"""

import os
import sys
import warnings
from tqdm import tqdm
import argparse

# inner modules
from datacat4ml.const import *
from datacat4ml.utils import mkdirs
from datacat4ml.Scripts.model_dev.data_process import Data
from datacat4ml.Scripts.model_dev.ml import RFC, RFR
from datacat4ml.Scripts.model_dev.tune_alpha_low import BayesianOptimization4reg, BayesianOptimization4cls

# ===================  Hyperparameter tuning ====================
N_CALLS = 50 # n optimization attempts
algo4reg = [RFR]
algo4cls = [RFC]

def hyperparam_tune(file_path = SPLIT_CAT_DATASETS_DIR, task: str = 'cls', use_clustering: int=1,
                    use_smote: int=1, descriptor: str='ECFP4'):
    
    use_smote = bool(use_smote)
    use_clustering = bool(use_clustering)

    print(f"file_path: {file_path}\n")
    print(f"task: {task}\n")
    print(f"use_clustering: {use_clustering}\n")

    file_folder = os.path.join(file_path, task, 'use_clustering' +'_'+str(use_clustering))
    filenames = os.listdir(file_folder)
    for filename in tqdm(filenames):
        print(f"file: {filename}\n")

        print(f"use_smote: {use_smote}\n")
        if task == 'reg':
            use_smote = False
            algos = algo4reg
            bayesianopt = BayesianOptimization4reg
        elif task == 'cls':
            use_smote = use_smote
            algos = algo4cls
            bayesianopt = BayesianOptimization4cls

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
                try:
                    # Get the hyperparameter space for the given algorithm
                    print(f"read hyperparam space for {algo.__name__} ...")
                    hyperparam_space = (os.path.join(HYPERPARAM_SPACE_DIR, f"{algo.__name__}.yml"))
                    print(f"hyperparam_space: {hyperparam_space}")
                    print(f'Done')

                    # Perform hyperparameter tuning using Bayesian optimization
                    print(f"tuning hyperparam for {algo.__name__} ...")
                    opt = bayesianopt(algo, task)
                    if data.x_smote_train is not None:
                        print(f"smote is used")
                        opt.optimize(data.x_smote_train, data.y_smote_train, hyperparam_space, n_calls= N_CALLS)
                    else:
                        print(f"smote is not used")
                        opt.optimize(data.x_train, data.y_train, hyperparam_space, n_calls= N_CALLS)
                    print(f'Done')

                    # Save best hyperparameters as a yaml file
                    print(f"save best hyperparam for {algo.__name__} ...")
                    output_dir = os.path.join(BEST_CONFIG_CAT_DIR, task, 'use_clustering' +'_'+str(use_clustering), 'use_smote'+'_'+str(use_smote), filename[:-10])
                    mkdirs(output_dir)
                    print(f"output_dir: {output_dir}")
                    opt.save_config(os.path.join(output_dir, f"{algo.__name__}_{descriptor}.yml"))
                    print("Done")

                    # Plot the optimiztion progress and save the figure in the output_dir
                    print(f"plot optimization progress for {algo.__name__} ...")
                    opt.plot_progress(os.path.join(output_dir, f"{algo.__name__}_{descriptor}.png"))
                    print("Done")
                
                except Exception as e:
                    warnings.warn(f" -- FAILED to optimize hyperparams for {filename}-{task}-use_smote_{use_smote}{algo.__name__}-{descriptor} :\n {e} --")
        
        except Exception as e:
            warnings.warn(f" -- FAILED to create Data object for {filename}: {e} --")
            continue

# ============================== main ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--file_path', type=str, help='path to the dataset')
    parser.add_argument('--task', type=str, help='reg or cls')
    parser.add_argument('--use_clustering', type=int, help='use clustering or not')
    parser.add_argument('--use_smote', type=int, default=1, help='use smote or not')
    parser.add_argument('--descriptor', type=str, default='ECFP4', help='descriptor category')

    args = parser.parse_args()

    hyperparam_tune(file_path=args.file_path, 
                    task=args.task, 
                    use_clustering=args.use_clustering, 
                    use_smote=args.use_smote, 
                    descriptor=args.descriptor)