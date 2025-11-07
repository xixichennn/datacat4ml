
import numpy as np
import pandas as pd
from typing import List

from datacat4ml.Scripts.data_prep.data_split.intSplit_mldata import random_split, cluster_kfold_split

#====================== class MLData ======================
def find_best_index(A: List[int], B: List[int], threshold: int = 2) -> int:
    """
    Finds the index 'i' where A[i] and B[i] are both strictly greater than 
    the threshold, and the sum (A[i] + B[i]) is maximized.

    params
    ------
    A: list of int => train_minClass_counts in outer splits
    B: list of int => test_minClass_counts in outer splits
    threshold: int => the number of folds for inner splits
        The threshold value to compare against.

    Returns the index of the best match, or 0 if match is found.
    """
    if all(x == 0 for x in A) or all(x == 0 for x in B):
        print(f'Each fold in outer splits only has one class in y_train and y_test, cannot generate any at least 2-fold inner splits.\n Skip model training for this dataset.')
        best_index = -1
    else:
        best_index = None
        max_sum = float('-inf')
        for i, (a_val, b_val) in enumerate(zip(A, B)):
            if a_val >= threshold and b_val >= threshold: # ensure at least 2 fold cross-validation.
                current_sum = a_val + b_val
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_index = i
        
        if best_index is None:
            print(f'No suitable outer fold found with both y_train and y_test having at least 2 points for each class (0, 1)\n. Take the first outer fold as default.')
            best_index = 0

    return best_index
    
def retrieve_splits(assignments, x, y, activity_ids, smis, spl, verbose=False):
    """
    A helper function to retrieve the split data based on the provided assignments.

    params
    ------
    assigns: list
        A list of assignments indicating 'train' or 'test' for each data point for a specific kind of splitting columns.
        for example,
        [['train', 'train', 'test', 'test', ...], # outer fold
            ['test', 'train', 'train', 'test', ...],
            ...]
    x: list
        A list of input features for the dataset.(original dataset)
    y: list
        A list of labels for the dataset. (original dataset)
    activity_ids: list
        A list of activity IDs for the dataset. (original dataset)
    smis: list
        A list of SMILES for the dataset. (original dataset)
    spl: str
        The splitting strategy. Options: 'rs-lo', 'rs-vs', 'cs', 'ch'

    returns
    -------
    outer_splits: list of dictionaries
        e.g. [{'outer_train_idx': [2, 3, 4, 5, 6, 7], 'outer_test_idx': [0, 1]},
              {'outer_train_idx': [0, 1, 3, 5, 6, 7], 'outer_test_idx': [2, 4]}, 
              {'outer_train_idx': [0, 1, 2, 3, 4, 7], 'outer_test_idx': [5, 6]}, 
              {'outer_train_idx': [0, 1, 2, 4, 5, 6, 7], 'outer_test_idx': [3]}, 
              {'outer_train_idx': [0, 1, 2, 3, 4, 5, 6], 'outer_test_idx': [7]}]
    outer_n_fold: int
        number of outer folds
    outer_x_train_pick: list
        A list of input features for the picked outer training set.
    outer_y_train_pick: list
        A list of labels for the picked outer training set.
    outer_x_test_pick: list
        A list of input features for the picked outer test set.
    outer_y_test_pick: list
        A list of labels for the picked outer test set.
    inner_splits: list of dictionaries
        e.g. [{'inner_train_idx': [2, 3, 5, 6], 'inner_valid_idx': [4]},
              {'inner_train_idx': [3, 4, 5, 6], 'inner_valid_idx': [2]},
              {'inner_train_idx': [2, 4, 5, 6], 'inner_valid_idx': [3]},
              {'inner_train_idx': [2, 3, 4, 6], 'inner_valid_idx': [5]},
              {'inner_train_idx': [2, 3, 4, 5], 'inner_valid_idx': [6]}]
    inner_splits_all: list of list of dictionaries
    """
    train_minClass_counts = []
    test_minClass_counts = []

    inner_splits_all = []

    outer_train_idx_full = []
    outer_test_idx_full = []
    outer_splits = []

    

    for fold_id, assignment in enumerate(assignments):

        #=========================== Prepare outer train and test idx ===========================
        outer_train_idx = [i for i, v in enumerate(assignment) if v == 'train']
        outer_test_idx = [i for i, v in enumerate(assignment) if v == 'test']

        if len(outer_test_idx) == 0:
            print(f' No test set found in this outer fold {fold_id+1}. Skipping this fold.') if verbose else None
        else:
            # When spl is 'rs-lo' or 'rs-vs':
            # get x, y, activity_ids
            outer_x_train = [x[i] for i in outer_train_idx]
            outer_y_train = [y[i] for i in outer_train_idx]
            outer_actid_train = [activity_ids[i] for i in outer_train_idx]

            outer_x_test = [x[i] for i in outer_test_idx]
            outer_y_test = [y[i] for i in outer_test_idx]
            outer_actid_train = [activity_ids[i] for i in outer_train_idx]

            # When spl is 'cs' or 'ch':
            # get smiles
            outer_smi_train = [smis[i] for i in outer_train_idx]
            outer_smi_test = [smis[i] for i in outer_test_idx]
            
            # ============================ Pick the best outer fold ============================
            train_unique, train_counts = np.unique(outer_y_train, return_counts=True)
            test_unique, test_counts = np.unique(outer_y_test, return_counts=True)

            if len(train_unique) <2 or len(test_unique) <2:
                print(f'Only one class in outer_y_train or outer_y_test in outer fold {fold_id+1}. Cannot do k-fold inner splits, skip this fold.')
            else:
                # ========================== Get the inner splits all ==========================
                try:
                    if spl in ['rs-lo', 'rs-vs']:
                        inner_train_folds, inner_valid_folds = random_split(x=outer_x_train, y=outer_y_train, n_folds=5)
                    elif spl== 'cs':
                        inner_train_folds, inner_valid_folds = cluster_kfold_split(x=outer_smi_train,selectionStrategy='clust_stratified')
                    elif spl== 'ch':
                        inner_train_folds, inner_valid_folds = cluster_kfold_split(x=outer_smi_train,selectionStrategy='clust_holdout')

                    if inner_train_folds is None or inner_valid_folds is None:
                        print(f'Cannot generate inner splits for this outer fold {fold_id+1}.Skipping this fold.') if verbose else None
                    else:
                        
                        # for picking the best outer fold
                        train_minClass_count = min(train_counts)
                        test_minClass_count = min(test_counts)
                        train_minClass_counts.append(train_minClass_count)
                        test_minClass_counts.append(test_minClass_count)

                        # for getting the inner splits all
                        inner_splits = []
                        for tr, va in zip(inner_train_folds, inner_valid_folds):
                            inner_splits.append({
                                'inner_train_idx':[outer_train_idx[i] for i in tr], 
                                'inner_valid_idx':[outer_train_idx[i] for i in va]})
                        
                        inner_splits_all.append(inner_splits)
                        # ====================================Get the outer splits ====================================
                        outer_train_idx_full.append(outer_train_idx)
                        outer_test_idx_full.append(outer_test_idx)
                        outer_splits.append({'outer_train_idx': outer_train_idx,
                                            'outer_test_idx': outer_test_idx})
                        
                except Exception as e:
                    print(f'Error during inner split generation for outer fold {fold_id+1}: {e}')
                continue

    outer_n_fold = len(outer_splits)

    best_index = find_best_index(train_minClass_counts, test_minClass_counts, threshold=2)

    if best_index == -1:
        return outer_splits, outer_n_fold, None, None, None, None, None, None
    else:
        outer_train_idx_pick = outer_train_idx_full[best_index]
        outer_test_idx_pick = outer_test_idx_full[best_index]

        outer_x_train_pick = [x[i] for i in outer_train_idx_pick]
        outer_y_train_pick = [y[i] for i in outer_train_idx_pick]
        outer_x_test_pick = [x[i] for i in outer_test_idx_pick]
        outer_y_test_pick = [y[i] for i in outer_test_idx_pick]

        inner_splits_pick = inner_splits_all[best_index]

        return outer_splits, outer_n_fold,\
                outer_x_train_pick, outer_y_train_pick,\
                outer_x_test_pick, outer_y_test_pick,\
                inner_splits_pick, inner_splits_all

class MLData:

    def __init__(self, fpath:str):
        """
        Initialize the MLData object

        params
        ------
        filepath: str
            Full path to the dataset file.
        """
        #=============== Get the identifiers of the dataset ================
        # based on `ds_path`, e.g. feat_mhd_or
        self.ds_path = fpath.split('/')[-3] # e.g. feat_mhd_or
        self.ds_cat_level = self.ds_path.split('_')[1] # e.g. mhd
        self.ds_type = self.ds_path.split('_')[2] # e.g. or
        # based on `rmvD`, e.g. rmvD0
        self.rmvD = fpath.split('/')[-2]
        # based on `fname`, e.g. CHEMBL233_antag_G-GTP_IC50_None_mhd_b50_b50_featurized.pkl
        self.fname = fpath.split('/')[-1]
        self.fids = self.fname.split('_') # e.g. ['CHEMBL233', 'antag', 'None', 'None', 'None', 'mhd-effect', 'b50', 'b50', 'featurized.pkl']
        self.f_prefix = '_'.join(self.fname.split('_')[:6]) # e.g. CHEMBL233_antag_G-GTP_IC50_None_mhd

        (self.target_chembl_id, 
         self.effect, 
         self.assay, 
         self.standard_type, 
         self.assay_chembl_id) = (self.fids[0], 
                                  self.fids[1], 
                                  self.fids[2], 
                                  self.fids[3], 
                                  self.fids[4])
        
        # ================= Load the dataset ====================
        self.df = pd.read_pickle(fpath) # filepath is the full path to the dataset pickle file

    def get_x_and_y(self, descriptor, aim, spl, **kwargs):
        """
        Get x and y for model training based on the specified parameters.

        params
        ------
        descriptor: str
            The feature descriptor to be used. Options: 'ECFP4', 'ECFP6', 'MACCS', ...
        aim: str
            The application aim of the trained model. Options: 'lo' (lead optimization), 'vs' (virtual screening).
        spl: str
            The splitting strategy. Options: 'rs-lo', 'rs-vs', 'cs', 'ch'

        generated attributes
        -------
        self.x: list
            A list of input features.
        self.y: list
            A list of labels.
        self.activity_ids: list
            A list of activity IDs.

        """
        # ================= Get the data identifiers ====================
        self.descriptor = descriptor
        self.aim = aim
        self.spl = spl

        # ================= Get x and y ====================
        #stereo_idx = self.df[self.df['stereoSiblings']==True].index.tolist() # not necessary to get stereo_idx here, due to the assignments of rmvS1 has already set the stereo siblings to be Nan.
        self.x = self.df[descriptor].tolist()
        self.y = self.df[f'{self.aim}_activity'].tolist()
        self.activity_ids = self.df['activity_id'].tolist() # `activity_id` is unique among each row in the dataset.
        self.smiles = self.df['canonical_smiles_by_Std'].tolist()

        # ================ get stats of x and y ===================
        self.ds_size = len(self.x)
        if self.ds_size > 50:
            self.ds_size_level = 'b50'
        else:
            self.ds_size_level = 's50'
        self.threshold = self.df[f'{self.aim}_threshold'].iloc[0]
        self.percent_a = sum(self.y) / self.ds_size

    def get_int_splits(self, rmvS, verbose=False, **kwargs):
        """
        Generate data splits based on internal splitting columns, which will be used in different ML pipelines later.
        
        params
        ------
        rmvS: int
            wherether the splitting is done with stereochemistry removed. Options: 0, 1.

        returns
        -------
        An example of 'outer_splits':
        [
            {'outer_train_idx': [0, 1, 2, 3, 4, 5], 'outer_test_idx': [6, 7]},
            {'outer_train_idx': [2, 3, 4, 5, 6, 7], 'outer_test_idx': [0, 1]},
            ...
        ]
        """
        print (f'\n------> Run  get_int_splits ') if verbose else None
        self.rmvS = rmvS
        #  internal split columns
        self.int_col_prefix = f'rmvS{self.rmvS}_{self.spl}'
        int_col_names = [col for col in self.df.columns.tolist() if col.startswith(self.int_col_prefix)]
        if len(int_col_names) == 0:
            print(f'No internal split columns found for prefix: {self.int_col_prefix}')
            self.int_outer_n_fold = None
            self.int_outer_x_test_pick = None
        else:
            int_assigns = [self.df[col].to_list() for col in int_col_names]
            # generate splits
            self.int_outer_splits, self.int_outer_n_fold,\
            self.int_outer_x_train_pick, self.int_outer_y_train_pick,\
            self.int_outer_x_test_pick, self.int_outer_y_test_pick,\
            self.int_inner_splits_pick, self.int_inner_splits_all = retrieve_splits(int_assigns, self.x, self.y, self.activity_ids, self.smiles,
                                                                               self.spl, verbose=verbose)

    def get_aln_splits(self, pf_prefix:str=None, cf_prefix:str=None, verbose=False, **kwargs):
        """
        Generate data splits based on aligned splitting columns, which will be used in different ML pipelines later.

        params
        ------
        pf_prefix: str. e.g. CHEMBL233_None_None_Ki_None_hhd
            if pf_prefix is provided, it indicate the self.f_prefix is for child dataset.
            to get the parent aligned split columns, we need to use pf_prefix.
        cf_prefix: str. e.g. CHEMBL233_bind_RBA_Ki_None_mhd
            if cf_prefix is provided, it indicate the self.f_prefix is for parent dataset.
            to get the child aligned split columns, we need to use cf_prefix.

        returns
        -------
        if pf_prefix is provided:
            cf-related aligned split columns will be retrieved.
        if cf_prefix is provided:
            pf-related aligned split columns will be retrieved.
        """
        print (f'\n------> Run  get_aln_splits ...') if verbose else None
        # aligned split columns
        # if spl starts with 'parent', 
        if cf_prefix:
            pf_aln_col_prefix = f'parent.{self.f_prefix}.{cf_prefix}.{self.int_col_prefix}'
            pf_aln_col_names = [col for col in self.df.columns.tolist() if col.startswith(pf_aln_col_prefix)]
            if len(pf_aln_col_names) == 0:
                print(f'No aligned split columns found for prefix: {pf_aln_col_prefix}')
                self.pf_aln_outer_n_fold = None
                self.pf_aln_outer_x_test_pick = None
            else:
                pf_aln_assigns = [self.df[col].to_list() for col in pf_aln_col_names]
                # ================== retrieve split data for pf_aln_assigns ====================
                self.pf_aln_outer_splits, self.pf_aln_outer_n_fold,\
                self.pf_aln_outer_x_train_pick, self.pf_aln_outer_y_train_pick,\
                self.pf_aln_outer_x_test_pick, self.pf_aln_outer_y_test_pick,\
                self.pf_aln_inner_splits_pick, self.pf_aln_inner_splits_all = retrieve_splits(pf_aln_assigns, self.x, self.y, self.activity_ids, self.smiles,
                                                                                         self.spl, verbose=verbose)
        
        if pf_prefix:
            # if spl starts with 'child', the training set should be same as that from the corresponding columns with 'int_col_prefix', just the test set is different.
            cf_aln_col_prefix = f'child.{pf_prefix}.{self.f_prefix}.{self.int_col_prefix}'
            cf_aln_col_names = [col for col in self.df.columns.tolist() if col.startswith(cf_aln_col_prefix)]
            if len(cf_aln_col_names) == 0:
                print(f'No aligned split columns found for prefix: {cf_aln_col_prefix}')
                self.cf_aln_outer_n_fold = None
                self.cf_aln_outer_x_test_pick = None
            else:
                cf_aln_assigns = [self.df[col].to_list() for col in cf_aln_col_names]
                # ================== retrieve split data for cf_aln_assigns ====================
                self.cf_aln_outer_splits, self.cf_aln_outer_n_fold,\
                self.cf_aln_outer_x_train_pick, self.cf_aln_outer_y_train_pick,\
                self.cf_aln_outer_x_test_pick, self.cf_aln_outer_y_test_pick,\
                self.cf_aln_inner_splits_pick, self.cf_aln_inner_splits_all = retrieve_splits(cf_aln_assigns, self.x, self.y, self.activity_ids, self.smiles,
                                                                                         self.spl, verbose=verbose)

    def featurize_data(self):
        # SMILES to descriptors
        pass

    def balance_data(self):
        # undersampling
        # oversampling
        pass

    def augment_data(self):
        # non-canonical SMILES
        # DeepCoy negatives
        pass

    def shuffle_data(self):
        # Shuffle the data
        pass

    def __call__(self, descriptor, aim, spl, **kwargs):
        self.get_x_and_y(descriptor, aim, spl, **kwargs)

    def __repr__(self):
        return f"MLData Object: {self.fname} with {self.df.shape[0]} samples."
