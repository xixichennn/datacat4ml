
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
    
    if len(A) == 0 or len(B) == 0:
        print(f'Empty train_minClass_counts or test_minClass_counts list found.\n Skip model training for this dataset.')
        best_index = -1
    else:
        best_index = None
        max_sum = float('-inf')
        for i, (a_val, b_val) in enumerate(zip(A, B)):
            if a_val >= threshold and b_val >= threshold:
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
    print(f'\nGenerating outer splits ...') if verbose else None
    
    outer_train_idx_full = []
    outer_test_idx_full = []
    outer_splits = []

    train_minClass_counts = []
    test_minClass_counts = []

    # ========================== Generate outer splits ==========================
    for assignment in assignments:
        outer_train_idx = [i for i, v in enumerate(assignment) if v == 'train']
        outer_test_idx = [i for i, v in enumerate(assignment) if v == 'test']

        outer_train_idx_full.append(outer_train_idx)
        outer_test_idx_full.append(outer_test_idx)
        outer_splits.append({'outer_train_idx': outer_train_idx, 
                             'outer_test_idx': outer_test_idx})

        # ============================================================================
        # Select one outer_fold which has the maximum min-class count in outer_y_train,
        # thus to ensure the inner splits can be generated as many folds as possible.
        # ============================================================================
        outer_y_train = [y[i] for i in outer_train_idx]
        #print(f'outer_y_train: {outer_y_train}')
        outer_y_test = [y[i] for i in outer_test_idx]
        #print(f'outer_y_test: {outer_y_test}')

        train_unique, train_counts = np.unique(outer_y_train, return_counts=True)
        test_unique, test_counts = np.unique(outer_y_test, return_counts=True)
        #print(f'classes: {unique}, counts: {counts}')
        if len(train_unique) >= 2:
            train_minClass_count = min(train_counts)
            #print(f'minClass_count: {minClass_count}')
            train_minClass_counts.append(train_minClass_count)
        else:
            print('Only one class in outer_y_train, skip this fold.')
            train_minClass_counts.append(0)

        if len(test_unique) >= 2:
            test_minClass_count = min(test_counts)
            #print(f'minClass_count: {minClass_count}')
            test_minClass_counts.append(test_minClass_count)
        else:
            test_minClass_counts.append(0)
            print('Only one class in outer_y_test, skip this fold.')

    print(f'train_minClass_counts: \n{train_minClass_counts}') if verbose else None
    print(f'test_minClass_counts: \n{test_minClass_counts}') if verbose else None

    # length of the outer splits
    outer_n_fold = len(outer_splits)

    best_index = find_best_index(train_minClass_counts, test_minClass_counts, threshold=2)

    if best_index == -1:
        return outer_splits, outer_n_fold, None, None, None, None, None, None
    else:
        # the idx of the outer fold picked
        outer_train_idx_pick = outer_train_idx_full[best_index]
        outer_test_idx_pick = outer_test_idx_full[best_index]

        # when spl is 'rs-lo' or 'rs-vs': 
        # get the picked outer train and test sets of x, y, activity_ids
        outer_x_train_pick = [x[i] for i in outer_train_idx_pick]
        outer_y_train_pick = [y[i] for i in outer_train_idx_pick]
        outer_actid_train_pick = [activity_ids[i] for i in outer_train_idx_pick]

        outer_x_test_pick = [x[i] for i in outer_test_idx_pick]
        outer_y_test_pick = [y[i] for i in outer_test_idx_pick]
        outer_actid_test_pick = [activity_ids[i] for i in outer_test_idx_pick]

        # when spl is 'cs' or 'ch':
        # get the picked outer train and test sets of smiles
        outer_smi_train_pick = [smis[i] for i in outer_train_idx_pick]
        outer_smi_test_pick = [smis[i] for i in outer_test_idx_pick]

        # ================== Get the inner splits ====================
        print(f'\nGenerating inner splits for ...') if verbose else None
        inner_splits = []
        try:
            if spl in ['rs-lo', 'rs-vs']:
                # generate the inner splits
                inner_train_folds, inner_valid_folds = random_split(x=outer_x_train_pick, y=outer_y_train_pick, n_folds=5)
            
            elif spl== 'cs':
                # generate the inner splits
                inner_train_folds, inner_valid_folds = cluster_kfold_split(x=outer_smi_train_pick, selectionStrategy='clust_stratified')
            
            elif spl== 'ch':
                inner_train_folds, inner_valid_folds = cluster_kfold_split(x=outer_smi_train_pick, selectionStrategy='clust_holdout')
            #print(f'Inner train folds: {inner_train_folds}')
            #print(f'Inner valid folds: {inner_valid_folds}\n')

            for tr, va in zip(inner_train_folds, inner_valid_folds):
                inner_splits.append({
                    'inner_train_idx':[outer_train_idx_pick[i] for i in tr], # map back to the original dataset indices
                    'inner_valid_idx':[outer_train_idx_pick[i] for i in va]})

        except Exception as e:
            print(f'Error during inner split generation: {e}')

        # ================= Get the inner splits all ====================
        print(f'\nGenerating all inner splits ...') if verbose else None
        inner_splits_all = []
        for outer_train_idx in outer_train_idx_full:
            try:
                if spl in ['rs-lo', 'rs-vs']:
                    inner_train_folds, inner_valid_folds = random_split(x=[x[i] for i in outer_train_idx],
                                                                        y=[y[i] for i in outer_train_idx],
                                                                        n_folds=5)
                elif spl== 'cs':
                    inner_train_folds, inner_valid_folds = cluster_kfold_split(x=[smis[i] for i in outer_train_idx],
                                                                            selectionStrategy='clust_stratified')
                elif spl== 'ch':
                    inner_train_folds, inner_valid_folds = cluster_kfold_split(x=[smis[i] for i in outer_train_idx],
                                                                            selectionStrategy='clust_holdout')
            except Exception as e:
                print(f'Error during inner split generation for outer_train_idx {outer_train_idx}: {e}')
                continue

            inner_splits = []
            for tr, va in zip(inner_train_folds, inner_valid_folds):
                inner_splits.append({
                    'inner_train_idx':[outer_train_idx[i] for i in tr], 
                    'inner_valid_idx':[outer_train_idx[i] for i in va]})

            inner_splits_all.append(inner_splits)

        return outer_splits, outer_n_fold,\
            outer_x_train_pick, outer_y_train_pick,\
            outer_x_test_pick, outer_y_test_pick,\
            inner_splits, inner_splits_all
        


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
        print(f'ds_path: {self.ds_path}')
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
            self.int_inner_splits, self.int_inner_splits_all = retrieve_splits(int_assigns, self.x, self.y, self.activity_ids, self.smiles,
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
                self.pf_aln_inner_splits, self.pf_aln_inner_splits_all = retrieve_splits(pf_aln_assigns, self.x, self.y, self.activity_ids, self.smiles,
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
                self.cf_aln_inner_splits, self.cf_aln_inner_splits_all = retrieve_splits(cf_aln_assigns, self.x, self.y, self.activity_ids, self.smiles,
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
