# conda env: pyg(Python 3.9.16)

"""
class MDLDeploy:

a class to processing data for modelling, including:
    - loading the dataset
    - featurizing the dataset
    - balancing the dataset
    - shuffling the dataset
"""

import os
import sys
import re

import pandas as pd
import random
import re
from yaml import load, Loader, dump
import json
from tqdm import tqdm
from typing import List, Union
import warnings
import numpy as np
from imblearn.over_sampling import SMOTE


from datacat4ml.const import *
from datacat4ml.Scripts.data_prep.data_featurize.feat_list_smi import Featurizer

class Data:
    
    def __init__(self, file_folder: str, filename: str, task: str, use_smote: bool = False):
        """
        Initialize the Data class.
        
        :param file_folder: (str) Path to the folder containing the data file.
        :param filename: (str) Name of the data file in CSV format.
        :param task: (str) Task type. Either 'reg' for regression or 'cls' for classification.
        :param use_smote: (bool) Whether to use SMOTE for data balancing in the classification task.
        """
        df = pd.read_csv(os.path.join(file_folder, filename))

        self.task = task

        # The training dataset
        self.smiles_train = df[df['split'] == 'train']['canonical_smiles_by_Std'].tolist()
        if self.task == 'reg':
            self.y_train = df[df['split'] == 'train']['pStandard_value'].tolist()
            self.cliff_mols_train = df[df['split'] == 'train']['cliff_mol'].tolist()
        elif self.task == 'cls':
            self.y_train = df[df['split'] == 'train']['activity'].tolist()
        

        # The testing dataset
        self.smiles_test = df[df['split'] == 'test']['canonical_smiles_by_Std'].tolist()
        if self.task == 'reg':
            self.y_test = df[df['split'] == 'test']['pStandard_value'].tolist()
            self.cliff_mols_test = df[df['split'] == 'test']['cliff_mol'].tolist()
        elif self.task == 'cls':
            self.y_test = df[df['split'] == 'test']['activity'].tolist()
        
        self.featurizer = Featurizer()

        self.x_train = None
        self.x_test = None
        
        # values that specify the origin of dataset
        self.target = df['target'].tolist()[0]
        self.effect = df['effect'].tolist()[0]
        self.assay = df['assay'].tolist()[0]
        self.std_type = df['std_type'].tolist()[0]

        self.use_smote = use_smote
        self.x_smote_train = None
        self.y_smote_train = None

        self.featurized_as = 'Nothing'
    
    def featurize_data(self, descriptor:str, **kwargs):
        """
        Featurize data using the provided descriptor

        :param descriptor: (str) descriptor name, the value stored in the const dictionary Descriptors
        :param kwargs: (dict) additional arguments for the descriptor
        """

        self.x_train = self.featurizer(descriptor=descriptor, smi= self.smiles_train, **kwargs)
        self.x_test = self.featurizer(descriptor=descriptor, smi= self.smiles_test, **kwargs)
        self.featurized_as = descriptor

    def balance_data(self):
        """
        Balance the training data only for classification task

        :param method: (str) balancing method name, the value stored in the const dictionary BalancingMethods
        :param kwargs: (dict) additional arguments for the balancing method
        """

        n_neighbors = min(6, len(self.y_train))
        
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=n_neighbors)
        self.x_smote_train, self.y_smote_train = smote.fit_resample(self.x_train, self.y_train)

    def shuffle(self):
        """
        Shuffle the training data
        """
        if self.task == 'reg':
            c = list(zip(self.x_train, self.y_train, self.cliff_mols_train)) # `zip` is used to combine three lists into a list of tuples
            random.shuffle(c)
            self.x_train, self.y_train, self.cliff_mols_train = zip(*c) # `zip(*c)` returns a list of tuples, which are unzipped using `*` and assigned to different variables

            self.x_train = list(self.x_train)
            self.y_train = list(self.y_train)
            self.cliff_mols_train = list(self.cliff_mols_train)
        
        elif self.task == 'cls':
            if self.use_smote:
                c = list(zip(self.x_smote_train, self.y_smote_train))
            else:
                c = list(zip(self.x_train, self.y_train))

            random.shuffle(c)
            x_data, y_data = zip(*c)

            if self.use_smote:
                self.x_smote_train = list(x_data)
                self.y_smote_train = list(y_data)
            else:
                self.x_train = list(x_data)
                self.y_train = list(y_data)

    def __call__(self, descriptor:str, **kwargs):
        self.featurize_data(descriptor, **kwargs)
    
    def __repr__(self):
        return f"MLdeployer(featurized_as={self.featurized_as}, use_smote={self.use_smote})"