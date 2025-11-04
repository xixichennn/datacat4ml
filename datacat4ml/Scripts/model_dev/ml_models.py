"""
A collection of classical machine learning algorithms for classifcation task

    -MLmodel:       parent class used by all machine learning algorithms
        -RF:           Random Forest Classifier
        -GB:           Gradient Boosting Classifier
        -SVM:           Support Vector Classifier
        -KNN:          K-Nearest neighbour Classifier
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from numpy.typing import ArrayLike
from typing import List, Union, Dict

class MLmodel:
    def __init__(self):
        self.model = None
        self.epoch = None

    def train(self, x_train: ArrayLike, y_train: Union[List[float], ArrayLike], *args, **kwargs):
        if type(y_train) is list:
            y_train = np.array(y_train)
        self.model.fit(x_train, y_train)

    def predict(self, x: ArrayLike, *args, **kwargs):
        return self.model.predict(x)
    
    def predict_proba(self, x: ArrayLike, *args, **kwargs):
        return self.model.predict_proba(x)


#===============  Classification or regression algorithms ====================
class RF(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = RandomForestClassifier(**hyperparameters)
        self.name = 'RF'

    def __repr__(self):
        return f"{self.name}: {self.model.get_params()}"

class GB(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = GradientBoostingClassifier(**hyperparameters)
        self.name = 'GB'

    def __repr__(self):
        return f"{self.name}: {self.model.get_params()}"

class SVM(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = SVC(**hyperparameters)
        self.name = 'SVM'

    def __repr__(self):
        return f"{self.name}: {self.model.get_params()}"

class KNN(MLmodel):
    def __init__(self, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__()
        self.model = KNeighborsClassifier(**hyperparameters)
        self.name = 'KNN'

    def __repr__(self):
        return f"{self.name}: {self.model.get_params()}"
  