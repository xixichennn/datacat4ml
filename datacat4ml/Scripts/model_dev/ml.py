"""
A collection of classical machine learning algorithms

    -MLmodel:       parent class used by all machine learning algorithms
    -regression:
        -RFR:            Random Forest Regressor
        -GBR:           Gradient Boosting Regressor
        -SVRR:           Support Vector Regressor
        -KNNR:           K-Nearest neighbour Regressor

    -classification:
        -RFC:           Random Forest Classifier
        -GBC:           Gradient Boosting Classifier
        -SVCC:           Support Vector Classifier
        -KNNC:          K-Nearest neighbour Classifier
"""

import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from numpy.typing import ArrayLike
from typing import List, Union, Dict

class MLmodel:
    def __init__(self, task: str):
        self.model = None
        self.epoch = None
        self.task = task

    def predict(self, x: ArrayLike, *args, **kwargs):
        return self.model.predict(x)
    
    def predict_proba(self, x: ArrayLike, *args, **kwargs):
        return self.model.predict_proba(x)

    def train(self, x_train: ArrayLike, y_train: Union[List[float], ArrayLike], *args, **kwargs):
        if type(y_train) is list:
            y_train = np.array(y_train)
        self.model.fit(x_train, y_train)

    def test(self, x_test: ArrayLike, y_test: Union[List[float], ArrayLike], *args, **kwargs):
        if type(y_test) is list:
            y_test = np.array(y_test)
        y_hat = self.model.predict(x_test)

        return y_hat, y_test

    def __call__(self, x: ArrayLike):
        return self.model.predict(x)

#===============  Classification algorithms ====================
class RFC(MLmodel):
    def __init__(self, task:str, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = RandomForestClassifier(**hyperparameters)
        self.name = 'RFC'

    def __repr__(self):
        return f"Random Forest Classifier: {self.model.get_params()}"

class GBC(MLmodel):
    def __init__(self, task:str, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = GradientBoostingClassifier(**hyperparameters)
        self.name = 'GBC'

    def __repr__(self):
        return f"Gradient Boosting Classifier: {self.model.get_params()}"

class SVCC(MLmodel):
    def __init__(self, task:str, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = SVC(**hyperparameters)
        self.name = 'SVCC'

    def __repr__(self):
        return f"Support Vector Classifier: {self.model.get_params()}"

class KNNC(MLmodel):
    def __init__(self, task:str, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = KNeighborsClassifier(**hyperparameters)
        self.name = 'KNNC'

    def __repr__(self):
        return f"K-Nearest Neighbour Classifier: {self.model.get_params()}"



#===============  Regression algorithms ====================
class RFR(MLmodel):
    def __init__(self, task:str,  **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = RandomForestRegressor(**hyperparameters)
        self.name = 'RFR'

    def __repr__(self):
        return f"Random Forest Regressor: {self.model.get_params()}"


class GBR(MLmodel):
    def __init__(self, task:str, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = GradientBoostingRegressor(**hyperparameters)
        self.name = 'GBR'

    def __repr__(self):
        return f"Gradient Boosting Regressor: {self.model.get_params()}"


class SVRR(MLmodel):
    def __init__(self, task:str, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = SVR(**hyperparameters)
        self.name = 'SVRR'

    def __repr__(self):
        return f"Support Vector Regressor: {self.model.get_params()}"


class KNNR(MLmodel):
    def __init__(self, task:str, **hyperparameters: Dict[str, Union[str, float, int]]):
        super().__init__(task)
        self.model = KNeighborsRegressor(**hyperparameters)
        self.name = 'KNNR'

    def __repr__(self):
        return f"K-Nearest Neighbour regressor: {self.model.get_params()}"
  