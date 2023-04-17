import sys

import numpy as np
import pandas as pd


class MlflowModel:
    def __init__(self):
        pass
    def make_model_sklearn(self, train, test, n):
        model = self._get_model(n)
        model.fit(train,test)
        model_info = { 'score':model.score(train,test), 'params':model.get_params()}
        
        return model, model_info
    
    def make_model_pytorch(self, train, test, n):
        pass
    