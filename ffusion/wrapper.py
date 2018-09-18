#!/usr/bin/env python

from ffusion.common import Timer
from ffusion.engine import Engine

from ffusion.zitnik import df_zitnik, df_transform
from ffusion.dfcod import df_cod
import numpy as np

class Fusion():
    def __init__(self, X, k_list, technique='zitnik', max_iter=100, basedata='name', verbose=False):
        self.X = X
        self.k_list = k_list
        self.technique = technique
        self.tasks = []
        func = None
        
        if technique == 'zitnik':
            func = df_zitnik
        if technique == 'cod':
            func = df_cod
        self.func = func
        self.params = {'engine': Engine(), 'X': X, 'k': k_list, 'seed': 0, 'method': 'fusion', 'technique': technique, 
                'max_iter': max_iter, 'verbose': verbose, 'store_results': False, 'basename': basedata, 
                'label': "%s-p6-10" % basedata}
        self.timer = Timer()
        
    def fit(self):
        self.timer
        factors, err_history = self.func(self.params)
        self.factors = factors
    
    def predict(self, X_test, target):
        params2 = {}
        for key, value in self.params.items():
            params2[key] = value
        params2['G'] = self.factors[0]
        params2['S'] = self.factors[1]
        params2['X'] = X_test
        params2['target'] = target
        params2['max_iter'] = 10
        Gi, history = df_transform(params2)
        
        #G, S = self.factors
        #key = list(X_test.keys())[0]
        #gsx = np.dot(np.dot(Gi, S[key]), G[key[1]].T)
        #X1 = np.subtract(X_test[key], gsx)
        #E = np.sum(np.multiply(X1, X1))
        #print("Transform error:", E)
        return Gi