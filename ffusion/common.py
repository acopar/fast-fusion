#!/usr/bin/env python

import os
import sys
import csv
import time
import argparse
import numpy as np
import scipy.linalg as la
import threading

from scipy.sparse import csr_matrix, csc_matrix
from ffusion.loader import *
from ffusion.worker import *
from ffusion.zengine import Zengine
from mpi4py import MPI

def nprand(x, y, dtype=np.float64, seed=42):
    X = np.random.rand(x, y)
    X = np.array(X, dtype=dtype, order='C')
    return X


def dump_history(params, err_history):
    k = params['k'][0]
    filename = 'results/%s/%s/%d_%d.csv' % (params['label'], params['technique'],k, params['seed'])
    ensure_dir(filename)
    fp = open(filename, 'w')
    writer = csv.writer(fp, delimiter=',')
    for i, h in enumerate(err_history):
        writer.writerow([params['method'], params['technique'], i, h, k, params['seed']])
    fp.close()

def validate_factors(factors):
    for f in factors:
        for idx in f:
            if np.any(f[idx] < 0):
                print("Assert exception: factor contains negative values")


def transform(func):
    def new_f(params):
        X = params['X']
        G = params['G']
        S = params['S']
        target = params['target']
        seed = params['seed']
        k = params['k']
        max_iter = params['max_iter']
        verbose = params['verbose']
        method = params['method']
        engine = params['engine']
        technique = params['technique']
        
        print("Task started: (%s, %s)" % (method, technique))
        np.random.seed(seed)
        dtype = None
        # calculate data sizes
        n = {}
        for i, j in X:
            if dtype == None:
                dtype = X[i,j].dtype
            if i not in n:
                n[i] = X[i,j].shape[0]
            else:
                if n[i] != X[i,j].shape[0]:
                    raise Exception("Datasets not aligned at block: (%d, %d)" % (i,j))
            if j not in n:
                n[j] = X[i,j].shape[1]
            else:
                if n[j] != X[i,j].shape[1]:
                    raise Exception("Datasets not aligned at block: (%d, %d)" % (i,j))
        
        for i, j in X:
            sparse = False
            if type(X[i,j]) == csr_matrix or type(X[i,j]) == csc_matrix:
                sparse = True
        Gi = {}
        Gi[target] = nprand(n[target], k[target], dtype=dtype)
            #print("G shape", i, G[i].shape)
        
        TrX = {}
        Xt = {}
        mask = {}
        for i, j in X:
            if type(X[i,j]) == csr_matrix or type(X[i,j]) == csc_matrix:
                # X.power(2)
                TrX[i,j] = np.sum(X[i,j].multiply(X[i,j]))
                Xt[i,j] = csr_matrix(X[i,j].T)
            else:
                TrX[i,j] = np.sum(np.multiply(X[i,j], X[i,j]))
                Xt[i,j] = np.array(X[i,j].T, order='C')
            
            if TrX[i,j] == 0:
                mask[i,j] = False
            else:
                mask[i,j] = True
        
        t0 = time.time()
        engine.clean()
        
        factors = func(engine, X, Xt, mask, G, S, Gi, TrX, k=k, max_iter=max_iter, verbose=verbose)
        err_history = []
        
        return factors, err_history
    return new_f


class Timer():
    def __init__(self, system=True):
        self.t = {}
        self.c = {}
        self.last = None
        self.system = system
    
    def time(self):
        if self.system == False:
            return time.time()
        else:
            return os.times()[4]
    
    def get(self, label=None):
        if label not in self.t:
            return None
        else:
            return self.t[label]
    
    def check(self, label=None):
        if label not in self.t or self.t[label] == None:
            self.t[label] = 0.0
    
    def labelize(self, label):
        if label == None:
            if self.last:
                label = self.last
        return label
    
    def clear(self):
        self.t = {}
        self.c = {}
        self.last = None
    
    def reset(self, label=None):
        label = self.labelize(label)
        self.t[label] = 0.0
        self.c[label] = self.time()
    
    def start(self, label=None):
        label = self.labelize(label)
        if label == None:
            return
        self.check(label=label)
        self.c[label] = self.time()
        self.last = label
    
    def pause(self, label=None):
        label = self.labelize(label)
        if label == None:
            return
        if label in self.c or self.c[label] != None:
            self.t[label] += self.time() - self.c[label]
    
    def stop(self, label=None):
        self.pause(label=label)
        if self.last:
            self.last = None
    
    def split(self, label=None):
        label = self.labelize(label)
        if label == None or label == self.last:
            return
        if self.last:
            self.stop(label=self.last)
        self.start(label=label)
        
    def __str__(self):
        elements = sorted(self.t.items(), key=lambda x: x[1], reverse=True)
        total = sum([t[1] for t in elements])
        portions = [(key, 100*value/total) for key, value in elements]
        portions = [(key, '%.2f' % value) for key, value in portions]
        return str(portions)

    def add(self, other):
        for key in other.t:
            if key in self.t:
                self.t[key] = self.t[key] + other.t[key]
            else:
                self.t[key] = other.t[key]
    
    def asdict(self):
        elements = sorted(self.t.items(), key=lambda x: x[1], reverse=True)
        total = sum([t[1] for t in elements])
        portions = [(key, value/total) for key, value in elements]
        portions = {key: value for key, value in portions}
        return portions

    def elapsed(self):
        return self.t
    
    def total_elapsed(self):
        elements = sorted(self.t.items(), key=lambda x: x[1], reverse=True)
        total = sum([t[1] for t in elements])
        return total

def fusion(func):
    #def new_f(X, k=20, seed=42, max_iter=10, verbose=False):
    def new_f(params):
        X = params['X']
        seed = params['seed']
        k = params['k']
        max_iter = params['max_iter']
        verbose = params['verbose']
        method = params['method']
        engine = params['engine']
        technique = params['technique']
        
        np.random.seed(0)
        dtype = None#np.float32
        # calculate data sizes
        
        comm = MPI.COMM_WORLD
        rank = comm.rank
        n_proc = params['n_proc']

        if rank == 0:
            print("Task started: (%s, %s)" % (method, technique))
        

        n = {}
        for i, j in X:
            #X[i,j] = X[i,j].astype(dtype)
            if dtype == None:
                dtype = X[i,j].dtype
            if i not in n:
                n[i] = X[i,j].shape[0]
            else:
                if n[i] != X[i,j].shape[0]:
                    print("Not aligned", n[i], X[i,j].shape[0])
                    raise Exception("Datasets not aligned at block: (%d, %d)" % (i,j))
            if j not in n:
                n[j] = X[i,j].shape[1]
            else:
                if n[j] != X[i,j].shape[1]:
                    print("Not aligned", n[j], X[i,j].shape[1])
                    raise Exception("Datasets not aligned at block: (%d, %d)" % (i,j))
        
        G = {}
        for i in range(len(k)):
            np.random.seed(42)
            G[i] = nprand(n[i], k[i], dtype=dtype)
            #print("G shape", i, G[i].shape)
        
        TrX = {}
        mask = {}
        
        for i, j in X:
            if type(X[i,j]) == csr_matrix or type(X[i,j]) == csc_matrix:
                # X.power(2)
                TrX[i,j] = np.sum(X[i,j].multiply(X[i,j]))
            else:
                TrX[i,j] = np.sum(np.multiply(X[i,j], X[i,j]))
            
            if TrX[i,j] == 0:
                mask[i,j] = False
            else:
                mask[i,j] = True

        
        if n_proc != 1:
            # slice
            for i, j in X:
                a = X[i,j].shape[0]*(rank)//n_proc
                b = X[i,j].shape[0]*(rank+1)//n_proc
                X[i,j] = X[i,j][a:b,:]
            
        
        for i, j in X:
            sparse = False
            if type(X[i,j]) == csr_matrix or type(X[i,j]) == csc_matrix:
                sparse = True

        
        Xt = {}
        for i, j in X:
            if type(X[i,j]) == csr_matrix or type(X[i,j]) == csc_matrix:
                # X.power(2)
                Xt[i,j] = csr_matrix(X[i,j].T)
            else:
                Xt[i,j] = np.array(X[i,j].T, order='C')
            
        
        m = {}
        Gi = {}
        S = {}
        slices = {}
        
        if n_proc == 1:
            Gi = G
        else:
            for i in range(len(k)):
                np.random.seed(42)
                a = G[i].shape[0]*(rank)//n_proc
                b = G[i].shape[0]*(rank+1)//n_proc
                Gi[i] = G[i][a:b,:]
            
        
            for i in range(len(k)):
                m[i] = {}
                slices[i] = {}
                for d in range(n_proc):
                    a = G[i].shape[0]*(d)//n_proc
                    b = G[i].shape[0]*(d+1)//n_proc
                    m[i][d] = b-a
                    slices[i][d] = (a,b)
            
        for i, j in X:
            np.random.seed(42)
            S[i,j] = nprand(k[i], k[j], dtype=dtype)
            #print("S", i, j, S[i,j].sum(), S[i,j].shape)
        
        
        t0 = time.time()
        if engine.profile:
            engine.clear_profile()
        
        engine.dtype = dtype
        
        """
        z = Zengine()
        func(z, X, Xt, mask, G, S, TrX, k=k, max_iter=1)
        print(z._locals.keys())
        for key in z._locals:
            if type(z._locals[key]) == type({}):
                print("adding local", key)
                engine._locals[key] = z._locals[key]
        """
        dimensions = {'n': n, 'm': m, 'k': k, 'slices': slices}
        factors, err_history = func(engine, X, Xt, mask, (G, Gi), S, TrX, dimensions, k=k, max_iter=max_iter, verbose=verbose)
        
        t1 = time.time()
        
        if rank == 0:
            print("Task (%s) finished in:", (technique, t1-t0))
        #print("Error", err_history[-1])
        if engine.profile:
            print("Engine Mflops/iteration:", float(engine.operations/max_iter)/1000000, float(engine.soperations/max_iter)/1000000)
            print("Engine timer:", str(engine.timer))
        #validate_factors(factors)
        
        if params['store_results']:
            dump_history(params, err_history[1:])
            #filename = 'factors/%s/%s/%d_%d.pkl' % (params['label'], params['technique'], params['k'], params['seed'])
            #dump_file(filename, factors)
            
            #filename = 'time-d5/%s/%s/%d_%d.pkl' % (params['label'], params['technique'], params['k'], params['seed'])
            #dump_file(filename, {'time': t1-t0, 'iterations': max_iter})
        
        return factors, err_history
    return new_f

