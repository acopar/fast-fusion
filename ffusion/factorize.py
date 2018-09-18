#!/usr/bin/env python

import os
import sys
import csv
import time
import argparse
import numpy as np
import scipy.linalg as la

from scipy.sparse import csr_matrix, csc_matrix
from ffusion.loader import *
from ffusion.worker import *
from ffusion.common import *
from ffusion.engine import Engine
from ffusion.proengine import EngineProfiler

from ffusion.zitnik import df_zitnik
from ffusion.dfcod import df_cod
def normalize_data(X):
    return X / np.max(X)

def pprint(X):
    for i in range(X.shape[0]):
        s = ''
        for j in range(X.shape[1]):
            s += "%.1f" % X[i,j] + ' '
        print(s)

def main():
    parser = argparse.ArgumentParser(description='fast-fusion')
    parser.add_argument('-i', '--iterations', type=int, default=10, help="Maximum number of iterations.")
    parser.add_argument('-t', '--technique', default='', help="Optimization technique (mu, cod, als, pg)")
    parser.add_argument('-k', '--k', default='20', help="Factorization rank")
    parser.add_argument('-p', '--parallel', type=int, default=1, help="Number of workers")
    parser.add_argument('-S', '--seed', default='0', help="Random seed")
    
    parser.add_argument('-V', '--verbose', action="store_true", help="Print error function in each iteration")
    parser.add_argument('data', nargs='*', help='Other args')
    
    args = parser.parse_args()
    
    folder = args.data[0]
    #data = load_data(filename)
    X = load_many(folder)

    
    basedata = os.path.basename(os.path.dirname(folder))
    
    # double
    for i,j in X:
        X[i,j] = X[i,j].astype(np.float64)
        print("Shape %d,%d" % (i,j), X[i,j].shape)
    
    
    #np.random.seed(args.seed)
    max_iter = args.iterations
    
    function_dict = {
        'zitnik': df_zitnik,
        'cod': df_cod,
    }
    
    method_list = ['fusion']
    technique_list = args.technique.split(',')
    if args.technique == '':
        technique_list = ['zitnik', 'cod']
        
    k_list = [int(s) for s in args.k.split(',')]
    seed_list = [int(s) for s in args.seed.split(',')]
    
    # count object types
    imax = 0
    jmax = 0
    for i,j in X:
        if i > imax:
            imax = i
        if j > jmax:
            jmax = j
    nobjects = max(imax, jmax) +1
    if len(k_list) != nobjects:
        print("Number of objects != len(k): %d != %d" % (nobjects, len(k_list)))
        raise Exception("Factorization rank and number of objects must be equal")
    
    engine = EngineProfiler()
    #engine = Engine()
    tasks = []
    conv_trace = {}
    for t in technique_list:
        if t not in function_dict:
            print("Technique %s is not available" % t)
            continue
        
        conv_trace[t] = []
        for seed in seed_list:
            params = {'engine': engine, 'X': X, 'k': k_list, 'seed': seed, 'method': 'fusion', 'technique': t, 
                'max_iter': max_iter, 'verbose': args.verbose, 'store_results': True, 'basename': basedata, 
                'label': "%s-p6-10" % basedata}
#                factors, hist = function_dict[t](params)
            tasks.append(Task(function_dict[t], params))
    
    model = Model()
    mt = MainThread(model, n_workers=args.parallel)
    mt.start()
    for task in tasks:
        model.q.put(task)
    model.q.put(0)
    
    try:
        while mt.isAlive():
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        mt.stop()
    
    mt.join()


if __name__ == '__main__':
    main()
