#!/usr/bin/env python

import os
import csv
import subprocess
import argparse

from ffusion.loader import *
import numpy as np
from ffusion.stop import score_history, score_history2


IMG_DIR = 'img'


def data_from_csv(filename):
    fp = open(filename, 'r')
    reader = csv.reader(fp, delimiter=',')
    data = []
    for line in reader:
        it = int(line[2])
        value = float(line[3])
        data.append(value)#[it, value])
        
    return data



def main2(k=20):
    titles = {"dicty": "Dicty", "movies": "Movies"}
    max_iter = 100000
    
    technique_map = {'ZIT': 'zitnik', 'COD': 'cod'}
    
    frames = {}
    lines = {}
    best_seed = {}
    best_lines = {}
    k_dict = {'dicty': 50, 'movies': 10}
    
    datasets = ['dicty', 'movies']
    for dataset in datasets:
        k = k_dict[dataset]
        fname = titles[dataset]
        frames[fname] = {}
        lines = {}
        best_seed = {}
        #print dataset
        for technique in ['ZIT', 'COD']:
            #if technique == 'ALS' and dataset == 'coil20':
            #    continue
            technique_name = technique_map[technique]
            
            datapoint = {}
            best = 100.0
            best_seed[technique] = None
            lines[technique] = {}
            score_list = []
            score_list2 = []
            for seed in [0]:
                filename = "results/%s-p6-10/%s/%d_%d.csv" % (dataset, technique_name, k, seed)
                if not os.path.isfile(filename):
                    print("File missing", filename)
                    raise Exception("File Missing", filename)
                    continue
                hist = data_from_csv(filename)
                
                #print(filename, len(hist))
                #if len(hist) == 99:
                #    print("Warning for score %s, t=%s, k=%d, s=%d" % (dataset, technique, k, seed))
                sco10 = len(hist)
                #sco10 = score_history2(hist, stop='p10', epsilon=6)
                
                if sco10 == -1:
                    print("No score for %s in dataset %s, seed=%d k=%d" % (technique, dataset, seed, k))
                    continue
                hist = hist[:sco10]
                score_list.append(sco10)
                
                if hist[-1] < best:
                    best = hist[-1]
                    best_seed[technique] = seed
                
                #print(dataset, technique, seed, len(hist))
                for i, h in enumerate(hist):
                    if i > 0 and i <= max_iter:
                        if i not in datapoint:
                            datapoint[i] = []
                        datapoint[i].append(h)
                        #frames.append([technique, i, h, k])
                #print filename, len(hist)
                lines[technique][seed] = hist[1:]
            frames[fname][technique] = datapoint
            print(dataset, technique, score_list, score_list2)
        
        best_lines[fname] = {}
        for t in best_seed:
            #print(t, lines[t].keys())
            if len(lines[t]) == 0:
                continue
            bs = best_seed[t]
            if bs is None:
                print("Warning best seed is none for technique %s" % t)
                continue
            print(dataset, lines[t].keys())
            best_lines[fname][t] = lines[t][bs]
        
    dnames = [titles[d] for d in datasets]
    
    dump_file('results/visdata/convergence%d.pkl' % k, (dnames, frames, best_lines))
    
if __name__ == '__main__':
    main2()