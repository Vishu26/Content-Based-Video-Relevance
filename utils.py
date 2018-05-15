import csv
import os
from collections import OrderedDict
import numpy as np

def split_index(path):

    idx = []

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            idx.append(row[0])
    
    return idx


def load_pool(path, idx):
    
    pool = OrderedDict()
    for i in idx:
        pool_path = os.path.join(path, i+'/'+i+'-c3d-pool5.npy')
        feature = np.load(pool_path)
        pool[i] = feature
    
    return pool
    

def load_relevance(path):
    
    rel = OrderedDict()
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\n')
        for row in reader:
            li = row[0].strip().split(',')
            rel[li[0]] = [x for x in li[1:] if x!='']
    
    return rel

def load_inception(path, idx):
    
    pool = OrderedDict()
    for i in idx:
        pool_path = os.path.join(path, i+'/'+i+'-inception-pool3.npy')
        feature = np.load(pool_path)
        pool[i] = feature
    
    return pool
