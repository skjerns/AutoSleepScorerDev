# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon

These are tools for the AutoSleepScorer.
"""

import mne.io
import csv
import numpy as np
import os.path
#import pyfftw
from scipy import fft
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import json
import os
import re


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]      
        
def jsondict2csv(json_file, csv_file):
    
    key_set = set()
    dict_list = list()
    with open(json_file) as f:
        for line in f:
            dic = json.loads(line)
            map(key_set.add,dic.keys())
            dict_list.append(dic)
    keys = list(sorted(key_set, key = natural_key))
    
    with open(csv_file, 'wb') as f:
        w = csv.DictWriter(f, keys, delimiter=';')
        w.writeheader()
        w.writerows(dict_list)
    
def append_json(json_filename, dic):
    with open(json_filename, 'a') as f:
        json.dump(dic, f)
        f.write('\n')    

def memory():
    import os
    from wmi import WMI
    w = WMI('.')
    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    return int(result[0].WorkingSet)/1024**2
    


def one_hot(hypno, n_categories):
    enc = OneHotEncoder(n_values=n_categories)
    hypno = enc.fit_transform(hypno).toarray()
    return np.array(hypno,'int32')
    
    
def shuffle_lists(*args,**options):
     """ function which shuffles two lists and keeps their elements aligned
         for now use sklearn, maybe later get rid of dependency
     """
     return shuffle(*args,**options)
    

    

        
    

#    return data     no need for return as data is immutable


def split_eeg(df, chunk_length):
    splits = int(len(df)/( chunk_length ))
    data = []
    for i in np.arange(splits):
        data.append(df[i*chunk_length:(i+1)*chunk_length])
    return data
    
    
def get_freq_bands (epoch):

    w = (fft(epoch,axis=0)).real
    w = w[:len(w)/2]
    w = np.split(w,50)
    for i in np.arange(50):
        w[i] = np.mean(w[i],axis=0)
    
    return np.array(np.sqrt(np.power(w,2)))
    

print ('loaded tools.py')
    

    