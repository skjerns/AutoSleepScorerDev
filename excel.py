# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:02:04 2017

@author: Simon
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import re
import time
from tools import plot_confusion_matrix
stop

#%%
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]   
def x():
    time.sleep(1)

#l=a['feat_eeg']
#val_acc = [y[0] for y in [x for x in l]]
#val_f1 = [y[1] for y in [x for x in l]]
#test_acc = [y[2] for y in [x for x in l]]
#test_f1 = [y[3] for y in [x for x in l]]
#
#val = np.vstack([val_acc, val_f1]).T
#test = np.vstack([test_acc, test_f1]).T

a   = pickle.load(open('.\\results\\results_electrodes.pkl','rb'))
names = sorted(a.keys(), key=natural_key)
res = [[a[key]] for key in names]



i=0
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s1 = np.vstack(all_scores).T
i=1
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s2 = np.vstack(all_scores).T
i=2
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s3 = np.vstack(all_scores).T
i=3
all_scores = list()
for exp in res:
    print (exp)
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
s4 = np.vstack(all_scores).T


stop
#%%

#l=a['feat_eeg']
#val_acc = [y[0] for y in [x for x in l]]
#val_f1 = [y[1] for y in [x for x in l]]
#test_acc = [y[2] for y in [x for x in l]]
#test_f1 = [y[3] for y in [x for x in l]]
#
#val = np.vstack([val_acc, val_f1]).T
#test = np.vstack([test_acc, test_f1]).T

file = '.\\results\\results_rnn_extracted.pkl'
a = pickle.load(open(file,'rb'))
l1 = a['ann_rrn_5']
l2 = a['pure_rnn_5']
l3 = a['pure_rnnx3_5']
l4 = a['pure_rrn_do_5']

i=3
s1 = [y[i] for y in [x for x in a]]
s2 = [y[i] for y in [x for x in l2]]
s3 = [y[i] for y in [x for x in l3]]
s4 = [y[i] for y in [x for x in l4]]


s = np.vstack([s1, s2,s3,s4]).T

stop
#%%

