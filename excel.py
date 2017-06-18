# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:02:04 2017

@author: Simon
"""
import pickle
import numpy as np
import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]   

#l=a['feat_eeg']
#val_acc = [y[0] for y in [x for x in l]]
#val_f1 = [y[1] for y in [x for x in l]]
#test_acc = [y[2] for y in [x for x in l]]
#test_f1 = [y[3] for y in [x for x in l]]
#
#val = np.vstack([val_acc, val_f1]).T
#test = np.vstack([test_acc, test_f1]).T

a   = pickle.load(open('results_recurrent_seqlen1-9.pkl','rb'))
res = [[a[key]] for key in sorted(a.keys(), key=natural_key)]



i=3
all_scores = list()
for exp in res:
    scores = list()
    for fold in exp[0]:
        scores.append(fold[i])
    all_scores.append(scores)
#s1 = [y[i] for y in [x for x in l1] for l1 in res]
#s2 = [y[i] for y in [x for x in l2]]
#s3 = [y[i] for y in [x for x in l3]]
#s4 = [y[i] for y in [x for x in l4]]
#s5 = [y[i] for y in [x for x in l5]]


s = np.vstack(all_scores).T
