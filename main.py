# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from loaddata import load_eeg, load_hypnogram
import os
import gc
import test
import numpy as np
from tools import split_eeg, get_freq_bands
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import svm
gc.collect()

datadir = 'd:\\sleep\\data\EMSA\\'

#data  = load_eeg('D:\sleep\data\EMSA\EMSA_asc_preprocout_datanum_1.vhdr')
#hypno = load_hypnogram('test-Hypnogram.edf')

files = [s for s in os.listdir(datadir) if s.endswith('.txt')]
hypno = list()
for file in files:
    hypno.append(load_hypnogram(datadir + file))

files = [s for s in os.listdir(datadir) if s.endswith('.vhdr')]
data = list()
i = 0
for file in files:
    eeg = load_eeg(datadir + file)
    eeg = split_eeg(eeg, 30, 100)
    data.append(eeg)
    i = i + 1
#    if i == 50: break;
    
    
print('-----------------------------------------------------------------------')
#%%
i=0
freq = []
for pp in data:
    i = i +1
    for epoch in pp:
        freq.append((np.hstack(get_freq_bands(epoch))).T)
    print (i)
    
hypno = np.hstack(hypno)
hypno = hypno[0:len(freq)]

split = int(len(freq)*0.8)

#%%
print('firing up the classifiers')
idx = np.arange(len(freq)); np.random.shuffle(idx);

freq = [freq[x] for x in idx]
hypno = [hypno[x] for x in idx]


y = hypno[split:]
#np.random.shuffle(y)
print('ADA 100')
clf = AdaBoostClassifier(n_estimators=100, random_state=42)
clf = clf.fit(freq[0:split], hypno[0:split])
pred = clf.score(freq[split:], y)
np.random.shuffle(y)
shuff = clf.score(freq[split:], y)
print (['prediction score: ', pred])
print (['random expected: ', shuff])