# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import gc
import numpy as np
from loader import load_eeg_header, load_hypnogram, trim_channels
from tools import split_eeg, get_freq_bands
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import svm
gc.collect()

datadir = 'd:\\sleep\\data\\'




#files = [s for s in os.listdir(datadir) if s.endswith('.txt')]
#hypno = list()
#for file in files:
#    hypno.append(load_hypnogram(datadir + file))

files = [s for s in os.listdir(datadir) if s.endswith('.edf')]
data = list()
i = 0
premem = memory()/(1024*1024)
for file in files:
    channels = dict({'EOG' :'EOG',
                    'VEOG':'EOG',
                    'HEOG':'EOG',
                    'EMG' :'EMG',
                    'EEG' :'EEG',
                    'C3'  :'EEG',
                    'C4'  :'EEG'})
    print(premem-memory()/(1024*1024))
                    
    eeg = load_eeg_header(datadir + file)
    gc.collect()
    premem = memory()/(1024*1024)
#    eeg = trim_channels(eeg, channels)
#    eeg = split_eeg(eeg, 30, 100)
#    data.append(eeg)
    i = i + 1
#    if i == 50: break;
    gc.collect()


STOP
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