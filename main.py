# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import os
import gc
import numpy as np
from loader import load_eeg_header, load_hypnogram, trim_channels,split_eeg, check_for_normalization
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn import svm
gc.collect()

datadir = 'd:\\sleep\\data\\'

channels = dict({'EOG' :'EOG',
                 'EOGmix':'EOG',
                'VEOG':'EOG',
                'HEOG':'EOG',
                'EMG' :'EMG',
                'EEG' :'EEG',
                'C3'  :'EEG',
                'C4'  :'EEG',
                'EEG1':'EEG',
                'EEG2':'EEG'})

files = [s for s in os.listdir(datadir) if s.endswith('.txt')]
hypno = list()
for file in files:
    hypno.append(load_hypnogram(datadir + file))
    
files = [s for s in os.listdir(datadir) if s.endswith('.edf')]
data = list()
i = 0

for file in files:
    header = load_eeg_header(datadir + file,verbose='CRITICAL', preload=False)
#    trim_channels(header, channels)
#    check_for_normalization(header)
#    eeg = header.to_data_frame()
#    eeg = split_eeg(eeg,30,100)
    print([str(file[-6]) + str(file[-5]), header.ch_names])
    # DO STUFF WITH DATA
#    data.append(eeg)
#    del 
    i = i + 1
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