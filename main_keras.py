# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os
import gc; gc.collect()
import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras
import tools
import scipy
from keras_utils import cv
if not 'sleeploader' in vars() : import sleeploader  # prevent reloading module
import matplotlib; matplotlib.rcParams['figure.figsize'] = (10, 3)
np.random.seed(42)

try:
    with open('count') as f:
        counter = int(f.read())
except IOError:
    print('No previous experiment?')
    counter = 0
     
with open('count', 'w') as f:
  f.write(str(counter+1))        

#%%
if os.name == 'posix':
    datadir  = '../'

else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'C:\\sleep\\vinc\\brainvision\\correct\\'
    datadir = 'd:\\sleep\\corrupted\\'

def load_data(tsinalis=False):
    global sleep
    global data
    sleep = sleeploader.SleepDataset(datadir)
    if 'data' in vars():  
        del data; 
        gc.collect()
    if not sleep.loaded: sleep.load_object()

    data, target, groups = sleep.get_all_data(groups=True)

    data    = scipy.stats.mstats.zscore(data , axis = None)

    target[target==5] = 4

    target[target==8] = 0
    target = keras.utils.to_categorical(target)
    if tsinalis:
        data = data[:,:,0]
        data = tools.future(data,4)
        data = np.expand_dims(data,-1)
        data = np.expand_dims(data,1)
#    else:
#        data = np.swapaxes(data,1,2)
    return data, target, groups
    
#data,target,groups = load_data()
#%%

print('Extracting features')
target = np.load('target.npy')
groups = np.load('groups.npy')
feats_eeg = np.load('feats_eeg.npy')# tools.feat_eeg(data[:,:,0])
feats_eog = np.load('feats_eog.npy')#tools.feat_eog(data[:,:,1])
feats_emg = np.load('feats_emg.npy')#tools.feat_emg(data[:,:,2])
feats = np.hstack([feats_eeg, feats_eog, feats_emg])

#train_data = np.hstack( (tools.feat_eeg(train_data[:,:,0]), tools.feat_eog(train_data[:,:,1]),tools.feat_emg(train_data[:,:,2])))
#test_data  = np.hstack( (tools.feat_eeg(test_data[:,:,0]), tools.feat_eog(test_data[:,:,1]), tools.feat_emg(test_data[:,:,2])))
#child_data = np.hstack( (tools.feat_eeg(child_data[:,:,0]), tools.feat_eog(child_data[:,:,1]), tools.feat_emg(child_data[:,:,2])))
#train_data =  tools.feat_eeg(train_data[:,:,0])
#test_data  =  tools.feat_eeg(test_data[:,:,0])
 
#train_data =   np.hstack([tools.get_freqs(train_data[:,:,0],50),tools.feat_eog(train_data[:,:,1])])
#test_data  =   np.hstack([tools.get_freqs(test_data[:,:,0], 50),tools.feat_eog(test_data[:,:,1])])
 

#train_data =  tools.future(train_data, 4)
#test_data  =  tools.future(test_data, 4)

#training_data   = custom_datasets.DynamicData(train_data, train_target, batch_size=batch_size)
#validation_data = custom_datasets.DynamicData(test_data, test_target, batch_size=batch_size)             
#train_data = np.expand_dims(train_data,1)
#test_data = np.expand_dims(test_data,1)
#train_data=train_data.swapaxes(1,2)
#test_data=test_data.swapaxes(1,2)
#train_data = train_data[:,0:1]
#test_data = test_data[:,0:1]
#train_data = train_data[:,0:1,:]
#test_data = test_data[:,0:1,:]

# 
if 'data' in vars():
    if np.sum(np.isnan(data)) or np.sum(np.isnan(data)):print('Warning! NaNs detected')
#%%
print("starting")

n_classes = target.shape[1]
batch_size = 768    
val_batch_size = 768
epochs = 2
comment = 'model selection feat'
print(comment)

feats = np.hstack([feats_eeg,feats_eog,feats_emg])
print("starting")


n_classes = target.shape[1]
batch_size = 64    
val_batch_size = 64
epochs = 4
comment = 'Keras test'
print(comment)
r = []
input_shape = (data.shape[1:]) #train_data.shape
#for modfun in [models.cnn3adam,models.cnn3morefilter]:
for modfun in []:
    r.append(cv(data, target, groups, modfun))
     
