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
import models
import imp
import pickle
import keras_utils
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

    data = scipy.stats.mstats.zscore(data , axis = None)
    data = data.astype(np.float32)
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
    
data,target,groups = load_data()
#%%

print('Extracting features')
target = np.load('target.npy')
groups = np.load('groups.npy')
feats_eeg = np.load('feats_eeg.npy')# tools.feat_eeg(data[:,:,0])
feats_eog = np.load('feats_eog.npy')#tools.feat_eog(data[:,:,1])
feats_emg = np.load('feats_emg.npy')#tools.feat_emg(data[:,:,2])
feats = np.hstack([feats_eeg, feats_eog, feats_emg])
# 
if 'data' in vars():
    if np.sum(np.isnan(data)) or np.sum(np.isnan(data)):print('Warning! NaNs detected')
#%%

comment = 'rnn_test'
print(comment)

print("starting at")
#%%   model comparison
#r = dict()
#r['pure_rnn_5'] = cv(feats5, target5, groups5, models.pure_rnn, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
#r['pure_rnnx3_5'] = cv(feats5, target5, groups5, models.pure_rnn_3, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
#r['pure_rrn_do_5'] = cv(feats5, target5, groups5, models.pure_rnn_do, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
#r['ann_rrn_5'] = cv(feats5, target5, groups5, models.ann_rnn, name='5',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
#with open('results_recurrent_architectures.pkl', 'wb') as f:
#            pickle.dump(r, f)

#%%   seqlen
#r = dict()
##r['pure_rrn_do_1'] = cv(feats1, target1, groups1, models.pure_rnn_do, name='1',epochs=epochs, folds=10,batch_size=batch_size, counter=counter, plot=True, stop_after=35)
#for i in [1,2,3,4,5,6,7,8,9,10,15]:
#    feats_seq, target_seq, group_seq = tools.to_sequences(feats, target, groups, seqlen = i, tolist=False)
#    r['pure_rrn_do_' + str(i)] = cv(feats_seq, target_seq, group_seq, models.pure_rnn_do_stateful,
#                                    name=str(i),epochs=epochs, folds=10, batch_size=batch_size, 
#                                    counter=counter, plot=True, stop_after=35)
#
#with open('results_recurrent_seqlen1-15.pkl', 'wb') as f:
#            pickle.dump(r, f)

#%%
imp.reload(models)
imp.reload(keras_utils)

batch_size = 360
epochs = 250
#data_seq, target_seq, group_seq = tools.to_sequences(data,target,groups, seqlen=0, tolist=True)
gc.collect()


r = dict()
#r['rnn_extracted_-4'] = keras_utils.cv(data, target, groups, models.cnn3adam_filter, name='rnn_extracted_-4',
#                         epochs=epochs, folds=5,batch_size=batch_size, counter=counter, plot=True, stop_after=25,
#                         rnn=[models.pure_rnn, -4, 6, 1024])
#r['rnn_extracted_-1'] = keras_utils.cv(data, target, groups, models.cnn3adam_filter, name='rnn_extracted_-1',
#                         epochs=epochs, folds=5,batch_size=batch_size, counter=counter, plot=True, stop_after=25,
#                         rnn=[models.pure_rnn, -1, 6, 1024])
r['rnn_extracted_-2'] = keras_utils.cv(data, target, groups, models.cnn3adam_filter, name='rnn_extracted_-2',
                         epochs=epochs, folds=5,batch_size=batch_size, counter=counter, plot=True, stop_after=25,
                         rnn=[models.pure_rnn, -2, 6, 1024])
r['rnn_extracted_-3'] = keras_utils.cv(data, target, groups, models.cnn3adam_filter, name='rnn_extracted_-3',
                         epochs=epochs, folds=5,batch_size=batch_size, counter=counter, plot=True, stop_after=25,
                         rnn=[models.pure_rnn, -3, 6, 1024])


#with open('results_rnn_extracted.pkl', 'wb') as f:
#            pickle.dump(r, f)
