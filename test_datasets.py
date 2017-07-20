# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
import os
import gc; gc.collect()
import matplotlib;matplotlib.use('Agg')
import keras
import tools
import dill as pickle
import scipy
import numpy as np
import keras_utils
import sleeploader
import matplotlib
np.random.seed(42)

def create_object(datadir):
    sleep = sleeploader.SleepDataset(datadir)
    sleep.load()
    sleep.save_object()

def load_data(datadir, filename='sleepdata'):
    if 'data' in vars(): del data
    if 'sleep' in vars(): del sleep;  gc.collect()
        
    sleep = sleeploader.SleepDataset(datadir)

    if not sleep.load_object(filename): sleep.load()

    data, target, groups = sleep.get_all_data(groups=True)

    data = scipy.stats.mstats.zscore(data , axis = None)
    target[target==5] = 4
    target[target==8] = 0
    return data, target, groups
    

if __name__=='__main__':
    w_cnn = 'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/weights/1116new sleeploadercnn3adam_filter/1116new sleeploadercnn3adam_filter_1_0.858-0.751'
    w_rnn = 'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/weights/1116rnn new sleeploaderfc1_1_0.878-0.796'
#    datadir = 'c:\\sleep\\EMSA\\'
#    datadir = 'd:\\sleep\\vinc\\'
    datadir = 'c:\\sleep\\cshs100\\'
    create_object(datadir)
    data,target,groups = load_data(datadir)
#    feats_eeg = tools.feat_eeg(data[:,:,0])
#    feats_eog = tools.feat_eog(data[:,:,1])
#    feats_emg = tools.feat_emg(data[:,:,2])
#    np.save('feats_eeg', feats_eeg)
#    np.save('feats_eog', feats_eog)
#    np.save('feats_emg', feats_emg)

    
    cnn = keras.models.load_model(w_cnn)
    rnn = keras.models.load_model(w_rnn)
    result = keras_utils.test_data_cnn_rnn(data, target, cnn, 'fc1', rnn, cropsize=2700)
    print(result)
#sleep.load_object()
#data,target = sleep.get_all_data()
#target[target==5] = 4
#target[target==8] = 0
#
#emsa_feats = tools.get_all_features(data)
#
#start = time.time()
#tools.get_all_features(data)
#print(time.time()-start)
#
#start = time.time()
#tools.get_all_features_m(data)
#print(time.time()-start)