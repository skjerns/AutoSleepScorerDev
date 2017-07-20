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

import keras
import tools
import dill as pickle
import scipy
import models
import numpy as np
import keras_utils
from keras_utils import cv
if not 'sleeploader' in vars() : import sleeploader  # prevent reloading module
import matplotlib; matplotlib.rcParams['figure.figsize'] = (10, 3)
np.random.seed(42)
if __name__ == '__main__':
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
        datadir = 'c:\\sleep\\cshs50\\'
    
    def load_data(tsinalis=False):
        global sleep
        global data
        sleep = sleeploader.SleepDataset(datadir)
        if 'data' in vars():  
            del data; 
            gc.collect()
        sleep.load_object()
    
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
        return data, target, groups
        
    data,target,groups = load_data()
    #%%
    
    print('Extracting features')
    target = np.load('target.npy')
    groups = np.load('groups.npy')
    feats_eeg = np.load('feats_eeg.npy') # tools.feat_eeg(data[:,:,0])
    feats_eog = np.load('feats_eog.npy') #tools.feat_eog(data[:,:,1])
    feats_emg = np.load('feats_emg.npy') #tools.feat_emg(data[:,:,2])
    feats = np.hstack([feats_eeg, feats_eog, feats_emg])
    
    # 
    if 'data' in vars():
        if np.sum(np.isnan(data)) or np.sum(np.isnan(data)):print('Warning! NaNs detected')
    
    #
    #%%
    print("starting")
    
    result = dict()
    comment = 'testing_electrodes'
    print(comment)
    
    
    ##%% 
    epochs = 2
    batch_size = 500
    #val_batch_size = 30000
    #
    #result['feat_eeg'] = cv(feats_eeg, target, groups, models.ann, name = 'eeg', stop_after=0, plot=True)
    #result['feat_eog'] = cv(np.hstack([feats_eeg,feats_eog]), target, groups, models.ann, name = 'eeg+eog', stop_after=100)  
    #result['feat_emg'] = cv(np.hstack([feats_eeg,feats_emg]), target, groups, models.ann, name = 'eeg+emg', stop_after=100) 
#    r = cv(np.hstack([feats_eeg,feats_eog,feats_emg]), target, groups, models.ann,epochs=epochs, balanced=True, name = 'all', stop_after=25) 
    #
    #with open('results_electrodes.pkl', 'wb') as f:
    #            pickle.dump(result, f)
    ###%% 
    #val_batch_size = 1440
    #epochs = 250
    #
    #result['cnn_eeg_morefilter'] = cv(data[:,:,0:1], target, groups, models.cnn3adam_filter, name = 'eeg', stop_after=25, counter=counter,batch_size=batch_size)
    #result['cnn_eog_morefilter'] = cv(data[:,:,0:2], target, groups, models.cnn3adam_filter, name = 'eeg+eog', stop_after=25, counter=counter,batch_size=batch_size)  
    #result['cnn_emg_morefilter'] = cv(data[:,:,[0,2]], target, groups, models.cnn3adam_filter, name = 'eeg+emg', stop_after=25, counter=counter,batch_size=batch_size) 
    #result['cnn_all_morefilter'] = cv(data[:,:,:], target, groups, models.cnn3adam_filter, name = 'all_morefilter', stop_after=25, counter=counter,batch_size=batch_size) 
    #with open('results_electrodes_balanced.pkl', 'wb') as f:
    #            pickle.dump(result, f)
                
    
    
    #%% weighting test
    
#    r = pickle.load('results_balanced.pkl')
    batch_size = 256
    name = 'new sleeploader'
    r = keras_utils.cv (data, target, groups, models.cnn3adam_filter, name=name,
                         epochs=50, folds=5,batch_size=batch_size, counter=counter,
                         plot=True, stop_after=15, balanced=False, cropsize=2700)
    #r['cnn3_balanced'] = cv(data, target, groups, models.cnn3adam_filter,
    #                                name='balanced',epochs=300, folds=5, batch_size=batch_size, 
    #                                counter=counter, plot=True, stop_after=15, balanced=True)
    #r['cnn3_balanced_l2'] = cv(data, target, groups, models.cnn3adam_filter_l2,
    #                                name='balanced_l2',epochs=300, folds=5, batch_size=batch_size, 
    #                                counter=counter, plot=True, stop_after=15, balanced=True)
    #r['cnn3_l2'] = cv(data, target, groups, models.cnn3adam_filter_l2,
    #                                name='l2',epochs=300, folds=5, batch_size=batch_size, 
    #                                counter=counter, plot=True, stop_after=15)
    #pickle.dump(r, open('results_weighted.pkl', 'wb'))
    #r['cnn3_2logweighted'] = cv(data, target, groups, models.cnn3adam_filter,
    #                                name='weighted',epochs=300, folds=5, batch_size=batch_size, 
    #                                counter=counter, plot=True, stop_after=15, weighted=True, log=True)
    #pickle.dump(r, open('results_weighted.pkl', 'wb'))
    #r = pickle.load(open('results_weighted.pkl', 'rb'))
    #r['cnn3_1not_weighted'] = cv(data, target, groups, models.cnn3adam_filter,
    #                                name='not weighted',epochs=300, folds=5, batch_size=batch_size, 
    #                                counter=counter, plot=True, stop_after=15, weighted=False, log=False)
#    pickle.dump(r, open('results_balanced_ss2.pkl', 'wb'))
    
    
    
