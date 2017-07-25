# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
import os
import gc; gc.collect()
import matplotlib;matplotlib.use('Agg')
if __name__=='__main__': import keras;import keras_utils
import tools
import dill as pickle
import scipy
from scipy.stats.mstats import zmap
import numpy as np
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
#    if datadir =='c:\\sleep\\emsa': filename='adult'
    sleep = sleeploader.SleepDataset(datadir)

    if not sleep.load_object(filename): sleep.load()

    data, target, groups = sleep.get_all_data(groups=True)

#    data = scipy.stats.mstats.zscore(data , axis = None)
    target[target==5] = 4
    target[target==8] = 0
    return data, target, groups
    

if __name__=='__main__':
    w_rnn = './weights/1148rnn new sleeploaderfc1_3_0.890-0.828'
    w_cnn = './weights/1148rnn new sleeploadercnn3adam_filter_l2_3_0.853-0.785'
    cnn = keras.models.load_model(w_cnn)
    rnn = keras.models.load_model(w_rnn)
    
    trainset =  'c:\\sleep\\cshs50'
    datadirs = ['c:\\sleep\\cshs50']
#                'c:\\sleep\\emsa']
    
    data    = {}
    targets = {}
    groups  = {}
    results = {}
    
#    data['train'], targets['train'], groups['train'] = load_data(trainset)
    
    for datadir in datadirs:
        name = os.path.basename(datadir[:-1] if (datadir[-1] == '\\' or datadir[-1]=='/') else datadir)
        print('results for ', name)
        data[name], targets[name], groups[name] = load_data(datadir)
        data[name] = zmap(data[name], data[name], None)
        results[name] = keras_utils.test_data_cnn_rnn(data[name], targets[name], cnn, 'fc1', rnn, cropsize=2700, mode = 'preds')
        tools.plot_results_per_patient(results[name][0], targets[name], groups[name], title='CNN '+ name)
        tools.plot_results_per_patient(results[name][1], results[name][3], groups[name][5:], title='CNN+LSTM '+ name)
