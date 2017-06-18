# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon

These are tools for the AutoSleepScorer.
"""

import csv
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#import pyfftw
from scipy import fft
from scipy import stats
from scipy.signal import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import json
import os
import re

use_sfreq=100.0



def confmat_to_numpy(confmat_str):
    rows = confmat_str.split('] [')
    new_array = []
    for s in rows:
        s = s.replace('[[','')
        s = s.replace(']]','')
        s = s.split(' ')
        s = [int(x) for x in s if x is not '']
        new_array.append(s)
    return np.array(new_array)
        
    
def convert_Y_to_seq_batches(Y, batch_size):
    if (len(Y)%batch_size)!= 0: Y = Y[:-(len(Y)%batch_size)]
    idx = np.arange(len(Y))
    idx = idx.reshape([batch_size,-1]).flatten('F')
    return Y[idx]


def to_sequences(data, *args, seqlen = 0, tolist = True, wrap = False):
    '''
    Creates time-sequences
    :returns list: list of list of numpy arrays. this way no memory redundance is created
    '''
    if seqlen==0: return data if args is  () else [data]+list(args)
    if seqlen==1: return  np.expand_dims(data,1) if args is () else [np.expand_dims(data,1)]+list(args)
    if args is not (): assert np.all([len(data)==len(x) for x in args]), 'Data and Targets must have same length'
    assert data.shape[0] > seqlen, 'Future steps must be smaller than number of datapoints'
    
    data = [x for x in data]
    new_data = []
    for i in range((len(data))if wrap else (len(data)-seqlen+1) ):
        seq = []
        for s in range(seqlen):
            seq.append(data[(i+s)%len(data)])
        new_data.append(seq)
    if not tolist: 
        new_data = np.array(new_data, dtype=np.float32)  
        
    if args is not ():
        new_data = [new_data]
        for array in args:
            new_array = np.roll(array, -seqlen+1)
            if not wrap: new_array = new_array[:-seqlen+1]
            assert len(new_array)==len(new_data[0]), 'something went wrong {}!={}'.format(len(new_array),len(new_data[0])) 
            new_data.append(new_array)
    
    return new_data


#def normalize(input_directory, output_directory):
#    """
#    Takes all EEG files in the directory and does the following:
#    - Remove all unused headers
#    - Resample to 100hz
#    """   
#    eeg_files = [s for s in os.listdir(input_directory) if s.endswith(('.vhdr','rec','edf'))]
#    for eeg_file in eeg_files:
#        header = load_eeg_header( os.path.join( input_directory, eeg_file), preload=True)
#        trim_channels(header, sleeploader.SleepDataset.channels)
#        header = header.resample(100.0)

def label_to_one_hot(y):
    '''
    Convert labels into "one-hot" representation
    '''
    n_values = np.max(y) + 1
    y_one_hot = np.eye(n_values)[y]
    return y_one_hot




def normalize(signals):
    """
    :param signals: 1D, 2D or 3D signals
    returns each element to have mean 0
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0) 
    if signals.ndim == 2: signals = np.expand_dims(signals,2)
    new_signals = np.zeros(signals.shape, dtype=np.int32)
    for i in np.arange(signals.shape[2]):
        new_signals[:,:,i] = np.subtract(signals[:,:,i].T,np.mean(signals[:,:,i],axis=1)).T
        
    return new_signals.squeeze() if new_signals.shape[2]==1 else new_signals




def future(signals, fsteps):
    """
    adds fsteps points of the future to the signal
    :param signals: 2D or 3D signals
    :param fsteps: how many future steps should be added to each data point
    """
    if fsteps==0: return signals
    assert signals.shape[0] > fsteps, 'Future steps must be smaller than number of datapoints'
    if signals.ndim == 2: signals = np.expand_dims(signals,2) 
    nsamp = signals.shape[1]
    new_signals = np.zeros((signals.shape[0],signals.shape[1]*(fsteps+1), signals.shape[2]),dtype=np.float32)
    for i in np.arange(fsteps+1):
        new_signals[:,i*nsamp:(i+1)*nsamp,:] = np.roll(signals[:,:,:],-i,axis=0)
    return new_signals.squeeze() if new_signals.shape[2]==1 else new_signals



def feat_eeg(signals):
    """
    calculate the relative power as defined by Leangkvist (2012),
    assuming signal is recorded with 100hz
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    feats = np.zeros((signals.shape[0],9),dtype='float32')
    # 5 FEATURE for freq babnds
    w = (fft(signals,axis=1)).real
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    spindle = np.sum(np.abs(w[:,np.arange(12*nsamp/sfreq,14*nsamp/sfreq, dtype=int)]),axis=1)
    sum_abs_pow = delta + theta + alpha + beta + gamma + spindle
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = spindle /sum_abs_pow
    feats[:,6] = np.log10(stats.kurtosis(signals,fisher=False,axis=1))        # kurtosis
    feats[:,7] = np.log10(-np.sum([(x/nsamp)*(np.log(x/nsamp)) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    #feats[:,7] = np.polynomial.polynomial.polyfit(np.log(f[np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]), np.log(w[0,np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),1)
    feats[:,8] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5)
    return np.nan_to_num(feats)



def feat_wavelet(signals):
    """
    calculate the relative power as defined by Leangkvist (2012),
    assuming signal is recorded with 100hz
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    feats = np.zeros((signals.shape[0],8),dtype='float32')
    # 5 FEATURE for freq babnds
    w = (fft(signals,axis=1)).real
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    sum_abs_pow = delta + theta + alpha + beta + gamma
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = np.log10(stats.kurtosis(signals,fisher=False,axis=1))        # kurtosis
    feats[:,6] = np.log10(-np.sum([(x/nsamp)*(np.log(x/nsamp)) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    #feats[:,7] = np.polynomial.polynomial.polyfit(np.log(f[np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]), np.log(w[0,np.arange(0.5*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),1)
    feats[:,7] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5)
    return np.nan_to_num(feats)


def feat_eog(signals):
    """
    calculate the EOG features
    :param signals: 1D or 2D signals
    """

    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    w = (fft(signals,axis=1)).real   
    feats = np.zeros((signals.shape[0],15),dtype='float32')
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    sum_abs_pow = delta + theta + alpha + beta + gamma
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5) #smean
    feats[:,6] = np.max(signals, axis=1)    #PAV
    feats[:,7] = np.min(signals, axis=1)    #VAV   
    feats[:,8] = np.argmax(signals, axis=1)/nsamp #PAP
    feats[:,9] = np.argmin(signals, axis=1)/nsamp #VAP
    feats[:,10] = np.sum(np.abs(signals), axis=1)/ np.mean(np.sum(np.abs(signals), axis=1)) # AUC
    feats[:,11] = np.sum(((np.roll(np.sign(signals), 1,axis=1) - np.sign(signals)) != 0).astype(int),axis=1)/nsamp #TVC
    feats[:,12] = np.log10(np.std(signals, axis=1)) #STD/VAR
    feats[:,13] = np.log10(stats.kurtosis(signals,fisher=False,axis=1))       # kurtosis
    feats[:,14] = np.log10(-np.sum([(x/nsamp)*(np.log(x/nsamp)) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    
    return np.nan_to_num(feats)


def feat_emg(signals):
    """
    calculate the EMG median as defined by Leangkvist (2012),
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    sfreq = use_sfreq
    nsamp = float(signals.shape[1])
    w = (fft(signals,axis=1)).real   
    feats = np.zeros((signals.shape[0],13),dtype='float32')
    delta = np.sum(np.abs(w[:,np.arange(0.5*nsamp/sfreq,4*nsamp/sfreq, dtype=int)]),axis=1)
    theta = np.sum(np.abs(w[:,np.arange(4*nsamp/sfreq,8*nsamp/sfreq, dtype=int)]),axis=1)
    alpha = np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,13*nsamp/sfreq, dtype=int)]),axis=1)
    beta  = np.sum(np.abs(w[:,np.arange(13*nsamp/sfreq,20*nsamp/sfreq, dtype=int)]),axis=1)
    gamma = np.sum(np.abs(w[:,np.arange(20*nsamp/sfreq,50*nsamp/sfreq, dtype=int)]),axis=1)   # only until 50, because hz=100
    sum_abs_pow = delta + theta + alpha + beta + gamma
    feats[:,0] = delta /sum_abs_pow
    feats[:,1] = theta /sum_abs_pow
    feats[:,2] = alpha /sum_abs_pow
    feats[:,3] = beta  /sum_abs_pow
    feats[:,4] = gamma /sum_abs_pow
    feats[:,5] = np.dot(np.array([3.5,4,5,7,30]),feats[:,0:5].T ) / (sfreq/2-0.5) #smean
    emg = np.sum(np.abs(w[:,np.arange(12.5*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)
    feats[:,6] = emg / np.sum(np.abs(w[:,np.arange(8*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)  # ratio of high freq to total motor
    feats[:,7] = np.median(np.abs(w[:,np.arange(8*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)    # median freq
    feats[:,8] = np.mean(np.abs(w[:,np.arange(8*nsamp/sfreq,32*nsamp/sfreq, dtype=int)]),axis=1)    #  mean freq
    feats[:,9] = np.std(signals, axis=1)    #  std 
    feats[:,10] = np.mean(signals,axis=1)
    feats[:,11] = np.log10(stats.kurtosis(signals,fisher=False,axis=1) )
    feats[:,12] = np.log10(-np.sum([(x/nsamp)*(np.log(x/nsamp)) for x in np.apply_along_axis(lambda x: np.histogram(x, bins=8)[0], 1, signals)],axis=1))  # entropy.. yay, one line...
    return np.nan_to_num(feats)


def feat_emgmedianfreq(signals):
    """
    calculate the EMG median as defined by Leangkvist (2012),
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    return np.median(abs(signals),axis=1)


def get_features(data):
    """
    returns a vector with extraced features
    :param data: datapoints x samples x dimensions (dimensions: EEG,EOG,EMG)
    """
#    assert(ndims=3)
    for i in np.arange(data.shape[2]):
        
        pass
    
def save_results(save_dict=None, **kwargs):
    np.set_printoptions(precision=2,threshold=np.nan)
    if save_dict==None:
        save_dict=kwargs
    for key in save_dict.keys():
        save_dict[key] = str(save_dict[key])
    np.set_printoptions(precision=2,threshold=1000)
    append_json('experiments.json', save_dict)
    jsondict2csv('experiments.json', 'experiments.csv')
    
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]      
        
def jsondict2csv(json_file, csv_file):
    
    key_set = set()
    dict_list = list()
    try:
        with open(json_file,'r') as f:
            for line in f:
                dic = json.loads(line)
                key_set.update(dic.keys())
                dict_list.append(dic)
        keys = list(sorted(list(key_set), key = natural_key))
    
        with open(csv_file, 'w') as f:
            w = csv.DictWriter(f, keys, delimiter=';', lineterminator='\n')
            w.writeheader()
            w.writerows(dict_list)
    except IOError:
        print('could not convert to csv-file. ')
        
    
def append_json(json_filename, dic):
    with open(json_filename, 'a') as f:
        json.dump(dic, f)
        f.write('\n')    


def plot_confusion_matrix(fname, conf_mat, target_names,
                          title='Confusion matrix', cmap='Blues', perc=True):
    """Plot Confusion Matrix."""
    cm = conf_mat
    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df = pd.DataFrame(data=cm, columns=target_names, index=target_names)
    g  = sns.heatmap(df, annot=True if perc else conf_mat , fmt=".1f" if perc else ".0f",
                     linewidths=.5, vmin=0, vmax=100, cmap=cmap)    
    g.set_title(title)
    g.set_ylabel('True label')
    g.set_xlabel('Predicted label')
    g.figure.savefig('.//results//' + fname)


def memory():
    from wmi import WMI
    w = WMI('.')
    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    return int(result[0].WorkingSet)/1024**2
    


def one_hot(hypno, n_categories):
    enc = OneHotEncoder(n_values=n_categories)
    hypno = enc.fit_transform(hypno).toarray()
    return np.array(hypno,'int32')
    
    
def shuffle_lists(*args,**options):
     """ function which shuffles two lists and keeps their elements aligned
         for now use sklearn, maybe later get rid of dependency
     """
     return shuffle(*args,**options)
    
    
def epoch_voting(Y, chunk_size):
    
    
    Y_new = Y.copy()
    
    for i in range(1+len(Y_new)/chunk_size):
        epoch = Y_new[i*chunk_size:(i+1)*chunk_size]
        if len(epoch) != 0: winner = np.bincount(epoch).argmax()
        Y_new[i*chunk_size:(i+1)*chunk_size] = winner              
    return Y_new

        

    
    
def get_freqs (signals, nbins=0):
    """ extracts relative fft frequencies and bins them in n bins
    :param signals: 1D or 2D signals
    :param nbins:  number of bins used as output (default: maximum possible)
    """
    if signals.ndim == 1: signals = np.expand_dims(signals,0)
    sfreq = use_sfreq
    if nbins == 0: nbins = int(sfreq/2)
    
    nsamp = float(signals.shape[1])
    assert nsamp/2 >= nbins, 'more bins than fft results' 
    
    feats = np.zeros((int(signals.shape[0]),nbins),dtype='float32')
    w = (fft(signals,axis=1)).real
    for i in np.arange(nbins):
        feats[:,i] =  np.sum(np.abs(w[:,np.arange(i*nsamp/sfreq,(i+1)*nsamp/sfreq, dtype=int)]),axis=1)
    sum_abs_pow = np.sum(feats,axis=1)
    feats = (feats.T / sum_abs_pow).T
    return feats


def trim_channels(data, channels):
        print(data.ch_names)
    #    channels dict should look like this:
    #            channels = dict({'EOG' :'EOG',
    #                    'VEOG':'EOG',
    #                    'HEOG':'EOG',
    #                    'EMG' :'EMG',
    #                    'EEG' :'EEG',
    #                    'C3'  :'EEG',
    #                    'C4'  :'EEG'})
    
        curr_ch = data.ch_names
        to_drop = list(curr_ch)
        # find EMG, take first
        for ch in curr_ch:
            if ch in channels.keys():
                if channels[ch] == 'EMG': 
                    to_drop.remove(ch)
                    data.rename_channels(dict({ch:'EMG'}))
                    break
    #            
        # find EOG, take first
        for ch in curr_ch:
            if ch in channels.keys():
                if channels[ch] == 'EOG': 
                    to_drop.remove(ch)
                    data.rename_channels(dict({ch:'EOG'}))
                    break
        # find EEG, take first
        for ch in curr_ch:
            if ch in channels.keys():
                if channels[ch] == 'EEG': 
                    to_drop.remove(ch)
                    data.rename_channels(dict({ch:'EEG'}))
                    break       
        data.drop_channels(to_drop)
        
        
def load_eeg_header(filename, dataformat = '', **kwargs):            # CHECK include kwargs
        dataformats = dict({
                            #'bin' :'artemis123',
                            '???' :'bti',                                           # CHECK
                            'cnt' :'cnt',
                            'ds'  :'ctf',
                            'edf' :'edf',
                            'rec' :'edf',
                            'bdf' :'edf',
                            'sqd' :'kit',
                            'data':'nicolet',
                            'set' :'eeglab',
                            'vhdr':'brainvision',
                            'egi' :'egi',
                            'fif':'fif',
                            'gz':'fif',
                            })
        if dataformat == '' :      # try to guess format by extension 
            ext = os.path.splitext(filename)[1][1:].strip().lower()                
            dataformat = dataformats[ext]
            
        if dataformat == 'artemis123':
            data = mne.io.read_raw_artemis123(filename, **kwargs)             # CHECK if now in stable release
        elif dataformat == 'bti':
            data = mne.io.read_raw_bti(filename, **kwargs)
        elif dataformat == 'cnt':
            data = mne.io.read_raw_cnt(filename, **kwargs)
        elif dataformat == 'ctf':
            data = mne.io.read_raw_ctf(filename, **kwargs)
        elif dataformat == 'edf':
            data = mne.io.read_raw_edf(filename, **kwargs)
        elif dataformat == 'kit':
            data = mne.io.read_raw_kit(filename, **kwargs)
        elif dataformat == 'nicolet':
            data = mne.io.read_raw_nicolet(filename, **kwargs)
        elif dataformat == 'eeglab':
            data = mne.io.read_raw_eeglab(filename, **kwargs)
        elif dataformat == 'brainvision':                                            # CHECK NoOptionError: No option 'markerfile' in section: 'Common Infos' 
            data = mne.io.read_raw_brainvision(filename, **kwargs)
        elif dataformat == 'egi':
            data = mne.io.read_raw_egi(filename, **kwargs)
        elif dataformat == 'fif':
            data = mne.io.read_raw_fif(filename, **kwargs)
        else: 
            print(['Failed extension not recognized for file: ', filename])           # CHECK throw error here    
          
        if not 'verbose' in  kwargs: print('loaded header ' + filename);
        
        return data       

print ('loaded tools.py')
    

    