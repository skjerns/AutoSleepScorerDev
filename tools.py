# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon

These are tools for the AutoSleepScorer.
"""

import mne.io
import csv
import numpy as np
import os.path
#import pyfftw
from scipy import fft
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import json
import os
import re

with open('..\AutoSleepScorer\experiments.csv', 'r') as csvfile:
    writer = csv.DictReader(csvfile, delimiter=";")
    dic = list()
    for row in writer:
        dic.append(row)
        
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]      
        
def jsondict2csv(json_file, csv_file):
    
    key_set = set()
    dict_list = list()
    with open(json_file) as f:
        for line in f:
            dic = json.loads(line)
            map(key_set.add,dic.keys())
            dict_list.append(dic)
    keys = list(sorted(key_set, key = natural_key))
    
    with open(csv_file, 'wb') as f:
        w = csv.DictWriter(f, keys, delimiter=';')
        w.writeheader()
        w.writerows(dict_list)
    
def append_json(json_filename, dic):
    with open(json_filename, 'a') as f:
        json.dump(dic, f)
        f.write('\n')    

def memory():
    import os
    from wmi import WMI
    w = WMI('.')
    result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
    return int(result[0].WorkingSet)
    
def load_hypnogram(filename, dataformat = '', csv_delimiter='\t'):
    
    dataformats = dict({
                        'txt' :'csv',
                        'csv' :'csv',                                           
                        })
    if dataformat == '' :      # try to guess format by extension 
        ext = os.path.splitext(filename)[1][1:].strip().lower()                
        dataformat = dataformats[ext]
        
    if dataformat == 'csv':
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=csv_delimiter)
            data = []
            for row in reader:
                data.append(int(row[0]))
        
    else:
        print('unkown hypnogram format. please use CSV with rows as epoch')        
        
    
    data = np.array(data).reshape(-1, 1)
    return data
    

def one_hot(hypno, n_categories):
    enc = OneHotEncoder(n_values=n_categories)
    hypno = enc.fit_transform(hypno).toarray()
    return np.array(hypno,'int32')
    
    
def shuffle_lists(*args,**options):
     """ function which shuffles two lists and keeps their elements aligned
         for now use sklearn, maybe later get rid of dependency
     """
     return shuffle(*args,**options)
    

# loads the header file using MNE
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
  
    if not data.info['sfreq'] == 100:
        print('Warning: Sampling frequency is not 100. Consider resampling')     # CHECK implement automatic resampling
        
    if not 'verbose' in  kwargs: print('loaded header ' + filename);
    
    return data

    
    
def check_for_normalization(data_header):
    
    if not data_header.info['sfreq'] == 100:
        print('WARNING: Data not with 100hz. Try resampling')      
        
    if not data_header.info['lowpass'] == 50:
        print('WARNING: lowpass not at 50')
        
        
    if not 'EOG' in data_header.ch_names:
        print('WARNING: EOG channel missing')
    if not 'EMG' in data_header.ch_names:
        print('WARNING: EMG channel missing')
    if not 'EEG' in data_header.ch_names:
        print('WARNING: EEG channel missing')
        
    
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
#    for ch in curr_ch:
#        if ch in channels.keys():
#            if channels[ch] == 'EMG': 
#                to_drop.remove(ch)
#                data.rename_channels(dict({ch:'EMG'}))
#                break
#            
    # find EOG, take first
#    for ch in curr_ch:
#        if ch in channels.keys():
#            if channels[ch] == 'EOG': 
#                to_drop.remove(ch)
#                data.rename_channels(dict({ch:'EOG'}))
#                break
    # find EEG, take first
    for ch in curr_ch:
        if ch in channels.keys():
            if channels[ch] == 'EEG': 
                to_drop.remove(ch)
                data.rename_channels(dict({ch:'EEG'}))
                break
            
    data.drop_channels(to_drop)
#    return data     no need for return as data is immutable


def split_eeg(df, epochlength=30, sample_freq = 100):
    splits = int(len(df)/( epochlength * sample_freq ))
    data = []
    for i in np.arange(splits):
        data.append(df[i*sample_freq*epochlength:(i+1)*sample_freq*epochlength])
    return data
    
    
def get_freq_bands (epoch):
#    print('new fft!')
#    fft = pyfftw.builders.fft(epoch,axis=0,threads=4)
#    w = (fft()).real
    w = (fft(epoch,axis=0)).real
    w = w[:len(w)/2]
    w = np.split(w,50)
    for i in np.arange(50):
        w[i] = sum(w[i])
    
    return np.array(np.sqrt(np.power(w,2)))
    

print ('loaded tools.py')
    

    