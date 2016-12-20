# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon

This is the loader for files for the AutoSleepScorer.
"""

edf = 'd:\sleep\data\EMSA_asc_preprocout_datanum_5.edf'
br = 'd:\sleep\data\EMSA_asc_preprocout_datanum_5.vhdr'

import mne.io
import csv
import numpy as np
import os.path
from scipy import fft

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
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=csv_delimiter)
            data = []
            for row in reader:
                data.append(row)
    else:
        print('unkown hypnogram format. please use CSV with rows as epoch')        
        
                
    return np.array(data)


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
    for ch in curr_ch:
        if ch in channels.keys():
            if channels[ch] == 'EMG': 
                to_drop.remove(ch)
                data.rename_channels(dict({ch:'EMG'}))
                break
            
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
#    return data     no need for return as data is immutable

def split_eeg(df, epoch, sample_freq = 100):
    len(df)
    splits = int(len(df)/( epoch * sample_freq ))
    data = []
    for i in np.arange(splits):
        data.append(df[i*sample_freq*epoch:(i+1)*sample_freq*epoch])
    return data
    
    
def get_freq_bands (epoch):
    w = (fft(epoch,axis=0)).real
    w = w[:len(w)/2]
    w = np.split(w,12)
    for i in np.arange(12):
        w[i] = sum(w[i])
    
    return np.array(np.log((np.abs(w)+0.0000000001)/2))


    
    
    
    
    
def test():
    print('this should not be loaded (test1 in loader.py)')
    
print ('loaded loader.py')
    

    