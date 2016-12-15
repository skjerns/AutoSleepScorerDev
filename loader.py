# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon

This is the loader for files for the AutoSleepScorer.
"""

fedf = 'd:\sleep\data\EMSA\EMSA_asc_preprocout_datanum_5.edf'
br = 'd:\sleep\data\EMSA\EMSA_asc_preprocout_datanum_5.vhdr'
import mne.io
import csv
import numpy as np
import os.path



# loads the header file using MNE
def load_eeg_header(filename, dataformat = '', verbose = 'WARNING'):
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
        data = mne.io.read_raw_artemis123(filename, verbose=verbose)             # CHECK if now in stable release
    elif dataformat == 'bti':
        data = mne.io.read_raw_bti(filename, verbose=verbose)
    elif dataformat == 'cnt':
        data = mne.io.read_raw_cnt(filename, verbose=verbose)
    elif dataformat == 'ctf':
        data = mne.io.read_raw_ctf(filename, verbose=verbose)
    elif dataformat == 'edf':
        data = mne.io.read_raw_edf(filename, verbose=verbose)
    elif dataformat == 'kit':
        data = mne.io.read_raw_kit(filename, verbose=verbose)
    elif dataformat == 'nicolet':
        data = mne.io.read_raw_nicolet(filename, verbose=verbose)
    elif dataformat == 'eeglab':
        data = mne.io.read_raw_eeglab(filename, verbose=verbose)
    elif dataformat == 'brainvision':                                            # CHECK NoOptionError: No option 'markerfile' in section: 'Common Infos' 
        data = mne.io.read_raw_brainvision(filename, verbose=verbose)
    elif dataformat == 'egi':
        data = mne.io.read_raw_egi(filename, verbose=verbose)
    elif dataformat == 'fif':
        data = mne.io.read_raw_fif(filename, verbose=verbose)
    else: 
        print(['Faile extension not recognized for file: ', filename])           # CHECK throw error here    
  
    if not data.info['sfreq'] == 100:
        print('Warning: Sampling frequency is not 100. Consider resampling')     # CHECK implement automatic resampling
    print('loaded header ' + filename)
    return data.load_data()

def trim_channels(data, channels):
    curr_ch = data.ch_names
    to_drop = curr_ch
    
    # find EMG, take first
    for ch in curr_ch:
        if ch in channels.keys():
            if channels[ch] == 'EMG': 
                print([ch, 'EMG'])
                to_drop.remove(ch)
                break
    # find EOG, take first
    for ch in curr_ch:
        if ch in channels.keys():
            if channels[ch] == 'EOG': 
                print([ch, 'EOG'])
                to_drop.remove(ch)
                break
    # find EEG, take first
    for ch in curr_ch:
        if ch in channels.keys():
            if channels[ch] == 'EEG': 
                print([ch, 'EEG'])
                to_drop.remove(ch)
                break
            
    print(to_drop)
    data.drop_channels(to_drop)
    return data



def load_hypnogram(filename):
    # if extension = eeg ...
    # elseif extension = edf...
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        data = []
        for row in reader:
            data.append(int(row[0]))
    return np.array(data)
    
def test1():
    print('this should not be loaded (test1 in loader.py)')
    
print ('loaded loader.py')
    

    