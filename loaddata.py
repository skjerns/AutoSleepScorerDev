# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 13:33:45 2016

@author: Simon
"""


import mne
import csv
import numpy as np
# Load dataset

def load_eeg(filename, dataformat = ''):
    
   # if dataformat == 'brainvision':
   #     data = mne.io.read_raw_brainvision(filename)
   #     return data
   #
   # ToDo NoOptionError: No option 'markerfile' in section: 'Common Infos' 
    data = mne.io.read_raw_brainvision(filename,verbose = 'CRITICAL')
    data = data.load_data(verbose = 'CRITICAL')
    print(data.ch_names)
    data = data.to_data_frame()
    data.drop(['VEOG','HEOG','STI 014','C3'],axis=1,inplace=True,errors='ignore')
    
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
    print('loaded test')
    
print ('loaded main')
    

