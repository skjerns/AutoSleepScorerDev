# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:19:34 2016

@author: Simon
"""

import numpy as np
from scipy import fft


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