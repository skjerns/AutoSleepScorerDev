# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:15:31 2017

@author: Simon
"""



import sleeploader
import imp
imp.reload(sleeploader)
if __name__=='__main__':
    sleep = sleeploader.SleepDataset('D:/sleep/isruc')
    channels = {'EEG':['C4-A1','C4-M1'], 'EMG':'X1','EOG':['LOC-A2','E1-M2']}
    references = {'RefEEG':False, 'RefEMG':False,'RefEOG':False}

    sleep.load(channels=channels)
    sleep.save_object()

