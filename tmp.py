# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:15:31 2017

@author: Simon
"""



import sleeploader
import imp
imp.reload(sleeploader)
if __name__=='__main__':
    sleep = sleeploader.SleepDataset('D:/sleep/mass/ss3')
    sleep.load(channels={'EEG':['EEG C3-CLE','EEG C3-LER'], 'EMG':'EMG CHIN2', 'EOG':'EOG RIGHT HORIZ'},
                   references={'RefEEG':['EEG A2-CLE','EEG A2-LER', 'EEG T4-LER'], 'RefEMG':'EMG CHIN1', 'RefEOG':'EOG LEFT HORIZ'})
    sleep.save_object()

