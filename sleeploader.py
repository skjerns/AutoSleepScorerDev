# -*- coding: utf-8 -*-
import os
import re
from loader import load_eeg_header, load_hypnogram, trim_channels,split_eeg, check_for_normalization
import numpy as np
print('loader')
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    
class MyClass:
    static_elem = 123

    def __init__(self):
        self.object_elem = 456
        
class SleepDataset():
    train_data = list()
    test_data  = list()
    loaded = False
    channels = dict({'EOG' :'EOG',
                 'EOGmix':'EOG',
                'VEOG':'EOG',
                'HEOG':'EOG',
                'EMG' :'EMG',
                'EEG' :'EEG',
                'C3'  :'EEG',
                'C4'  :'EEG',
                'EEG1':'EEG',
                'EEG2':'EEG'})
    
    def __init__(self, dirlist):
#        print(loaded)
        """
        :param dirlist: a directory string. yes it´s not a list yet.
        """
        if self.loaded == True:
            print("Data already loaded. To reload add parameter reload=True")
        self.dirlist = dirlist
            
       
    def load(self, force_reload=False):

        if self.loaded == True and force_reload == False:
            return self.train_data, self.test_data
        else:
            print('Reloading DataSet')
        self.rnd = np.random.RandomState(seed=42)
        self.hypno_files = [s for s in os.listdir(self.dirlist) if s.endswith('.txt')]
        self.hypno_files = sorted(self.hypno_files, key = natural_key)
        self.hypno = list()
        for i in range(len(self.hypno_files)):
            self.hypno.append(load_hypnogram(self.dirlist + self.hypno_files[i]))
        
        self.files = [s for s in os.listdir(self.dirlist) if s.endswith('.vhdr')]
        self.files = sorted(self.files, key = natural_key)
        assign = np.array([1]*10 + [0]*23)  # create split for train/test set
        self.rnd.shuffle(assign)
    
        assign = np.insert(assign,16,[2]*18)   # leave out child data for now
    
        
        
        for i in range(len(self.files)):
            if i > 14 and i < 32: continue;
            if assign[i]==2:continue;
            print(i)
            header = load_eeg_header(self.dirlist + self.files[i],verbose='CRITICAL', preload=True)
            trim_channels(header, self.channels)
            check_for_normalization(header)
            eeg = np.array(header.to_data_frame())
            eeg = split_eeg(eeg,30,100)
        #    print([str(files[i][-7:-5]), header.ch_names])
            # DO STUFF WITH DATA
            if assign[i] == 0:
                SleepDataset.train_data.extend(zip(eeg,self.hypno[i]))
            elif assign[i] == 1:
                SleepDataset.test_data.extend(zip(eeg,self.hypno[i]))
            else:
                print('should not happen')
           
        #    del 
            i = i + 1
        SleepDataset.loaded = True
        return SleepDataset.train_data, SleepDataset.test_data