# -*- coding: utf-8 -*-
import os
import re
from loader import load_eeg_header, load_hypnogram, trim_channels,split_eeg, check_for_normalization
import numpy as np
import numpy.random as random
print('loader')
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    
        
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
                'C3A2':'EEG',
                'EEG1':'EEG',
                'EEG2':'EEG'})
    
    def __init__(self, dirlist):
#        print(loaded)
        """
        :param dirlist: a directory string. yes it´s not a list yet.
        """
        if self.loaded == True:
            print("Data already loaded. To reload add parameter force_reload=True")
        self.dirlist = dirlist
        return None
            
       
    def load(self, sel = [], shuffle = True,  force_reload = False):

        if self.loaded == True and force_reload == False:
            return self.train_data, self.test_data
        else:
            print('Reloading DataSet')
        self.rnd = random.RandomState(seed=23)
        
        
        
        # check hypno_filenames
        self.hypno_files = [s for s in os.listdir(self.dirlist) if s.endswith('.txt')]
        self.hypno_files = sorted(self.hypno_files, key = natural_key)

        # check eeg_filenames
        self.eeg_files = [s for s in os.listdir(self.dirlist) if s.endswith(('.vhdr','rec','edf'))]
        self.eeg_files = sorted(self.eeg_files, key = natural_key)
        
        if len(self.hypno_files) != len(self.eeg_files): 
            print('ERROR: Not the same number of Hypno and EEG files. Hypno: ' + str(len(self.hypno_files))+ ', EEG: ' + str(len(self.eeg_files)))
            
        # select slice
        if sel==[]: sel = range(len(self.hypno_files))
        print(sel)
        self.hypno_files = map(self.hypno_files.__getitem__, sel)
        self.eeg_files = map(self.eeg_files.__getitem__, sel)
#        self.hypno_files = self.hypno_files[sel]
#        self.eeg_files   = self.eeg_files[sel]
        
        # shuffle if wanted
        if shuffle == True:
            self.shuffle_list = list(zip(self.hypno_files, self.eeg_files))
            self.rnd.shuffle(self.shuffle_list)
            self.hypno_files, self.eeg_files = zip(*self.shuffle_list)
        print('Hypno files')
        # load Hypno files
        self.hypno = list()
        for i in range(len(self.hypno_files)):
            self.hypno.append(load_hypnogram(self.dirlist + self.hypno_files[i]))
        
        print('EEG files')
        self.assign = np.array([0]*(len(self.hypno_files)-len(self.hypno_files)/3) + [1]*(len(self.hypno_files)/3))
        # load EEG files
        for i in range(len(self.eeg_files)):
            print(i)
            header = load_eeg_header(self.dirlist + self.eeg_files[i],verbose='CRITICAL', preload=True)
            trim_channels(header, self.channels)
            check_for_normalization(header)
            eeg = np.array(header.to_data_frame())
            eeg = split_eeg(eeg,30,100)
            # DO STUFF WITH DATA
            if(len(eeg) != len(self.hypno[i])):
                print('WARNING: EEG epochs and Hypno epochs have different length {}:{}'.format(len(eeg),len(self.hypno[i])))
            if self.assign[i] == 0:
                SleepDataset.train_data.extend(zip(eeg,self.hypno[i]))
            elif self.assign[i] == 1:
                SleepDataset.test_data.extend(zip(eeg,self.hypno[i]))
            else:
                print('should not happen')
           
        SleepDataset.loaded = True
        return SleepDataset.train_data, SleepDataset.test_data