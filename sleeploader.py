# -*- coding: utf-8 -*-
import os
import re
from tools import load_eeg_header, load_hypnogram, trim_channels,split_eeg, check_for_normalization
import numpy as np
import numpy.random as random
import tools
print('loaded SleepLoader.py')

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
  
    
class Singleton(object):
  _instances = {}
  def __new__(class_, *args, **kwargs):
    if class_ not in class_._instances:
        class_._instances[class_] = super(Singleton, class_).__new__(class_)#, *args, **kwargs)
        
    return class_._instances[class_]
    


   
class SleepDataset(Singleton):
    
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
        else:
            self.data = list()
            self.hypno = list()  
        self.dirlist = dirlist + '\\'
        return None
        
        
    def load_eeg(self,filename):
        header = load_eeg_header(self.dirlist + filename, verbose='CRITICAL', preload=True)
        trim_channels(header, self.channels)
        check_for_normalization(header)
        eeg = np.array(header.to_data_frame())
        eeg = split_eeg(eeg,30,100)
        return eeg
        
        
    def get_train(self, split=30, flat=True):
        """
        :param split: int 0-100, split ratio used for test set
        :param flat: select if data will be returned in a flat list or a list per subject
        """
        end = int(len(self.data)-len(self.data)*(split/100.0))
        if flat == True:
            return  [item for sublist in self.data[:end] for item in sublist], [item for sublist in self.hypno[:end] for item in sublist]
        else:
            return self.data[:end], self.hypno[:end]
        
        
    def get_test(self, split=30, flat=True):
        """
        :param split: int 0-100, split ratio used for test set
        :param flat: select if data will be returned in a flat list or a list per subject
        """
        start = int(len(self.data)-len(self.data)*(split/100.0))
        
        if flat == True:
            return  [item for sublist in self.data[start:] for item in sublist], [item for sublist in self.hypno[start:] for item in sublist]
        else:
            return self.data[start:], self.hypno[start:]
        
       
    def load(self, sel = [], shuffle = False,  force_reload = False, flat = True):
        """
        :param sel:          np.array with indices of files to load from the directory. Natural sorting.
        :param shuffle:      shuffle subjects or not
        :param force_reload: reload data even if already loaded
        :param flat:         select if data will be returned in a flat list or a list per subject
        """
        
        if self.loaded == True and force_reload == False and np.array_equal(sel, self.selection)==True:
            print('Getting Dataset')
            return [item for sublist in self.data for item in sublist], [item for sublist in self.hypno for item in sublist]
        elif force_reload==True:
            print('Reloading Dataset')
        else:
            print('Loading Dataset')   

        self.selection = sel    
        self.rng = random.RandomState(seed=23)
    
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
        self.hypno_files = map(self.hypno_files.__getitem__, sel)
        self.eeg_files   = map(self.eeg_files.__getitem__, sel)

        # shuffle if wanted
        if shuffle == True:
            self.hypno_files, self.eeg_files = tools.shuffle(self.hypno_files,self.eeg_files, random_state=self.rng)
            
        # load Hypno files
        for i in range(len(self.hypno_files)):
            self.hypno.append(load_hypnogram(self.dirlist + self.hypno_files[i]))
        
        # load EEG files
        for i in range(len(self.eeg_files)):
            eeg = self.load_eeg(self.eeg_files[i])
            if(len(eeg) != len(self.hypno[i])):
                print('WARNING: EEG epochs and Hypno epochs have different length {}:{}'.format(len(eeg),len(self.hypno[i])))
            self.data.append(eeg)
            
        self.loaded = True
        # select if data will be returned in a flat list or a list per subject
        if flat == True:
            return  [item for sublist in self.data for item in sublist], [item for sublist in self.hypno for item in sublist]
        else:
            return self.data,self.hypno