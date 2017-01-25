# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import numpy.random as random
import tools

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
    shuffle_index = list()
    subjects = list()
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

    def __init__(self, directory):
        """
        :param dirlist: a directory string. yes it´s not a list yet.
        """
        if self.loaded == True:
            print("Data already loaded. To reload add parameter force_reload=True")
        else:
            self.data = list()
            self.hypno = list()  
        if os.name != 'posix':
            self.directory = directory + '\\'
        else:
            self.directory = directory
        return None
        
        
    def load_eeg(self, filename, samp_per_epoch = 3000):
        """
        :param filename: loads the given eeg file
        """        
        header = tools.load_eeg_header(self.directory + filename, verbose='CRITICAL', preload=True)
        tools.trim_channels(header, self.channels)
        tools.check_for_normalization(header)
        eeg = np.array(header.to_data_frame())
        eeg = eeg[:(len(eeg)/samp_per_epoch)*samp_per_epoch]     # remove left over to ensure len(data)%3000==0
        eeg = tools.split_eeg(eeg,1,100)
        return eeg
        
        
    def shuffle_data(self):
        """
        Shuffle subjects that are loaded. Returns None
        """
        if self.loaded == False: print('ERROR: Data not loaded yet')
        self.data, self.hypno, self.shuffle_index, self.subjects = tools.shuffle(self.data, self.hypno, self.shuffle_index, self.subjects, random_state=self.rng)
        return None
        
        
    def get_subject(self, index):  
        """
        :param index: get subject [index] from loaded data. indexing from before shuffle is preserved
        """
        if self.loaded == False: print('ERROR: Data not loaded yet')
        return self.data[self.shuffle_index.index(index)], self.hypno[self.shuffle_index.index(index)] # index.index(index), beautiful isn't it?? :)
        
    def get_all_data(self, flat=True):
        """
        returns all data that is loaded
        :param flat: select if data will be returned in a flat list or a list per subject
        """
    
        if self.loaded == False: print('ERROR: Data not loaded yet')
            
        if flat == True:
            return  [item for sublist in self.data for item in sublist], [item for sublist in self.hypno for item in sublist]
        else:
            return self.data, self.hypno
        
        
    def get_train(self, split=30, flat=True):
        """
        :param split: int 0-100, split ratio used for test set
        :param flat: select if data will be returned in a flat list or a list per subject
        """
    
        if self.loaded == False: print('ERROR: Data not loaded yet')
            
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
        
        if self.loaded == False: print('ERROR: Data not loaded yet')
            
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
            if shuffle == True:
                self.shuffle_data()
            if flat == True:
                return [item for sublist in self.data for item in sublist], [item for sublist in self.hypno for item in sublist]
            else:
                return self.data,self.hypno    
            
        elif force_reload==True:
            print('Reloading Dataset')
            
        else:
            print('Loading Dataset') 
            
        self.data = list()
        self.hypno = list()  
        self.selection = sel    
        self.rng = random.RandomState(seed=23)
    
        # check hypno_filenames
        self.hypno_files = [s for s in os.listdir(self.directory) if s.endswith('.txt')]
        self.hypno_files = sorted(self.hypno_files, key = natural_key)

        # check eeg_filenames
        self.eeg_files = [s for s in os.listdir(self.directory) if s.endswith(('.vhdr','rec','edf'))]
        self.eeg_files = sorted(self.eeg_files, key = natural_key)
        
        if len(self.hypno_files) != len(self.eeg_files): 
            print('ERROR: Not the same number of Hypno and EEG files. Hypno: ' + str(len(self.hypno_files))+ ', EEG: ' + str(len(self.eeg_files)))
            
        # select slice
        if sel==[]: sel = range(len(self.hypno_files))
        self.hypno_files = map(self.hypno_files.__getitem__, sel)
        self.eeg_files   = map(self.eeg_files.__getitem__, sel)
        self.shuffle_index = list(sel);
        self.subjects = zip(self.eeg_files,self.hypno_files)

        # load Hypno files
        for i in range(len(self.hypno_files)):
            self.curr_hypno = tools.load_hypnogram(self.directory + self.hypno_files[i])
            self.curr_hypno = np.repeat(self.curr_hypno,30)
            self.hypno.append(self.curr_hypno)
            
        # load EEG files
        for i in range(len(self.eeg_files)):
            eeg = self.load_eeg(self.eeg_files[i], samp_per_epoch=3000)
            if(len(eeg) != len(self.hypno[i])):
                print('WARNING: EEG epochs and Hypno epochs have different length {}:{}'.format(len(eeg),len(self.hypno[i])))
            self.data.append(eeg)
            
        self.loaded = True
        
        # shuffle if wanted
        if shuffle == True:
            self.shuffle_data()
            
        

        # select if data will be returned in a flat list or a list per subject
        if flat == True:
            return  [item for sublist in self.data for item in sublist], [item for sublist in self.hypno for item in sublist]
        else:
            return self.data,self.hypno
            
print('loaded sleeploader.py')