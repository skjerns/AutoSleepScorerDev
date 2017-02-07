# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import numpy.random as random
import tools
import csv
import mne

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
    
    
    def check_for_normalization(self,data_header):
    
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
        
        
    def load_hypnogram(self, filename, dataformat = '', csv_delimiter='\t'):
        """
        returns an array with sleep stages
        :param filename: loads the given hypno file
        """
        dataformats = dict({
                            'txt' :'csv',
                            'csv' :'csv',                                           
                            })
        if dataformat == '' :      # try to guess format by extension 
            ext = os.path.splitext(filename)[1][1:].strip().lower()                
            dataformat = dataformats[ext]
            
        if dataformat == 'csv':
            with open(filename) as csvfile:
                reader = csv.reader(csvfile, delimiter=csv_delimiter)
                lhypno = []
                for row in reader:
                    lhypno.append(int(row[0]))
        else:
            print('unkown hypnogram format. please use CSV with rows as epoch')        

        lhypno = np.array(lhypno).reshape(-1, 1)
        return lhypno   
        
    
    def load_eeg_header(self,filename, dataformat = '', **kwargs):            # CHECK include kwargs
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
    
    
    def trim_channels(self,data, channels):
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
    #            
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

    def load_eeg_hypno(self, eeg_file, hypno_file, chuck_size = 3000):
        """
        :param filename: loads the given eeg file
        """
        hypno  = self.load_hypnogram(self.directory + hypno_file)
        header = self.load_eeg_header(self.directory + eeg_file, verbose='CRITICAL', preload=True)
        self.trim_channels(header, self.channels)
        self.check_for_normalization(header)
        eeg = np.array(header.to_data_frame())
        
        sfreq = header.info['sfreq']
        hypno_length = len(hypno)
        eeg_length   = len(eeg)
        
        epoch_len = int(eeg_length / hypno_length / sfreq) 
        samples_per_epoch = int(epoch_len * sfreq)
        hypno_repeat = int(samples_per_epoch / chuck_size)
        
        hypno = np.repeat(hypno,hypno_repeat)
        eeg = eeg[:(len(eeg)/samples_per_epoch)*samples_per_epoch]     # remove left over to ensure len(data)%3000==0
        eeg = tools.split_eeg(eeg,chuck_size)
        return eeg, hypno
        
        
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
        
       
    def load(self, sel = [], shuffle = False,  force_reload = False, flat = True, chunk_size = 3000):
        """
        :param sel:          np.array with indices of files to load from the directory. Natural sorting.
        :param shuffle:      shuffle subjects or not
        :param force_reload: reload data even if already loaded
        :param flat:         select if data will be returned in a flat list or a list per subject
        """
        
        if self.loaded == True and chunk_size==self.chunk_size and force_reload == False and np.array_equal(sel, self.selection)==True:
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
        self.chunk_size = chunk_size
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
        self.hypno_files = list(map(self.hypno_files.__getitem__, sel))
        self.eeg_files   = list(map(self.eeg_files.__getitem__, sel))
        self.shuffle_index = list(sel);
        self.subjects = zip(self.eeg_files,self.hypno_files)

           
        # load EEG and adapt Hypno files
        for i in range(len(self.eeg_files)):
            eeg, curr_hypno = self.load_eeg_hypno(self.eeg_files[i], self.hypno_files[i], chunk_size)
            if(len(eeg) != len(curr_hypno)):
                print('WARNING: EEG epochs and Hypno epochs have different length {}:{}'.format(len(eeg),len(curr_hypno)))
            self.data.append(eeg)
            self.hypno.append(curr_hypno)
            
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