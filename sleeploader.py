# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import numpy.random as random
import tools
from scipy.signal import resample
import csv
import cPickle
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
                'RightEye':'EOG',
                'EMG' :'EMG',
                'EEG' :'EEG',
                'C3'  :'EEG',
                'C3A2'  :'EEG',
                'C4'  :'EEG',
                'C3A2':'EEG',
                'EEG1':'EEG',
                'EEG2':'EEG',
                'EMG1':'EMG',
                'LOC' :'EOG'})

    def __init__(self, directory):
        """
        :param directory: a directory string. yes it´s not a list yet.
        """
        if self.loaded == True:
            print("Data already loaded.")
        else:
            self.data = list()
            self.hypno = list()  
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
        
        
    def load_hypnogram(self, filename, dataformat = '', csv_delimiter='\t', mode='standard'):
        """
        returns an array with sleep stages
        :param filename: loads the given hypno file
        :param mode: standard: just read first row, overwrite = if second row!=0,
                     take that value, concatenate = add values together
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
                    if mode == 'standard':
                        lhypno.append(int(row[0]))
                        
                    elif mode == 'overwrite':
                        if int(row[1]) == 0:
                            lhypno.append(int(row[0]))
                        else:
                            lhypno.append(8)
                            #lhypno.append(int(row[1]))
                            
                    elif mode == 'concatecate':
                        lhypno.append(int(x) for x in row)
        else:
            print('unkown hypnogram format. please use CSV with rows as epoch')        

        lhypno = np.array(lhypno, dtype=np.int32).reshape(-1, 1)
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
    
    
    def load_hypnopickle(self, filename, path = None):
        """
        loads hypnograms from a pickle file
        """
        if path == None: path = self.directory
        with open(os.path.join(path, filename), 'rb') as f:
            self.hypno, self.hypno_files = cPickle.load(f)
            self.subjects = zip(self.eeg_files,self.hypno_files)
            if len(self.hypno) != len(self.data): 
                print('WARNING: {} EEG files and {} Hypno files'.format(len(self.eeg_files),len(self.hypno)))
            else:
                for i in np.arange(len(self.data)):
                    if len(self.data[i])/ self.samples_per_epoch != len(self.hypno[i]):
                        print('WARNING, subject {} has EEG len {} and Hypno len {}'.format(i, len(self.data[i])/ self.samples_per_epoch,len(self.hypno[i])))               
            print ('Loaded hypnogram with {} subjects'.format(len(self.hypno)))
        
        
    def save_hypnopickle(self, filename, path = None):
        """
        saves the current hypnograms to a pickle file
        """
        if path == None: path = self.directory
        with open(os.path.join(path, filename), 'wb') as f:
            cPickle.dump((self.hypno,self.hypno_files),f,2)
        
    
    def load_object(self, filename = 'sleepdata.dat', path = None):
        """
        saves the entire state of the SleepData object
        """
        if path == None: path = self.directory
        with open(os.path.join(path, filename), 'rb') as f:
            tmp_dict = cPickle.load(f)
        self.__dict__.update(tmp_dict)


    def save_object(self, filename = 'sleepdata.dat', path = None):
        """
        restores a previously stored SleepData object
        """
        if path == None: path = self.directory
        with open(os.path.join(path, filename), 'wb') as f:
            cPickle.dump(self.__dict__,f,2)
    
    def load_hypno_(self, files):
        self.hypno = []
        self.hypno_files = files
        for f in files:
            hypno  = self.load_hypnogram(os.path.join(self.directory + f), mode = 'overwrite')
            self.hypno.append(hypno)


    def load_eeg_hypno(self, eeg_file, hypno_file, chuck_size = 3000, resampling = True):
        """
        :param filename: loads the given eeg file
        """
        
        hypno  = self.load_hypnogram(self.directory + hypno_file, mode = 'standard')
        header = self.load_eeg_header(self.directory + eeg_file, verbose='WARNING', preload=True)
        self.trim_channels(header, self.channels)
        self.check_for_normalization(header)
        
        mne.set_log_level(verbose=False)  # to get rid of the annoying 'convert to float64'
        eeg = np.array(header.to_data_frame().reindex_axis(['EEG','EOG','EMG'],axis=1), dtype=self.dtype)
        mne.set_log_level(verbose=True)
        
        self.sfreq     = header.info['sfreq']
        if not header.info['sfreq'] == 100:
            if resampling == True:
                print ('Resampling data from {}hz to 100hz'.format(int(self.sfreq)))
                eeg = resample(eeg, len(eeg)/int(np.round(self.sfreq))*100)
                self.sfreq = 100.0 
            else:
                print ('Not resampling')
        
        hypno_len = len(hypno)
        eeg_len   = len(eeg)
        epoch_len = int(eeg_len / hypno_len / self.sfreq) 
        self.samples_per_epoch = int(epoch_len * self.sfreq)
        

        eeg = eeg[:(len(eeg)/self.samples_per_epoch)*self.samples_per_epoch]     # remove left over to ensure len(data)%3000==0
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
        
    
    def get_all_data(self, flat=True, groups = False):
        """
        returns all data that is loaded
        :param flat: select if data will be returned in a flat list or a list per subject
        """
    
        if self.loaded == False: print('ERROR: Data not loaded yet')
            
        if flat == True:
            return  self._makeflat(groups=groups)
        else:
            return self.data, self.hypno
        
        
        
    def get_intrasub(self, splits=3, test=1 ):  
        train_data = list()
        train_target = list()
        test_data = list()
        test_target = list()
        hypno_repeat = self.samples_per_epoch / self.chunk_len
        
        for eeg, hypno in zip(self.data, self.hypno):
            per_split = len(eeg)/self.chunk_len/splits
            choice = self.rng.permutation(splits)
            tst = choice[0:test]  
            trn = choice[test:]
            for i in trn:
                train_data.append(eeg.reshape([-1, self.chunk_len,3])[per_split*i:(per_split*(i+1))])
                train_target.append(np.repeat(hypno, hypno_repeat)[per_split*i:(per_split*(i+1))])
            for i in tst:
                test_data.append(eeg.reshape([-1, self.chunk_len,3])[per_split*i:(per_split*(i+1))])
                test_target.append(np.repeat(hypno, hypno_repeat)[per_split*i:(per_split*(i+1))])
        return np.vstack(train_data), np.hstack(train_target), np.vstack(test_data), np.hstack(test_target)
    
    
        
    def get_train(self, split=30, flat=True):
        """
        :param split: int 0-100, split ratio used for test set
        :param flat: select if data will be returned in a flat list or a list per subject
        """
    
        if self.loaded == False: print('ERROR: Data not loaded yet')
            
        end = int(len(self.data)-len(self.data)*(split/100.0))
        
        if flat == True:
            return  self._makeflat(end=end)
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
             return  self._makeflat(start=start)
        else:
            return self.data[start:], self.hypno[start:]
        
        
    def _makeflat(self, start=None, end=None, groups = False):     
        eeg = list()
        for sub in self.data[start:end]:
            if len(sub) % self.chunk_len == 0:
                eeg.append(sub.reshape([-1, self.chunk_len,3]))
            else:
                print('ERROR: Please choose a chunk length that is a factor of {}'.format(self.samples_per_epoch))
                return [0,0]
        hypno = list()
        group = list()
        hypno_repeat = self.samples_per_epoch / self.chunk_len
        idx = 0
        for sub in self.hypno[start:end]:
            hypno.append(np.repeat(sub, hypno_repeat))
            group.append(np.repeat(idx, len(hypno[-1])))
            idx += 1
        
        if groups:
            return np.vstack(eeg), np.hstack(hypno), np.hstack(group)
        else:
            return np.vstack(eeg), np.hstack(hypno)
       
        
    def load(self, sel = [], shuffle = False,  force_reload = False, resampling =True, flat = None, chunk_len = 3000, dtype=np.float32):
        """
        :param sel:          np.array with indices of files to load from the directory. Natural sorting.
        :param shuffle:      shuffle subjects or not
        :param force_reload: reload data even if already loaded
        :param flat:         select if data will be returned in a flat array or a list per subject
        """
        
        self.chunk_len = chunk_len        
        if self.loaded == True and force_reload == False and np.array_equal(sel, self.selection)==True:
            print('Getting Dataset')
            if shuffle == True:
                self.shuffle_data()
            if flat == True:
                return self._makeflat()
            elif flat == False:
                return self.data,self.hypno   
            else:
                print('No return mode set. Just setting new chunk_len')
                return
            
        elif force_reload==True:
            print('Reloading Dataset')
            
        else:
            print('Loading Dataset') 
            
        self.dtype = dtype   
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
        
        if len(self.hypno_files)  != len(self.eeg_files): 
            print('ERROR: Not the same number of Hypno and EEG files. Hypno: ' + str(len(self.hypno_files))+ ', EEG: ' + str(len(self.eeg_files)))
            
        # select slice
        if sel==[]: sel = range(len(self.eeg_files))
        self.hypno_files = list(map(self.hypno_files.__getitem__, sel))
        self.eeg_files   = list(map(self.eeg_files.__getitem__, sel))
        self.shuffle_index = list(sel);
        self.subjects = zip(self.eeg_files,self.hypno_files)

        # load EEG and adapt Hypno files
        for i in range(len(self.eeg_files)):
            eeg, curr_hypno = self.load_eeg_hypno(self.eeg_files[i], self.hypno_files[i], chunk_len, resampling)
            if(len(eeg) != len(curr_hypno) * self.samples_per_epoch):
                print('WARNING: EEG epochs and Hypno epochs have different length {}:{}'.format(len(eeg),len(curr_hypno)* self.samples_per_epoch))
            self.data.append(eeg)
            self.hypno.append(curr_hypno)
            
        self.loaded = True
        
        # shuffle if wanted
        if shuffle == True:
            self.shuffle_data()
            
        # select if data will be returned in a flat array or a list per subject
        if flat == True:
            return  self._makeflat()
        elif flat == False:
            return self.data,self.hypno
            
print('loaded sleeploader.py')



