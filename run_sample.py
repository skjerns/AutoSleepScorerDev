import os
import edfx_database
import sleeploader
import keras
import models
import pickle
import keras_utils
from scipy.stats.mstats import zscore
import numpy as np

#%%   ## download edfx database and prepare it
if __name__ == '__main__':
    datadir = 'edfx'
    
    # prepare dataset if it does not exist
    if not os.path.isfile(os.path.join(datadir, 'sleepdata.pkl')):
        edfx_database.download_edfx(datadir)
        edfx_database.convert_hypnograms(datadir)
    
        channels = {'EEG':'EEG FPZ-CZ', 'EMG':'EMG SUBMENTAL', 'EOG':'EOG HORIZONTAL'} # set channels that are used
        references = {'RefEEG':False, 'RefEMG':False, 'RefEOG':False} # we do not set a reference, because the data is already referenced
        sleep = sleeploader.SleepDataset(datadir)
        # use float16 is you have problems with memory or a small hard disk.  Should be around 2.6 GB for float32.
        sleep.load( channels = channels, references = references, verbose=0, dtype=np.float32) 
        edfx_database.truncate_eeg(sleep)

    # if the pickle file already exist, just load that one.
    else:
        sleep = sleeploader.SleepDataset(datadir)
        sleep.load_object() # load the prepared files. Should be around 2.6 GB for float32
    
    
    # load data
    data, target, groups = sleep.get_all_data(groups=True)
    data = zscore(data,1)
    
    target[target==4] = 3  # Set S4 to S3
    target[target==5] = 4  # Set REM to now empty class 4
    target = keras.utils.to_categorical(target)
    
    
#%%
    batch_size = 256
    epochs = 256
    ###
    rnn = {'model':models.bi_lstm, 'layers': ['fc1'],  'seqlen':6,
           'epochs': 250,  'batch_size': 512,  'stop_after':15, 'balanced':False}
    print(rnn)
    model = models.cnn3adam_filter_morel2
    results = keras_utils.cv (data, target, groups, model, rnn=rnn, name='edfx-sample',
                             epochs=epochs, folds=5, batch_size=batch_size, counter=0,
                             plot=True, stop_after=15, balanced=False, cropsize=2800)
    with open('results_dataset_edfx-sample.pkl', 'wb') as f:
                pickle.dump(results, f)