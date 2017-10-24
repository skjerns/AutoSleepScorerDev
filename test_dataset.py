# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os
import gc; gc.collect()
import matplotlib
matplotlib.use('Agg')
import numpy as np
import keras
import tools
import time
import models
from scipy.stats.mstats import zscore
import pickle
#import telegram_send
import keras_utils
import sleeploader
import matplotlib; matplotlib.rcParams['figure.figsize'] = (10, 3)
np.random.seed(42)

if __name__ == '__main__':
    import argparse
    import matplotlib;matplotlib.use('Agg')
    import numpy as np
    parser = argparse.ArgumentParser(description='Start calculating that dataset.')
    parser.add_argument('modus', metavar='Dataset', type=str, help='which data set to run')
    args = parser.parse_args()
    dataset = args.modus 
    print(dataset)
    plot = False if os.name == 'posix' else True
    try:
        time.sleep(np.random.rand()) # prevent access problems of parallel scripts
        with open('count') as f:
            counter = int(f.read())
    except IOError:
        print('No previous experiment?')
        counter = 0
    with open('count', 'w') as f:
      f.write(str(counter+1))        
    
    #%%

    def load_data(dataset):
        datadir = './data' if os.name == 'posix' else 'd:\\sleep\\data'
        sleep = sleeploader.SleepDataset(datadir)
        sleep.load_object(str(dataset))
        data, target, groups = sleep.get_all_data(groups=True)
        data = tools.normalize(data)
#        data = zscore(data,1)
        target[target==4] = 3
        target[target==5] = 4
        target[target>5] = 0
        
        target = keras.utils.to_categorical(target)
        return data, target, groups
    
    data,target,groups = load_data(dataset)

    #%%
    #s
#    data = data[:15000]
#    target = target[:15000]
#    groups = groups[:15000]
    
    batch_size = 256
    epochs = 256
    name =  dataset
    ###
    rnn = {'model':models.bi_lstm, 'layers': ['fc1'],  'seqlen':6,
           'epochs': 250,  'batch_size': 512,  'stop_after':15, 'balanced':False}
    print(rnn)
    model = models.cnn3adam_filter_morel2
    results = keras_utils.cv (data, target, groups, model, rnn=rnn, name=name,
                             epochs=epochs, folds=5, batch_size=batch_size, counter=counter,
                             plot=plot, stop_after=25, balanced=False, cropsize=2800)
    with open('results_dataset_{}_zscore.pkl'.format(dataset), 'wb') as f:
                pickle.dump(results, f)

#    telegram_send.send(parse_mode='Markdown',messages=['DONE {}  {} \n```\n{}\n```\n'.format(os.path.basename(__file__), dataset, tools.print_string(results))])
#
