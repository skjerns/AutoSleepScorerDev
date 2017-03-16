# -*- coding: utf-8 -*-
"""
Spyder Editor

RFC confusion with handmade rules

"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os, sys
import scipy
import numpy as np
if not 'sleeploader' in vars() : import sleeploader  # prevent reloading module
import tools
import time
from sklearn import metrics
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 3)

try:
    with open('count') as f:
        counter = int(f.read())
except IOError:
    print('No previous experiment?')
    counter = 0
    
with open('count', 'w') as f:
  f.write(str(counter+1))        

#%%
#def main():
n_estimators = 100
chunk_size = 3000
future = 0
comment = 'RFC_corrupted_cv'

#%%

#for future in [0,2,5,10]
print(comment)
if os.name == 'posix':
    datadir  = '/home/simon/sleep/'
#    datadir  = '/home/simon/vinc/'
else:
#    datadir = 'c:\\sleep\\data\\'
#    datadir = 'C:\\sleep\\vinc\\brainvision\\'
    datadir = 'C:\\sleep\\corrupted\\'



sleep = sleeploader.SleepDataset(datadir)
sleep.load_object()
sleep.chunk_len = chunk_size
data, targets = sleep.get_all_data()

print('Extracting features')
data_feats = np.hstack( (tools.feat_eeg(data[:,:,0]), tools.feat_eog(data[:,:,1]),tools.feat_emg(data[:,:,2])))
del data
import gc; gc.collect()
#targets[targets==4] = 3
#targets[targets==5] = 4          
# normalize features
data_feats    = scipy.stats.mstats.zmap(data_feats, data_feats)
data_feats_rnd,targets_rnd = sklearn.utils.shuffle(data_feats, targets)

if np.sum(np.isnan(data_feats)) or np.sum(np.isnan(data_feats)):print('Warning! NaNs detected')
#%% training routine
print ('Starting first CV')
rf = RandomForestClassifier(200, n_jobs=3)
scores = []
scores_rnd = []
acc = []
acc_rnd = []
for i in np.arange(11):
    print i  
    sleep.load_hypnopickle('hypno' + str(i*10) + '.dat')
    data, targets = sleep.get_all_data()
    del data
    gc.collect()
    data_feats_rnd,targets_rnd = sklearn.utils.shuffle(data_feats, targets)
    scores.append (cross_val_score(rf, data_feats, targets, cv=5, n_jobs=1, scoring = 'f1_macro'))
    print str(i) + '.1'
    scores_rnd.append (cross_val_score(rf, data_feats_rnd, targets_rnd, cv=5, n_jobs=1, scoring = 'f1_macro'))
    print str(i) + '.2'
    acc.append (cross_val_score(rf, data_feats, targets, cv=5, n_jobs=1))
    print str(i) + '.3'
    acc_rnd.append (cross_val_score(rf, data_feats_rnd, targets_rnd, cv=5, n_jobs=1))
    print str(i) + '.4'
    print scores
    print scores_rnd
     
np.savez(comment + '.npz', scores=scores, scores_rnd=scores_rnd, acc=acc, acc_rnd=acc_rnd)
stop
#%%  Reporting and analysis.


ana = Analysis(ann.model,gpu=gpu, fname='results/{} {}-{}-{}-{}-{}-{}-{}'.format(counter,layers,neurons,epochs,clipping,decay,batch_size, comment))
start = time.time()

Y, T = ana.predict(training_data)
#Y = tools.epoch_voting(Y,30)
train_acc   = metrics.accuracy_score(Y,T)
train_f1    = metrics.f1_score(Y,T,average='macro')
train_conf   = ana.confusion_matrix(Y,T,plot=False)

Y, T = ana.predict(validation_data)
#Y = tools.epoch_voting(Y,30)
test_acc    = metrics.accuracy_score(Y,T)
test_f1     = metrics.f1_score(Y,T,average='macro')
classreport = metrics.classification_report(Y,T)
test_conf   = ana.confusion_matrix(Y,T,plot=True)
 
print("\nAccuracy: Test: {}%, Train: {}%".format("%.3f" % test_acc,"%.3f" % train_acc))
print("F1: \t  Test: {}%, Train: {}%".format("%.3f" % test_f1,"%.3f" % train_f1))
print(classreport)
print(test_conf)
print((time.time()-start) / 60.0)
#%%
# save results
train_loss = ann.log[('training','loss')][-1]
test_loss  = ann.log[('validation','loss')][-1]
np.set_printoptions(precision=2,threshold=np.nan)

save_dict = {'1 Time':time.strftime("%c"),
            'Dataset':datadir,
            'Runtime': "%.2f" % (runtime/60),
            '5 Layers':layers,
            '10 Neurons':neurons,
            '15 Epochs':epochs,
            'Clipping':clipping,
            'Weightdecay':decay,
            'Batch-Size':batch_size,
            'Cutoff':cutoff,
            '20 Train-Acc':"%.3f" % train_acc,
            '21 Val-Acc':"%.3f" % test_acc,
            'Train-Loss':"%.5f" %train_loss,
            'Val-Loss':"%.5f" %test_loss,
            '30 Comment':comment,
            'Selection':str(len(selection)) + ' in ' + str(selection) ,
            'Shape':str(train_data.shape) ,
            'Link':str(link),
            'Report':classreport,
            '22 Train-F1':"%.3f" %  train_f1,
            '27 Val-F1':"%.3f" %  test_f1,
            'Train-Confusion': str(train_conf),
            'Val-Confusion':  str(test_conf),
            'NLabels' : str(np.unique(T)),
            '29 Chunksize': str(chunk_size),
            '2 Number': counter,
            'Future': future
                      }

np.set_printoptions(precision=2,threshold=1000)
tools.append_json('experiments.json', save_dict)
tools.jsondict2csv('experiments.json', 'experiments.csv')
#return train_acc, test_acc
#if __name__ == "__main__":
#    main()
    
    