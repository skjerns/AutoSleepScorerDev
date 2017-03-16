# -*- coding: utf-8 -*-
"""
Spyder Editor

Comparing to the RFC baseline

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
comment = 'RFC_3000_rng_svm'

#%%

#for future in [0,2,5,10]
print(comment)
if os.name == 'posix':
    datadir  = '/home/simon/sleep/'
#    datadir  = '/home/simon/vinc/'
else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'C:\\sleep\\vinc\\brainvision\\'
#    datadir = 'C:\\sleep\\corrupted\\'



sleep = sleeploader.SleepDataset(datadir)
sleep.load_object("adults.dat")
sleep.chunk_len = chunk_size
data, targets = sleep.get_all_data()
sleep.load_object("children.dat")
sleep.chunk_len = chunk_size
childdata, childtargets = sleep.get_all_data()

print('Extracting features')
data_feats = np.hstack( (tools.feat_eeg(data[:,:,0]), tools.feat_eog(data[:,:,1]),tools.feat_emg(data[:,:,2])))
childdata  = np.hstack( (tools.feat_eeg(childdata[:,:,0]), tools.feat_eog(childdata[:,:,1]),tools.feat_emg(childdata[:,:,2])))
# reducing classes
del data
targets[targets==4] = 3
targets[targets==5] = 4          
data_feats   = np.delete(data_feats, np.where(targets==8) ,axis=0)     
targets      = np.delete(targets, np.where(targets==8) ,axis=0)

childtargets[childtargets==4] = 3
childtargets[childtargets==5] = 4          
childdata         = np.delete(childdata, np.where(childtargets==8) ,axis=0)     
childtargets      = np.delete(childtargets, np.where(childtargets==8) ,axis=0)

# normalize features
childdata     = scipy.stats.mstats.zmap(childdata, data_feats)
data_feats    = scipy.stats.mstats.zmap(data_feats, data_feats)

childdata, childtargets = sklearn.utils.shuffle(childdata, childtargets)
data_feats,targets = sklearn.utils.shuffle(data_feats, targets)

if np.sum(np.isnan(data_feats)) or np.sum(np.isnan(data_feats)):print('Warning! NaNs detected')
#%% training routine
print ('Starting first CV')
rf = RandomForestClassifier(250, n_jobs=1)

scores    = cross_val_score(rf, data_feats, targets, cv=5, n_jobs=1)
scores_ch = cross_val_score(rf, data_feats, targets, cv=5, n_jobs=1)

rf.fit(childdata, childtargets)
Y = rf.predict(data_feats)
ch_acc = rf.score(data_feats, targets)
chscore = metrics.f1_score(targets, Y, average = 'macro')

#np.savez(comment + '.npz', scores=scores,scores_ch=scores_ch,ch_acc=ch_acc,ch_f1=chscore)
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
    
    