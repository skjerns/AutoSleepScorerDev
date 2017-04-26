# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:53:58 2017

## this script does cross validation on the RFC and the CNN
## here I test how future outlook and past epochs improve training

@author: Simon
"""


if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/ANN')
import numpy as np
import lasagne
from tqdm import tnrange, tqdm
from lasagne import layers as L
from theano import tensor as T
import theano
#from custom_analysis import Analysis
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
if not 'sleeploader' in vars() : import sleeploader  # prevent reloading module
import tools
import sklearn
from sklearn import metrics
from sklearn.preprocessing import scale
#import matplotlib
import trainer

#matplotlib.rcParams['figure.figsize'] = (10, 3)
import scipy
try:
    with open('count') as f:
        counter = int(f.read())
except IOError:
    print('No previous experiment?')
    counter = 0
     
with open('count', 'w') as f:
  f.write(str(counter+1))      
  
if os.name == 'posix':
    datadir  = '/home/simon/sleep/'
#    datadir  = '/home/simon/vinc/'
#    datadir = '/media/simon/Windows/sleep/corrupted'
    datadir = '../'

else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'C:\\sleep\\vinc\\brainvision\\correct\\'
    datadir = 'd:\\sleep\\corrupted\\'

sleep = sleeploader.SleepDataset(datadir)
sleep.load_object()
data, targets, groups = sleep.get_all_data(groups=True)
del sleep.data
# normalize
data    = scipy.stats.mstats.zmap(data, data , axis = None)
data    = tools.future(data,6)

data = data.swapaxes(1,2)
targets[targets==5] = 4   

print('Extracting features')
feats = np.hstack( (tools.feat_eeg(data[:,:,0]), tools.feat_eog(data[:,:,1]),tools.feat_emg(data[:,:,2])))
feats = tools.future(feats,6)


if np.sum(np.isnan(data)):print('Warning! NaNs detected')
#%%
def CNN(data_size, n_classes, input_var):
    network = L.InputLayer(shape=(None, data_size[1], data_size[2]), input_var=input_var)
    print network.output_shape
    network = L.Conv1DLayer( network, num_filters=100, filter_size = 50, stride = 10, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
    network = L.batch_norm(network)
    network = L.DropoutLayer(network, p=0.2)
    network = L.Conv1DLayer( network, num_filters=150, filter_size = 5, stride = 1, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
    network = L.DropoutLayer(network, p=0.2)
    network = L.batch_norm(network)
    network = L.MaxPool1DLayer(network, pool_size = (5), stride = (2))

    network = L.Conv1DLayer( network, num_filters=150, filter_size = 5, stride = 2, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
    network = L.DropoutLayer(network, p=0.2)
    network = L.batch_norm(network)
    network = L.MaxPool1DLayer(network, pool_size = (5), stride = (2))
    
    print network.output_shape
    network = L.DenseLayer( network, num_units=500, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
    network = L.batch_norm(network)
    network = L.DropoutLayer(network, p=0.5)
    network = L.DenseLayer( network, num_units=500, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
    network = L.batch_norm(network)
    network = L.DropoutLayer(network, p=0.5)
    print network.output_shape

    network = L.DenseLayer( network, num_units=n_classes,W=lasagne.init.GlorotNormal(),      nonlinearity=lasagne.nonlinearities.softmax)
    return network

def training_function(network, input_tensor, target_tensor, learning_rate, use_l2_regularization=False):
    network_output = L.get_output(network)
    if use_l2_regularization:
        l2_loss = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean() + (l2_loss * 0.0001)
    else:
        loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean()  
    pred = T.argmax(network_output, axis=1)
    network_params  = L.get_all_params(network, trainable=True)
    weight_updates  = lasagne.updates.adadelta(loss, network_params)
    return theano.function([input_tensor, target_tensor], [loss, pred], updates=weight_updates)


def validate_function(network, input_tensor, target_tensor):
    network_output = L.get_output(network, deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(network_output, target_tensor).mean() 
#     accuracy = T.mean(T.eq(T.argmax(network_output, axis=1), target_tensor), dtype=theano.config.floatX)
    pred = T.argmax(network_output, axis=1)
    return theano.function([input_tensor, target_tensor], [loss, pred])


def evaluate_function(network, input_tensor):
    network_output = lasagne.layers.get_output(network, deterministic=True)
    return theano.function([input_tensor], network_output)





#%% Training routine for CNN

n_classes = len(np.unique(targets))
input_var  = T.tensor3('inputs')
target_var = T.ivector('targets')
batch_size = 1280
epochs = 100
network_name = 'CNN'


folds = 5
from sklearn.ensemble import RandomForestClassifier

## one electrode
print ('Starting CNN cv 1/2')
accs_cnn_fut = []
f1_cnn_fut   = []
cv = sklearn.model_selection.GroupKFold(folds)
for train_idx, test_idx in cv.split(groups, groups, groups):
    
    sub_group = groups[train_idx]
    sub_cv = sklearn.model_selection.GroupKFold(folds)
    train_sub_idx, val_idx = sub_cv.split(groups[train_idx], groups[train_idx], groups[train_idx]).next()
    val_idx      = train_idx[val_idx]  
    train_idx    = train_idx[train_sub_idx]
    train_data   = data[train_idx]
    train_target = targets[train_idx]
    print 'now val'
    val_data     = data[val_idx]
    val_target   = targets[val_idx]
    print 'now test'
    test_data    = data[test_idx]       
    test_target  = targets[test_idx]

    network = CNN(train_data.shape, n_classes, input_var)
    train_fn = training_function(network, input_var, target_var, 0.001)
    val_fn =  validate_function(network, input_var, target_var)
    f1, acc = trainer.train(network_name, network, train_fn, val_fn, train_data, train_target, val_data, 
                   val_target, test_data,test_target, epochs=50, batch_size=128*3)
    
    accs_cnn_fut.append(acc)
    f1_cnn_fut.append(f1)
    del train_data, val_data, test_data
    
print('{}/{}'.format(np.mean(accs_cnn_fut),np.mean(f1_cnn_fut)))

##
### all electrode
#print ('Starting CNN cv 2/2')
#accs_cnn_all = []
#f1_cnn_all   = []
#cv = sklearn.model_selection.GroupKFold(folds)
#for train_idx, test_idx in cv.split(groups, groups, groups):
#    
#    sub_group = groups[train_idx]
#    sub_cv = sklearn.model_selection.GroupKFold(folds)
#    train_sub_idx, val_idx = sub_cv.split(groups[train_idx], groups[train_idx], groups[train_idx]).next()
#    val_idx      = train_idx[val_idx]  
#    train_idx    = train_idx[train_sub_idx]
#    
#    train_data   = data[train_idx]
#    train_target = targets[train_idx]
#    val_data     = data[val_idx]
#    val_target   = targets[val_idx]
#    test_data    = data[test_idx]       
#    test_target  = targets[test_idx]
#    
#    network = CNN(train_data.shape, n_classes, input_var)
#    train_fn = training_function(network, input_var, target_var, 0.001)
#    val_fn =  validate_function(network, input_var, target_var)
#    f1, acc = trainer.train(network_name, network, train_fn, val_fn, train_data, train_target, val_data, 
#                   val_target, test_data,test_target, epochs=50, batch_size=1024)
#    
#    accs_cnn_all.append(acc)
#    f1_cnn_all.append(f1)
#print('{}/{}'.format(np.mean(accs_cnn_all),np.mean(f1_cnn_all)))
#%% Training routine for RFC

del data
feats = tools.future(feats,6)
folds = 5
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

## one electrode
print ('Starting RFC cv 1/2')
accs_rfc_fut = []
f1_rfc_fut   = []
cv = sklearn.model_selection.GroupKFold(folds)
for train_idx, test_idx in tqdm(cv.split(feats, targets, groups), total = folds):
    clf.fit(feats[train_idx], targets[train_idx])
    preds = clf.predict(feats[test_idx])
    f1  = f1_score(preds, targets[test_idx], average = 'macro' )
    acc = accuracy_score(preds, targets[test_idx])
    accs_rfc_fut.append(acc)
    f1_rfc_fut.append(f1)
print('{}/{}'.format(np.mean(accs_rfc_fut),np.mean(f1_rfc_fut)))
## all electrodes
#print ('Starting RFC cv 2/2')
#accs_rfc_all = []
#f1_rfc_all   = []
#cv = sklearn.model_selection.GroupKFold(folds)
#for train_idx, test_idx in tqdm(cv.split(data, targets, groups), total = folds):
#    clf.fit(feats[train_idx], targets[train_idx])
#    preds = clf.predict(feats[test_idx])
#    f1  = f1_score(preds, targets[test_idx], average = 'macro' )
#    acc = accuracy_score(preds, targets[test_idx])
#    accs_rfc_all.append(acc)
#    f1_rfc_all.append(f1)
#print('{}/{}'.format(np.mean(accs_rfc_all),np.mean(f1_rfc_all)))


np.savez(os.path.join('./' , 'test_future.npz'), accs_cnn_fut=accs_cnn_fut, f1_cnn_fut=f1_cnn_fut,
                                                     accs_rfc_fut=accs_rfc_fut, f1_rfc_fut=f1_rfc_fut )