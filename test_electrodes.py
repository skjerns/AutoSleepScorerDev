# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 13:53:58 2017

## this script does cross validation on the RFC and the CNN
## here I test how much better several electrodes are compared to one.

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
import matplotlib
import trainer

matplotlib.rcParams['figure.figsize'] = (10, 3)
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
    datadir = '/media/simon/Windows/sleep/corrupted'

else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'C:\\sleep\\vinc\\brainvision\\correct\\'
    datadir = 'C:\\sleep\\corrupted\\'

sleep = sleeploader.SleepDataset(datadir)
sleep.load_object()
data, targets, groups = sleep.get_all_data(groups=True)
# normalize
data    = scipy.stats.mstats.zmap(data, data , axis = None)

print('Extracting features')
#train_data = np.hstack( (tools.feat_eeg(train_data[:,:,0]), tools.feat_eog(train_data[:,:,1]),tools.feat_emg(train_data[:,:,2])))
feats = np.hstack( (tools.feat_eeg(data[:,:,0]), tools.feat_eog(data[:,:,1]),tools.feat_emg(data[:,:,2])))
targets[targets==5] = 4   
test_data    = np.delete(data, np.where(targets==8) ,axis=0)     
test_target  = np.delete(targets, np.where(targets==8) ,axis=0) 
data = data.swapaxes(1,2)

if np.sum(np.isnan(data)):print('Warning! NaNs detected')

#%%
def CNN(data_size, n_classes, input_var):
    network = L.InputLayer(shape=(None, data_size[1], data_size[2]), input_var=input_var)
    print network.output_shape
    network = L.Conv1DLayer( network, num_filters=50, filter_size = 50, stride = 10, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
    network = L.batch_norm(network)
    network = L.DropoutLayer(network, p=0.2)
    network = L.Conv1DLayer( network, num_filters=100, filter_size = 5, stride = 1, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
    network = L.DropoutLayer(network, p=0.2)
    network = L.batch_norm(network)
    network = L.MaxPool1DLayer(network, pool_size = (5), stride = (2))
    
    network = L.Conv1DLayer( network, num_filters=100, filter_size = 5, stride = 2, W=lasagne.init.HeNormal(), nonlinearity=lasagne.nonlinearities.elu)
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



#%% Training routine for RFC
folds = 5
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=250, n_jobs=3)

## one electrode
print ('Starting RFC cv 1/2')
accs_rfc_one = []
f1_rfc_one   = []
cv = sklearn.model_selection.GroupKFold(folds)
for train_idx, test_idx in tqdm(cv.split(data, targets, groups), total = folds):
    clf.fit(feats[train_idx, 0:8], targets[train_idx])
    preds = clf.predict(feats[test_idx, 0:8])
    f1  = f1_score(preds, targets[test_idx], average = 'macro' )
    acc = accuracy_score(preds, targets[test_idx])
    accs_rfc_one.append(f1)
    f1_rfc_one.append(acc)
print('{}/{}'.format(np.mean(accs_rfc_one),np.mean(f1_rfc_one)))
## all electrodes
print ('Starting RFC cv 2/2')
accs_rfc_all = []
f1_rfc_all   = []
cv = sklearn.model_selection.GroupKFold(folds)
for train_idx, test_idx in tqdm(cv.split(data, targets, groups), total = folds):
    clf.fit(feats[train_idx], targets[train_idx])
    preds = clf.predict(feats[test_idx])
    f1  = f1_score(preds, targets[test_idx], average = 'macro' )
    acc = accuracy_score(preds, targets[test_idx])
    accs_rfc_all.append(f1)
    f1_rfc_all.append(acc)
print('{}/{}'.format(np.mean(accs_rfc_all),np.mean(f1_rfc_all)))


#%% Training routine for CNN

n_classes = len(np.unique(targets))
input_var  = T.tensor3('inputs')
target_var = T.ivector('targets')
batch_size = 1280
epochs = 100
network_name = 'CNN'


#aa = trainer.train('test', network,train_fn, val_fn, train_data, train_target, test_data, 
#                   test_target, test_data,test_target, epochs=50, batch_size=1024)
folds = 5
from sklearn.ensemble import RandomForestClassifier

## one electrode
print ('Starting CNN cv 1/2')
accs_cnn_one = []
f1_cnn_one   = []
cv = sklearn.model_selection.GroupKFold(folds)
for train_idx, test_idx in tqdm(cv.split(groups, groups, groups), total = folds):
    
    sub_group = groups[train_idx]
    sub_cv = sklearn.model_selection.GroupKFold(folds)
    train_sub_idx, val_idx = sub_cv.split(groups[train_idx], groups[train_idx], groups[train_idx]).next()
    val_idx      = train_idx[val_idx]  
    train_idx    = train_idx[train_sub_idx]
    
    train_data   = data[train_idx,0:1,:]
    train_target = targets[train_idx]
    val_data     = data[val_idx,0:1,:]
    val_target   = targets[val_idx]
    test_data    = data[test_idx,0:1,:]       
    test_target  = targets[test_idx]

    network = CNN(train_data.shape, n_classes, input_var)
    train_fn = training_function(network, input_var, target_var, 0.001)
    val_fn =  validate_function(network, input_var, target_var)
    f1, acc = trainer.train(network_name, network, train_fn, val_fn, train_data, train_target, val_data, 
                   val_target, test_data,test_target, epochs=50, batch_size=1024)
    
    accs_cnn_one.append(f1)
    f1_cnn_one.append(acc)
print('{}/{}'.format(np.mean(accs_cnn_one),np.mean(f1_cnn_one)))

#
## all electrode
print ('Starting CNN cv 2/2')
accs_cnn_all = []
f1_cnn_all   = []
cv = sklearn.model_selection.GroupKFold(folds)
for train_idx, test_idx in tqdm(cv.split(groups, groups, groups), total = folds):
    
    sub_group = groups[train_idx]
    sub_cv = sklearn.model_selection.GroupKFold(folds)
    train_sub_idx, val_idx = sub_cv.split(groups[train_idx], groups[train_idx], groups[train_idx]).next()
    val_idx      = train_idx[val_idx]  
    train_idx    = train_idx[train_sub_idx]
    
    train_data   = data[train_idx]
    train_target = targets[train_idx]
    val_data     = data[val_idx]
    val_target   = targets[val_idx]
    test_data    = data[test_idx]       
    test_target  = targets[test_idx]
    
    network = CNN(train_data.shape, n_classes, input_var)
    train_fn = training_function(network, input_var, target_var, 0.001)
    val_fn =  validate_function(network, input_var, target_var)
    f1, acc = trainer.train(network_name, network, train_fn, val_fn, train_data, train_target, val_data, 
                   val_target, test_data,test_target, epochs=50, batch_size=1024)
    
    accs_cnn_all.append(f1)
    f1_cnn_all.append(acc)
print('{}/{}'.format(np.mean(accs_cnn_all),np.mean(f1_cnn_all)))

np.savez('\results\test_electrodes.npz', accs_cnn_one=accs_cnn_one, f1_cnn_one=f1_cnn_one, accs_cnn_all=accs_cnn_all, f1_cnn_all=f1_cnn_all,
                                         accs_rfc_one=accs_rfc_one, f1_rfc_one=f1_rfc_one, accs_rfc_all=accs_rfc_all, f1_rfc_all=f1_rfc_all)