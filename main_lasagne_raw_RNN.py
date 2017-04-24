# -*- coding: utf-8 -*-
"""
Spyder Editor
 
main script for training/classifying
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
import time
import sklearn
from sklearn import metrics
from sklearn.preprocessing import scale
import matplotlib
from lasagne.regularization import regularize_layer_params_weighted, l2
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
#%%
#def main():
neurons = 25
layers = 3
epochs= 200
chunk_size = 3000
decay = 1e-5
cutoff = None
future = 0
comment = 'Lasagna test'
#link = L.LSTM
gpu=-1
 
#result  = [str(neurons) + ': '+ str(runRNN(neurons, layers, epochs, clipping, decay, cutoff, link, gpu, batch_size, comment)) for neurons in nneurons]
 
#%%
print(comment)
if os.name == 'posix':
    datadir  = '/home/simon/sleep/'
#    datadir  = '/home/simon/vinc/'
    datadir = '/media/simon/Windows/sleep/corrupted'

else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'C:\\sleep\\vinc\\brainvision\\correct\\'
    datadir = 'C:\\sleep\\corrupted\\'

 
 
sleep = sleeploader.SleepDataset(datadir)

#sleep.load(selection, force_reload=False, shuffle=True, chunk_len=chunk_size)

#sleep.load_object('adults.dat')
sleep.load_object()

sleep.shuffle_data()
train_data, train_target = sleep.get_train()
test_data, test_target   = sleep.get_test()

#
test_data    = scipy.stats.mstats.zmap(test_data, train_data , axis = None)
train_data   = scipy.stats.mstats.zmap(train_data, train_data, axis = None)


#child_data, child_target = sleep.load(children_sel, force_reload=False, shuffle=True, flat=True, chunk_len=chunk_size)
#train_data, train_target, test_data, test_target = sleep.get_intrasub()

#test_data  = tools.normalize(test_data)
#train_data = tools.normalize(train_data)
 


 

print('Extracting features')
#train_data = np.hstack( (tools.feat_eeg(train_data[:,:,0]), tools.feat_eog(train_data[:,:,1]),tools.feat_emg(train_data[:,:,2])))
#test_data  = np.hstack( (tools.feat_eeg(test_data[:,:,0]), tools.feat_eog(test_data[:,:,1]), tools.feat_emg(test_data[:,:,2])))
#child_data = np.hstack( (tools.feat_eeg(child_data[:,:,0]), tools.feat_eog(child_data[:,:,1]), tools.feat_emg(child_data[:,:,2])))
#train_data =  tools.feat_eeg(train_data[:,:,0])
#test_data  =  tools.feat_eeg(test_data[:,:,0])
 
#train_data =   np.hstack([tools.get_freqs(train_data[:,:,0],50),tools.feat_eog(train_data[:,:,1])])
#test_data  =   np.hstack([tools.get_freqs(test_data[:,:,0], 50),tools.feat_eog(test_data[:,:,1])])
 
 
 
#train_target[train_target==4] = 3
train_target[train_target==5] = 4
#train_target[train_target==8] = 5
#test_target [test_target==4] = 3
test_target [test_target==5] = 4
#test_target [test_target==8] = 5

train_data   = np.delete(train_data, np.where(train_target==8) ,axis=0)     
train_target = np.delete(train_target, np.where(train_target==8) ,axis=0)     
test_data    = np.delete(test_data, np.where(test_target==8) ,axis=0)     
test_target  = np.delete(test_target, np.where(test_target==8) ,axis=0) 


#train_data =  tools.future(train_data, 4)
#test_data  =  tools.future(test_data, 4)

#training_data   = custom_datasets.DynamicData(train_data, train_target, batch_size=batch_size)
#validation_data = custom_datasets.DynamicData(test_data, test_target, batch_size=batch_size)             
#train_data = np.expand_dims(train_data,1)
#test_data = np.expand_dims(test_data,1)
#train_data = np.expand_dims(train_data,1)
#test_data = np.expand_dims(test_data,1)
#train_data = train_data.swapaxes(1,2)
#test_data = test_data.swapaxes(1,2)
#train_data = train_data[:,0:1,:]
#test_data = test_data[:,0:1,:]

# 
if np.sum(np.isnan(train_data)) or np.sum(np.isnan(test_data)):print('Warning! NaNs detected')
#%% training routine
#train_target = tools.label_to_one_hot(train_target)
#test_target  = tools.label_to_one_hot(test_target)


#def LSTM(data_size, n_classes):
#    network = L.InputLayer(shape=(None, None, data_size[2]))
#    network = L.LSTMLayer(network,  25)
#    network = L.LSTMLayer(network,  25)
#    network = L.LSTMLayer(network,  25)
#    network = L.DenseLayer( network, num_units=n_classes, W=lasagne.init.GlorotNormal(), nonlinearity=lasagne.nonlinearities.softmax)
#    return network
def tsinalis(data_size, n_classes):
    network = L.InputLayer(shape=(None, 1, 1, data_size[3]), input_var=input_var)
    print network.output_shape
    network = L.Conv2DLayer( network, num_filters=20, filter_size = (1,500),stride= (1,1),  nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.HeNormal()) 
    network = L.batch_norm(network)
    print network.output_shape
    network = L.MaxPool2DLayer(network, pool_size = (1,20), stride = (1,10))
    print network.output_shape
    network = L.reshape(network, ([0],[2],[1],-1))
    print network.output_shape
    network = L.Conv2DLayer( network, num_filters = 400, filter_size = (20,30), stride = (1,1),  nonlinearity=lasagne.nonlinearities.elu, W=lasagne.init.HeNormal()) 
    network = L.batch_norm(network)
    print network.output_shape
    network = L.MaxPool2DLayer(network, pool_size = (1,10), stride = (1,2))
    print network.output_shape
    network = L.DenseLayer( network, num_units=500, W=lasagne.init.HeNormal(),      nonlinearity=lasagne.nonlinearities.elu)
    network = L.batch_norm(network)
    network = L.DenseLayer( network, num_units=500, W=lasagne.init.HeNormal(),      nonlinearity=lasagne.nonlinearities.elu)
    network = L.batch_norm(network)
    network = L.DropoutLayer(network, p=0.3)
    network = L.DenseLayer( network, num_units=n_classes,W=lasagne.init.GlorotNormal(),      nonlinearity=lasagne.nonlinearities.softmax)
    return network

def CNN(data_size, n_classes):
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

def raw_RNN(data_size, n_classes):
    network = L.InputLayer(shape=(None, data_size[1], data_size[2]), input_var=input_var)
    network = L.LSTMLayer(network, num_units=50)
    network = L.LSTMLayer(network, num_units=50)
    network = L.LSTMLayer(network, num_units=50)
    print network.output_shape

    network = L.DenseLayer( network, num_units=n_classes,W=lasagne.init.GlorotNormal(), nonlinearity=lasagne.nonlinearities.softmax)
    print network.output_shape

    return network

def RNN(data_size, n_classes):
    network = L.InputLayer(shape=(None, data_size[1], data_size[2]), input_var=input_var)
    print network.output_shape
    network = L.Conv1DLayer( network, num_filters=50, filter_size = 50, stride = 10, W=lasagne.init.HeNormal())
    network = L.batch_norm(network)
    network = L.DropoutLayer(network, p=0.2)

    network = L.Conv1DLayer( network, num_filters=50, filter_size = 5, stride = 1, W=lasagne.init.HeNormal())
    network = L.DropoutLayer(network, p=0.2)

    network = L.batch_norm(network)
    network = L.MaxPool1DLayer(network, pool_size = (5), stride = (2))
    network = L.Conv1DLayer( network, num_filters=50, filter_size = 5, stride = 1, W=lasagne.init.HeNormal())
    network = L.DropoutLayer(network, p=0.2)
    network = L.batch_norm(network)
    network = L.MaxPool1DLayer(network, pool_size = (4), stride = (2))
    print network.output_shape

    network = L.FlattenLayer(network)
    print network.output_shape
    network = L.ReshapeLayer(network, [-1,4, [1]])
    print network.output_shape

    network = L.LSTMLayer(network, 250,only_return_final =True)
    network = L.DenseLayer( network, num_units=500, W=lasagne.init.HeNormal())
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

#%%
learning_rate = 0.001

data_size = train_data.shape
n_classes = len(np.unique(train_target))
input_var  = T.tensor3('inputs')
target_var = T.ivector('targets')
batch_size = 32
test_batch_size = 32
epochs = 100
network_name = 'RNN'

network = raw_RNN(data_size, n_classes)

train_fn = training_function(network, input_var, target_var, learning_rate)
val_fn =  validate_function(network, input_var, target_var)

aa = trainer.train('test', network,train_fn, val_fn, train_data, train_target, test_data, 
                   test_target, test_data,test_target, epochs=50, batch_size=128)
