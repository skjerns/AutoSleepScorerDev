# -*- coding: utf-8 -*-
"""
Spyder Editor

main script for training/classifying
"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/ANN')

import numpy as np
import datasets
import supervised_learning
import chainer
import chainer.functions as F
from analysis import Analysis
from models import neural_networks as models
from models.utilities import Classifier
from tools import append_line
if not 'sleeploader' in vars() :import sleeploader  # prevent reloading module
import tools
import time

def accuracy(X, T):
    """
    :param X: Test data
    :param T: Test labels
    """
    start=time.time()
    model.predictor.reset_state()
    # check if we are in train or test mode (e.g. for dropout)
    model.predictor.test = True
    model.predictor.train = False
    Y = []
    for step in xrange(X.shape[0]):
        x = chainer.Variable(np.asarray(X[step][None]), True)
        Y.append(model.predict(x))
    Y = np.squeeze(np.asarray(Y))
    acc = np.sum(np.argmax(Y, axis=1)==T) / float(len(T))     
    print((time.time()-start) / 60.0)           
    return acc/(len(T)) 
    

if os.name == 'posix':
    datadir  = '/media/simon/Windows/sleep/data/'
else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'c:\\sleep\\vinc\\'
    
sleep = sleeploader.SleepDataset(datadir)
selection = np.array(range(0,14)+range(33,50))
#selection = np.array(range(2))
sleep.load(selection, force_reload=False, shuffle=True, flat=True)

train_data, train_target = sleep.get_train()
test_data, test_target = sleep.get_test()

train_data = [tools.get_freq_bands(epoch) for epoch in train_data]
test_data = [tools.get_freq_bands(epoch) for epoch in test_data]

train_data = np.array(train_data,'float32').squeeze() 
train_target = np.array(train_target,'int32').squeeze()

test_data = np.array(test_data,'float32').squeeze()
test_target = np.array(test_target,'int32').squeeze()

#asd

#train_target[np.not_equal(train_target,5)]=0
train_target[train_target==8]=6
#
#test_target[np.not_equal(test_target,5)]=0
test_target[test_target==8]=6

#train_data = np.array(np.ravel(train_data),ndmin=2).T
#test_data  = np.array(np.ravel(test_data),ndmin=2).T
#
#train_target = np.repeat(train_target,3000)
#test_target = np.repeat(test_target,3000)


#%%
batch_size = 128
neurons = 50
layers = 3
epochs= 250
clipping =  15
decay = 1e-5
cutoff = 50
comment = 'rnn-freq-100samplebin-all-data'

#%% training routine
# get data

starttime = time.time()
training_data   = datasets.SupervisedData(train_data, train_target, batch_size=batch_size, shuffle=False)
validation_data = datasets.SupervisedData(test_data, test_target, batch_size=batch_size, shuffle=False)
#validation_data = training_data
#validation_data = training_data
# define model
nin = training_data.nin
nout = training_data.nout


# Enable/Disable different models here.

model = Classifier(models.RecurrentNeuralNetwork(nin, neurons, nout, nlayer=layers))
#model = Classifier(models.DeepNeuralNetwork(nin, neurons, nout, nlayer=layers))


# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(clipping))
optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=epochs, cutoff=cutoff)

runtime = time.time() - starttime
# plot loss and throughput
ann.report('results/tmp-{}-{}-{}-{}-{}-{}-{}'.format(layers,neurons,epochs,clipping,decay,batch_size, comment))

# create analysis object
ana = Analysis(ann.model, fname='results/tmp-{}-{}-{}-{}-{}-{}-{}'.format(layers,neurons,epochs,clipping,decay,batch_size, comment))

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)

#%% calculate accuracy
train_acc = accuracy(training_data.X,training_data.T)
test_acc  = accuracy(validation_data.X,validation_data.T)
print("\nAccuracy: Test: {}, Train: {}".format(test_acc,train_acc))

#%% save results
train_loss = ann.log[('training','loss')][-1]
test_loss  = ann.log[('validation','loss')][-1]
append_line('experiments.csv',[time.strftime("%c"),datadir,runtime/60, layers, neurons, epochs, clipping, decay, batch_size ,cutoff,train_acc, test_acc, train_loss,test_loss, comment, selection])