# -*- coding: utf-8 -*-
"""
Spyder Editor

main script for training/classifying
"""
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
import sleeploader
import loader

#if 'SleepDataset' not in vars(): from sleeploader import SleepDataset

if os.name == 'posix':
    datadir  = '/media/simon/Windows/sleep/data/'
else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'c:\\sleep\\vinc\\'
    
sleep = sleeploader.SleepDataset(datadir)
#selection = np.array(range(0,14))
selection = np.array(range(15))
train_touple,test_touple = sleep.load(selection,force_reload=False, shuffle=False)

train = [list(t) for t in zip(*train_touple)]
test  = [list(t) for t in zip(*test_touple)]

#train_data = [loader.get_freq_bands(epoch) for epoch in train[0]]
#test_data = [loader.get_freq_bands(epoch) for epoch in test[0]]

train_data = np.array(train[0],'float32').squeeze()
train_target = np.array(train[1],'int32').squeeze()

test_data = np.array(test[0],'float32').squeeze()
test_target = np.array(test[1],'int32').squeeze()

train_target[np.not_equal(train_target,5)]=0
train_target[train_target==5]=1
#
test_target[np.not_equal(test_target,5)]=0
test_target[test_target==5]=1

#train_data = np.array(np.ravel(train_data),ndmin=2).T
#test_data  = np.array(np.ravel(test_data),ndmin=2).T
#
#train_target = np.repeat(train_target,3000)
#test_target = np.repeat(test_target,3000)


#%%
batch_size = 64
neurons = 100
layers = 3
epochs= 1000
clipping = 25
decay = 1e-5

#%% training routine
# get data
np.random.shuffle(train_target)
training_data   = datasets.SupervisedData(train_data, train_target, batch_size=batch_size, shuffle=False)
validation_data = datasets.SupervisedData(test_data, test_target, batch_size=batch_size, shuffle=False)

#validation_data = training_data
# define model
nin = training_data.nin
nout = training_data.nout

model = Classifier(models.RecurrentNeuralNetwork(nin, neurons, nout, nlayer=layers))
# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(clipping))
optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=epochs)

# plot loss and throughput
ann.report('results/tmp-{}-{}-{}-{}-{}-{}'.format(layers,neurons,epochs,clipping,decay,batch_size))

# create analysis object
ana = Analysis(ann.model, fname='results/tmp-{}-{}-{}-{}-{}-{}'.format(layers,neurons,epochs,clipping,decay,batch_size))

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)
#%%
acc = 0
for i in xrange(len(train_data)/32):
    idx = slice(i*32,i*32+32)
    y = model.predict(train_data[idx])
    ymax = np.argmax(y,axis=1)
    t = train_target[idx] 
    acc += F.accuracy(y,t).data
acc = acc/(len(train_data)/32)
print(acc)
    
    