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

#if 'SleepDataset' not in vars(): from sleeploader import SleepDataset

if os.name == 'posix':
    datadir  = '/media/simon/Windows/sleep/data/'
else:
    datadir = 'c:\\sleep\\data\\'
    datadir = 'c:\\sleep\\vinc\\'
    
sleep = sleeploader.SleepDataset(datadir)
#selection = np.array(range(0,15)+range(33,51))
train_touple,test_touple = sleep.load(force_reload=False)

train = [list(t) for t in zip(*train_touple)]
test  = [list(t) for t in zip(*test_touple)]



train_data = np.array(train[0],'float32').squeeze()
train_target = np.array(train[1],'int32').squeeze()

test_data = np.array(test[0],'float32').squeeze()
test_target = np.array(test[1],'int32').squeeze()

#train_target[np.not_equal(train_target,3)]=0
train_target[train_target==8]=6
#
#test_target[np.not_equal(test_target,3)]=0
test_target[test_target==8]=6


#train_data = np.array(np.ravel(train_data),ndmin=2).T
#test_data  = np.array(np.ravel(test_data),ndmin=2).T
#
#train_target = np.repeat(train_target,3000)
#test_target = np.repeat(test_target,3000)


#%% training routine
# get data
training_data   = datasets.SupervisedData(train_data, train_target, batch_size=32, shuffle=False)
validation_data = datasets.SupervisedData(test_data, test_target, batch_size=32, shuffle=False)


# define model
nin = training_data.nin
nout = training_data.nout

model = Classifier(models.RecurrentNeuralNetwork(nin, 20, nout, nlayer=5))
# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(50))
#optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=200)

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)
#%%
acc = 0
for i in xrange(len(test_data)/32):
    idx = slice(i*32,i*32+32)
    y = model.predict(test_data[idx])
    ymax = np.argmax(y,axis=1)
    t = test_target[idx] 
    acc += F.accuracy(y,t).data
acc = acc/(len(test_data)/32)
print(acc)
    
    