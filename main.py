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
    # this method is terribly slow compared to batch-mode but I can be sure it's correct.
    # maybe later i'll implement a faster version
    acc = 0
    for i in xrange(len(T)):
        y = model.predict(X[i:i+1,:])
#        ymax = np.argmax(y,axis=1)
        t = T[i:i+1]
        acc += F.accuracy(y,t).data
    acc = acc/(len(T))
    return acc
    
    

if os.name == 'posix':
    datadir  = '/media/simon/Windows/sleep/data/'
else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'c:\\sleep\\vinc\\'
    
sleep = sleeploader.SleepDataset(datadir)
#selection = np.array(range(0,14)+range(33,50))
selection = np.array(range(4))
#train_touple,test_touple = sleep.load(selection,force_reload=False, shuffle=False)
data, target = sleep.load(selection,force_reload=False, shuffle=False)

shuffle_list = zip(data, target)
np.random.shuffle(shuffle_list)
data, target = [list(f) for f in zip(*shuffle_list)]

train_data = data[0:3*len(data)/4]
test_data  = data[3*len(data)/4:]

train_target = target[0:3*len(data)/4]
test_target  = target[3*len(data)/4:]

train_data = [tools.get_freq_bands(epoch) for epoch in train_data]
test_data = [tools.get_freq_bands(epoch) for epoch in test_data]

train_data = np.array(train_data,'float32').squeeze() 
train_target = np.array(train_target,'int32').squeeze()

test_data = np.array(test_data,'float32').squeeze()
test_target = np.array(test_target,'int32').squeeze()

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
batch_size = 32
neurons = 200
layers = 6
epochs= 20
clipping =  50
decay = 1e-5
cutoff= 50
comment = 'feed-forward'

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

#model = Classifier(models.RecurrentNeuralNetwork(nin, neurons, nout, nlayer=layers))
model = Classifier(models.DeepNeuralNetwork(nin, neurons, nout, nlayer=layers))


# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.GradientClipping(clipping))
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
sys.stdout.flush()

#%% save results
train_loss = ann.log[('training','loss')][-1]
test_loss  = ann.log[('validation','loss')][-1]
append_line('experiments.csv',[time.strftime("%c"),datadir,runtime/60, layers, neurons, epochs, clipping, decay, batch_size ,cutoff,train_acc, test_acc, train_loss,test_loss, comment])