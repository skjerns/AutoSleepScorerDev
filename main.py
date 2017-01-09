# -*- coding: utf-8 -*-
"""
Spyder Editor

main script for training/classifying
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/ANN')
import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions
from preproc import pre_process
import chainer
from analysis import Analysis
from environment import datasets
from models import neural_networks as models
from paradigms import supervised_learning
from models.utilities import Classifier
from sklearn.preprocessing import normalize





train_touple,test_touple = pre_process()

train = [list(t) for t in zip(*train_touple)]
test  = [list(t) for t in zip(*test_touple)]




train_data = np.array(train[0],'float32').squeeze()
#train_data = train_data
train_target = np.array(train[1],'int32').squeeze()

test_data = np.array(test[0],'float32').squeeze()
test_target = np.array(test[1],'int32').squeeze()

#train_target[np.not_equal(train_target,3)]=0
train_target[train_target==8]=6
#
#test_target[np.not_equal(test_target,3)]=0
test_target[test_target==8]=6

#np.random.shuffle(test_targ)

#%% training routine
# get data
training_data   = datasets.SupervisedData(train_data,train_target, batch_size=32, shuffle=False)
validation_data = datasets.SupervisedData(test_data ,test_target, batch_size=32, shuffle=False)


# define model
nin = training_data.nin
nout = training_data.nout

model = Classifier(models.RecurrentNeuralNetwork(nin, 10, nout, nlayer=2))
# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(5))
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

ann = supervised_learning.SupervisedLearner(optimizer)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=100)

# plot loss and throughput
ann.report('results/tmp')

# create analysis object
ana = Analysis(ann.model, fname='results/tmp')

# analyse data
ana.classification_analysis(validation_data.X, validation_data.T)