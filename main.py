# -*- coding: utf-8 -*-
"""
Spyder Editor

main script for training/classifying
"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/ANN')
import chainer.links as L
import numpy as np
import datasets
import supervised_learning
import chainer
import chainer.functions as F
from custom_analysis import Analysis
from models import neural_networks as models
from models.utilities import Classifier
if not 'sleeploader' in vars() : import sleeploader  # prevent reloading module
import tools
import time
import QRNN
from sklearn import metrics
from sklearn.preprocessing import scale
import matplotlib
matplotlib.rcParams['figure.figsize'] = (10, 3)

try:
    with open('count') as f:
        counter = int(f.read())
except IOError:
    print('No previous experiment?')
    counter = 0
        

#%%
#def main():
batch_size = 128
neurons = 30
layers = 3
epochs= 200
chunk_size = 3000
clipping = 250
decay = 1e-5
cutoff = None
future = 0
comment = 'LSTM-norm-3ch, new loader intrasub'
#comment = comment + raw_input('Comment? '+ comment)
link = L.LSTM
gpu=-1

#result  = [str(neurons) + ': '+ str(runRNN(neurons, layers, epochs, clipping, decay, cutoff, link, gpu, batch_size, comment)) for neurons in nneurons]

#%%

if os.name == 'posix':
    datadir  = '/home/simon/sleep/'
    datadir  = '/home/simon/vinc/'
else:
    datadir = 'c:\\sleep\\data\\'
#    datadir = 'C:\\sleep\\vinc\\brainvision\\'


    

sleep = sleeploader.SleepDataset(datadir)
selection = np.append(np.arange(0,14).flatten(),np.arange(33,50).flatten())
#selection = np.array(range(0,14))
#selection = np.array(range(6))
#selection=[]
sleep.load(selection, force_reload=False, shuffle=True, flat=True, chunk_len=chunk_size)

#train_data, train_target = sleep.get_train(flat=True)
#test_data, test_target   = sleep.get_test(flat=True)

train_data, train_target, test_data, test_target = sleep.get_intrasub()

#test_data  = tools.normalize(test_data)
#train_data = tools.normalize(train_data)

signals = train_data[100:103,:,1]
#d

print('Extracting features')
train_data = np.hstack( (tools.feat_eeg(train_data[:,:,0]), tools.feat_eog(train_data[:,:,1]),tools.feat_emg(train_data[:,:,2])))
test_data  = np.hstack( (tools.feat_eeg(test_data[:,:,0]), tools.feat_eog(test_data[:,:,1]), tools.feat_emg(test_data[:,:,2])))
#
#train_data =  tools.feat_eeg(train_data[:,:,0])
#test_data  =  tools.feat_eeg(test_data[:,:,0])

#train_data =   np.hstack([tools.get_freqs(train_data[:,:,0],50),tools.feat_eog(train_data[:,:,1])])
#test_data  =   np.hstack([tools.get_freqs(test_data[:,:,0], 50),tools.feat_eog(test_data[:,:,1])])

train_data =  tools.future(train_data, future)
test_data  =  tools.future(test_data, future)

## use this for freq data if more than 1 channel used
#test_data = test_data.reshape((-1,test_data.shape[-1]*test_data.shape[-2]),order='F')
#train_data = train_data.reshape((-1,train_data.shape[-1]*train_data.shape[-2]),order='F')

# use this for 1D data if more than 1 channel used
#test_data = test_data.reshape((-1,1),order='C')
#train_data = train_data.reshape((-1,1),order='C')
#train_target = train_target.repeat(3000)
#test_target = test_target.repeat(3000)

# 1D-Data
#train_data = train_data.flatten()
#test_data = train_data.flatten()

#train_data = train_data.reshape((-1,1))
#test_data = test_data.reshape((-1,1))
#train_data = train_data[np.newaxis].T
#test_data = test_data[np.newaxis].T
train_target[train_target==4]=3
train_target[train_target==5]=4
train_target[train_target==8]=5
test_target [test_target==4]=3
test_target [test_target==5]=4
test_target [test_target==8]=5
#train_data = np.array(np.ravel(train_data),ndmin=2).T
#test_data  = np.array(np.ravel(test_data),ndmin=2).T
#
#train_target = np.repeat(train_target,3000)
#test_target = np.repeat(test_target,3000)
#asd

##### binary classification SWS<>Other
#test_target [test_target==4]=3
#test_target [test_target!=3]=0
#test_target [test_target==3]=1
##            
#train_target [train_target==4]=3
#train_target [train_target!=3]=0
#train_target [train_target==3]=1
#del sleep.data
#del sleep
#import gc;gc.collect();
   
                     
# normalize features
train_data, test_data = tools.zscore(train_data,test_data)
#del sleep.data

if np.sum(np.isnan(train_data)) or np.sum(np.isnan(test_data)):print('Warning! NaNs detected')
#%% training routine

starttime = time.time()
training_data   = datasets.DynamicData(train_data, train_target, batch_size=batch_size)
validation_data = datasets.DynamicData(test_data, test_target, batch_size=batch_size)
#validation_data = training_data
#validation_data = training_data
# define model
nin = training_data.X.shape[1]
nout = np.max(training_data.T)+1


# Enable/Disable different models here.
model = Classifier(models.RecurrentNeuralNetwork(nin, neurons, nout, nlayer=layers, link=link))
#model = Classifier(models.DeepNeuralNetwork(nin, neurons, nout, nlayer=layers))


if gpu >= 0:
    chainer.cuda.memory_pool.free_all_blocks()
    chainer.cuda.get_device(0).use()
    model.to_gpu() 

# Set up an optimizer
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(clipping))
optimizer.add_hook(chainer.optimizer.WeightDecay(decay))

ann = supervised_learning.RecurrentLearner(optimizer, gpu=gpu, cutoff=cutoff)

# Finally we run the optimization
ann.optimize(training_data, validation_data=validation_data, epochs=epochs)

runtime = time.time() - starttime
                   
#%%  Reporting and analysis.
with open('count', 'w') as f:
  f.write(str(counter+1))


# plot loss and throughput
ann.report('results/{} {}-{}-{}-{}-{}-{}-{}'.format(counter,layers,neurons,epochs,clipping,decay,batch_size, comment))
# create analysis object
#ann.model.to_cpu()
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
    
    