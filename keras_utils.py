# -*- coding: utf-8 -*-
import os, sys
import time
import keras
import tools
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from tensorflow.python.client import device_lib

#%%

def get_available_gpus():
    """
    The function does what its name says. Simple as that.
    """
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])   

global MAX_GPU
MAX_GPU  = get_available_gpus()
print("{} GPUs available".format(MAX_GPU))


def switch_gpu():
    if MAX_GPU < 2: 
        os.environ["CUDA_VISIBLE_DEVICE"] = '0'
        return os.environ["CUDA_VISIBLE_DEVICE"]
    
    if "CUDA_VISIBLE_DEVICE" in os.environ:
        cdev = os.environ["CUDA_VISIBLE_DEVICE"]
    else:
        cdev = '0'
    os.environ["CUDA_VISIBLE_DEVICE"] = '0' if cdev=='1' else '1'
    print("using GPU: {}".format(os.environ["CUDA_VISIBLE_DEVICE"]))
    return os.environ["CUDA_VISIBLE_DEVICE"]




#%%
class Checkpoint(keras.callbacks.Callback):
    """
    Callback routine for Keras
    Calculates accuracy and f1-score on the validation data
    Implements early stopping if no improvement on validation data for X epochs
    """
    def __init__(self, generator, counter = 0, verbose=0, 
                 epochs_to_stop=15, plot = False):
        super(Checkpoint, self).__init__()
        self.gen = generator
        self.best_weights = None
        self.verbose = verbose
        self.counter = counter
        self.plot = plot
        self.epochs_to_stop = epochs_to_stop
        self.figures = []
        

    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_f1 = []
        self.val_acc = []
        
        self.not_improved=0
        self.best_f1 = 0
        self.best_acc = 0
        self.best_epoch = 0
        if self.plot: 
#            plt.close('all')
            self.figures.append(plt.figure())
        
    def on_epoch_end(self, epoch, logs={}):
        self.gen.reset() # to be sure
        if self.model.stateful: self.model.reset_states()
        y_pred = self.model.predict_generator(self.gen, self.gen.n_batches, max_q_size=1)
        if self.model.stateful: self.model.reset_states()
        y_pred = np.array(y_pred)
        y_true = self.gen.get_Y()
        
        f1 = f1_score(np.argmax(y_pred,1),np.argmax(y_true,1), average="macro")
        acc = accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
#        val_loss = keras.metrics.categorical_crossentropy(y_true, np.argmax(y_pred))
        val_loss = log_loss(y_true, y_pred)
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_loss.append(val_loss)
        self.val_f1.append(f1)
        self.val_acc.append(acc)

        if f1 > self.best_f1:
            self.not_improved = 0
            self.best_f1 = f1
            self.best_acc = acc
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            if self.verbose==1: print('+', end='')
        else:
            self.not_improved += 1
            if self.verbose==1: print('.', end='')
            if self.not_improved > self.epochs_to_stop and self.epochs_to_stop:
                print("\nNo improvement after epoch {}".format(epoch), flush=True)
                self.model.stop_training = True
                
        if self.plot:
            plt.clf()
            plt.subplot(1,2,1)
            plt.title
            plt.plot(self.loss)
            plt.plot(self.val_loss, 'r')
            plt.title('Loss')
            plt.legend(['loss', 'val_loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.subplot(1,2,2)
            plt.plot(self.val_acc)
            plt.plot(self.val_f1)
            plt.legend(['val acc', 'val f1'])
            plt.xlabel('Epoch')
            plt.ylabel('%')
            plt.title('Best: acc {:.1f}, f1 {:.1f}'.format(self.best_acc*100,self.best_f1*100))
            plt.show()
            plt.pause(0.0001)
        
        if self.verbose == 2:
            print('Epoch {}: , current: {:.1f}/{:.1f}, best: {:.1f}/{:.1f}'.format(epoch, acc*100, f1*100, self.best_acc*100 , self.best_f1*100))
        
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)
        sys.stdout.flush()
        try:
            self.model.save(os.path.join('.','weights', str(self.counter) + self.model.name))
        except Exception as error:
            print("Got an error while saving model: {}".format(error))
        return
    

#%%
         
class generator(object):
    """
        Data generator util for Keras. 
        Generates data in such a way that it can be used with a stateful RNN

    :param X: data (either train or val) with shape 
    :param Y: labels (either train or val) with shape 
    :param num_of_batches: number of batches (np.ceil(len(Y)/batch_size) for non truncated mode 
    :param sequential: for stateful training
    :param truncate: Only yield full batches, in sequential mode the batches will be wrapped in case of false
    :param val: I don't remember what the purpose is. But it only returns the data without the target
    :param random: randomize pos or neg within a batch, ignored in sequential mode 
    
    :return: patches (batch_size, 15, 15, 15) and labels (batch_size,)
    """
    def __init__(self, X, Y, batch_size, truncate=False, sequential=False, random=True, val=False):
        assert len(X) == len(Y), 'X and Y must be the same length {}!={}'.format(len(X),len(Y))
        self.X = X
        self.Y = Y
        self.Y_last_epoch = []
        self.val = val
        self.step = 0
        self.truncate = truncate
        self.random = False if sequential else random
        self.batch_size = batch_size
        self.sequential = sequential
        self.n_batches = int(len(X)//batch_size if truncate else np.ceil(len(X)/batch_size))
        
    def reset(self):
        """ Resets the state of the generator"""
        self.step = 0
        self.Y_last_epoch = []
        
    def get_Y(self):
        """ Get the last targets that were created/shuffled"""
        if self.sequential or self.truncate:
            y_len = self.n_batches * self.batch_size
        else:
            y_len = len(self.Y)
        return np.array(self.Y_last_epoch, dtype=np.int32)[:y_len]
    
    def __next__(self):
        if self.step==self.n_batches:
            self.step = 0
        if self.sequential: 
            return self.next_sequential()
        else:
            return self.next_normal()
        
    def next_sequential(self):
        x_batch = np.array([self.X[(seq * self.n_batches + self.step) % len(self.X)] for seq in range(self.batch_size)], dtype=np.float32)
        y_batch = np.array([self.Y[(seq * self.n_batches + self.step) % len(self.X)] for seq in range(self.batch_size)], dtype=np.int32)
        self.step+=1
        if self.val:
            self.Y_last_epoch.extend(y_batch)
            return x_batch # for validation generator, save the new y_labels
        else:
            return (x_batch,y_batch) 
        
    def next_normal(self):
        x_batch = np.array(self.X[self.step*self.batch_size:(self.step+1)*self.batch_size], dtype=np.float32)
        y_batch = np.array(self.Y[self.step*self.batch_size:(self.step+1)*self.batch_size], dtype=np.int32)
        self.step+=1
        if self.random:
           x_batch, y_batch = shuffle(x_batch, y_batch)
        if self.val:
            self.Y_last_epoch.extend(y_batch)
            return x_batch # for validation generator, save the new y_labels
        else:
            return (x_batch,y_batch) 

#%%
def cv(data, targets, groups, modfun, epochs=250, folds=5, batch_size=1440,
       val_batch_size=0, stop_after=0, name='', counter=0, plot = False, stateful=False):
    """
    Crossvalidation routinge for training with a keras model.
    
    :param ...: should be self-explanatory

    :returns results: (best_val_acc, best_val_f1, best_test_acc, best_test_f1)
    """
    if val_batch_size == 0: val_batch_size = batch_size
    input_shape = list((np.array(data[0])).shape) #train_data.shape
    if stateful: input_shape.insert(0,batch_size)
    n_classes = targets.shape[1]
    
    print('Starting {}:{} at {}'.format(modfun.__name__, name, time.ctime()))
    
    global results
    results =[]
    gcv = GroupKFold(folds)
    
    for i, idxs in enumerate(gcv.split(groups, groups, groups)):
        
        train_idx, test_idx = idxs
        sub_cv = GroupKFold(folds)
        train_sub_idx, val_idx = sub_cv.split(groups[train_idx], groups[train_idx], groups[train_idx]).__next__()
        val_idx      = train_idx[val_idx]  
        train_idx    = train_idx[train_sub_idx]
        
        train_data   = [data[i] for i in train_idx]
        train_target = targets[train_idx]
        val_data     = [data[i] for i in val_idx]
        val_target   = targets[val_idx]
        test_data    = [data[i] for i in test_idx]       
        test_target  = targets[test_idx]
        
        model = modfun(input_shape, n_classes)
        modelname = model.name
        g      = generator(train_data, train_target, batch_size, sequential=stateful, truncate=stateful)
        g_val  = generator(val_data, val_target, batch_size, sequential=stateful, truncate=stateful, val=True)
        g_test = generator(test_data, test_target, batch_size, sequential=stateful, truncate=stateful, val=True)
        cb     = Checkpoint(g_val, verbose=1, counter=counter, 
                 epochs_to_stop=stop_after, plot = plot)
        model.fit_generator(g, g.n_batches, epochs=epochs, verbose=0, callbacks=[cb])
        
        y_pred = model.predict_generator(g_test, g_test.n_batches)
        y_true = g_test.get_Y()
        val_acc = cb.best_acc
        val_f1  = cb.best_f1
        test_acc = accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
        test_f1  = f1_score(np.argmax(y_pred,1),np.argmax(y_true,1), average="macro")
        
        confmat = confusion_matrix(np.argmax(y_pred,1),np.argmax(y_true,1))
        
        print('val acc/f1: {:.5f}/{:.5f}, test acc/f1: {:.5f}/{:.5f}'.format(cb.best_acc, cb.best_f1, test_acc, test_f1))
        save_dict = {'1 Number':counter,
                     '2 Time':time.ctime(),
                     '3 CV':'{}/{}.'.format(i+1, folds),
                     '5 Model': modelname,
                     '100 Comment': name,
                     '10 Epochs': epochs,
                     '11 Val acc': '{:.2f}'.format(val_acc*100),
                     '12 Val f1': '{:.2f}'.format(val_f1*100),
                     '13 Test acc':'{:.2f}'.format( test_acc*100),
                     '14 Test f1': '{:.2f}'.format(test_f1*100),
                     'Test Conf': str(confmat).replace('\n','')}
        tools.save_results(save_dict=save_dict)
        results.append([val_acc, val_f1, test_acc, test_f1, confmat])
        
        try:
            with open('{}_{}_{}_results.pkl'.format(counter,modelname,name), 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print("Error while saving results: ", e)
        sys.stdout.flush()
        
    return results

print('loaded keras_utils.py')