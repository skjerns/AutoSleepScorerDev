# -*- coding: utf-8 -*-
import os, sys
import time
import keras
import tools
import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as K
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from tensorflow.python.client import device_lib
import tensorflow as tf
from tqdm import tqdm
from keras.layers.core import Lambda
from sklearn.utils import class_weight
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


def make_parallel(model, gpu_count=-1):
    """
    Distributes a model on two GPUs by creating a copy on each GPU
    and running slices of the batches on each GPU,
    then the two models outputs are merged again.
    Attention: The batches that are fed to the model must always be the same size
    @param (keras.models.Model) model: The model that should be distributed
    @param (int) gpu_count:            number of GPUs to use
    @return model
    """
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)
    if gpu_count == -1: gpu_count = get_available_gpus()
    if gpu_count < 2: return model # If only 1 GPU, nothing needs to be done.
    print('Making model parallel on {} GPUs'.format(gpu_count))
    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            # merged.append(keras.layers.merge(outputs, mode='concat', concat_axis=0))
            merged.append(keras.layers.concatenate(outputs, axis = 0))
        model = keras.models.Model(input=model.inputs, output=merged)
        model.multi_gpu = True
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.categorical_accuracy])
        return model
    
    

def get_activations(model, data, layername, batch_size=128, flatten=True):
#    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
#                                      [model.layers[layername].output if type(layername) is type(int) else model.get_layer(layername).output])
#    
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.get_layer(layername).output])
    activations = []
    batch_size = int(batch_size)
    for i in tqdm(range(int(np.ceil(len(data)/batch_size)))):
        batch = np.array(data[i*batch_size:(i+1)*batch_size], dtype=np.float32)
        act = get_layer_output([batch,0])[0]
        activations.extend(act)
    activations = np.array(activations, dtype=np.float32)
    if flatten: activations = activations.reshape([len(activations),-1])
    return activations

def test_data_cnn(data, target, model):
    """
    take a model and predict on the data
    """
    
    predictions = model.predict_classes(data)
    cnn_acc = accuracy_score(np.argmax(target,1),predictions)
    cnn_f1  = f1_score(np.argmax(target,1),predictions, average='macro')
    confmat = confusion_matrix(np.argmax(target,1),predictions)

    return cnn_acc, cnn_f1, confmat


def test_data_cnn_rnn(data, target, cnn, layername, rnn):
    """
    take two ready trained models (cnn+rnn)
    test on input data and return acc+f1
    """
    
    features = get_activations(cnn, data, layername)
    cnn_acc = accuracy_score(np.argmax(target,1),np.argmax(features,1))
    cnn_f1  = f1_score(np.argmax(target,1),np.argmax(features,1), average='macro')
    
    seqlen = rnn.input_shape[1]
    features_seq, target_seq = tools.to_sequences(features, target, seqlen=seqlen)
    results = np.argmax(get_activations(rnn, features_seq, -1), 1)
    
    rnn_acc = accuracy_score(np.argmax(target_seq,1),np.argmax(results,1))
    rnn_f1  = f1_score(np.argmax(target_seq,1),np.argmax(results,1), average='macro')
    return cnn_acc, cnn_f1, rnn_acc, rnn_f1


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
        self.start = time.time()
        
        self.not_improved=0
        self.best_f1 = 0
        self.best_acc = 0
        self.best_epoch = 0
        if self.plot: 
#            plt.close('all')
            self.figures.append(plt.figure())
        
    def on_epoch_end(self, epoch, logs={}):
        self.gen.reset() # to be sure
        y_pred = self.model.predict_generator(self.gen, self.gen.n_batches, max_q_size=1)
        y_pred = np.array(y_pred)
        y_true = self.gen.Y
        
        f1 = f1_score(np.argmax(y_true,1),np.argmax(y_pred,1), average="macro")
        acc = accuracy_score(np.argmax(y_true,1),np.argmax(y_pred,1))
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
            if self.verbose==1: print('.', end='', flush=True)
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
        if self.verbose > 0: print(' {:.1f} min'.format((time.time()-self.start)/60), flush=True)
        try:
            self.model.save(os.path.join('.','weights', str(self.counter) + self.model.name))
        except Exception as error:
            print("Got an error while saving model: {}".format(error))
        return
    

#%%
class generator_balanced(object):
    """
        Data generator util for Keras. 
        Generates data in such a way that it can be used with a stateful RNN

    :param X: data (either train or val) with shape 
    :param Y: labels (either train or val) with shape 
    :param num_of_batches: number of batches (np.ceil(len(Y)/batch_size) for non truncated mode 
    :param sequential: for stateful training
    :param truncate: Only yield full batches, in sequential mode the batches will be wrapped in case of false
    :param val: it only returns the data without the target
    :param random: randomize pos or neg within a batch, ignored in sequential mode 
    
    :return: patches (batch_size, 15, 15, 15) and labels (batch_size,)
    """
    def __init__(self, X, Y, batch_size):
        
        assert len(X) == len(Y), 'X and Y must be the same length {}!={}'.format(len(X),len(Y))
        self.X = X
        self.Y = Y
        self.batch_size = int(batch_size)
        self.reset()
        
    def reset(self):
        """ Resets the state of the generator"""
        self.step = 0
        Y = np.argmax(self.Y,1)
        labels = np.unique(Y)
        idx = []
        p = []
        smallest = len(Y)
        for i,label in enumerate(labels):
            where = np.where(Y==label)[0]
            if smallest > len(where): 
                self.slabel = i
                smallest = len(where)
            idx.append(where)
            p.append(np.ones(len(where))/len(where))
        self.p = p
        self.idx = idx
        self.labels = labels
        self.n_per_class = int(self.batch_size // len(labels))
        self.n_batches = int(np.ceil((smallest//self.n_per_class)*3))+1
        
        
    
    def __next__(self):
        if self.step==self.n_batches:
            self.reset()
        x_batch = []
        y_batch = []
        for label in self.labels:
            idx = self.idx[label]
            if len(idx)< self.n_per_class:
                x_batch.extend([self.X[i] for i in idx])
                y_batch.extend([self.Y[i] for i in idx])
                self.idx[label] = []
            else:
                number = self.n_per_class if label!=1 else self.n_per_class//3
                indexes = np.random.choice(np.arange(idx.size), number, p=self.p, replace = False)
                choice = idx[indexes]
                x_batch.extend([self.X[i] for i in choice])
                y_batch.extend([self.Y[i] for i in choice])
                self.idx[label] = np.delete (self.idx[label], indexes)
                self.p[label] = np.delete (self.p[label], indexes)
                    
        x_batch = np.array(x_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.int32)
    
        self.step+=1
        return (x_batch, y_batch)  
        



         
class generator(object):
    """
        Data generator util for Keras. 
        Generates data in such a way that it can be used with a stateful RNN

    :param X: data (either train or val) with shape 
    :param Y: labels (either train or val) with shape 
    :param num_of_batches: number of batches (np.ceil(len(Y)/batch_size) for non truncated mode 
    :param sequential: for stateful training
    :param truncate: Only yield full batches, in sequential mode the batches will be wrapped in case of false
    :param val: it only returns the data without the target
    :param random: randomize pos or neg within a batch, ignored in sequential mode 
    
    :return: patches (batch_size, 15, 15, 15) and labels (batch_size,)
    """
    def __init__(self, X, Y, batch_size, truncate=False, sequential=False, random=True, val=False, class_weights=None):
        
        assert len(X) == len(Y), 'X and Y must be the same length {}!={}'.format(len(X),len(Y))
        if sequential: print('Using sequential mode')
        
        self.X = X
        self.Y = Y
        self.rnd_idx = np.arange(len(Y))
        self.Y_last_epoch = []
        self.val = val
        self.step = 0
        self.i = 0
        self.truncate = truncate
        self.random = False if sequential or val else random
        self.batch_size = int(batch_size)
        self.sequential = sequential
        self.c_weights = class_weights if class_weights else dict(zip(np.unique(np.argmax(Y,1)),np.ones(len(np.argmax(Y,1)))))
        assert set(np.argmax(Y,1)) == set([int(x) for x in self.c_weights.keys()]), 'not all labels in class weights'
        self.n_batches = int(len(X)//batch_size if truncate else np.ceil(len(X)/batch_size))
        if self.random: self.randomize()
            
        
    def reset(self):
        """ Resets the state of the generator"""
        self.step = 0
        self.Y_last_epoch = []
        
        
    def randomize(self):
        self.X, self.Y, self.rnd_idx = shuffle(self.X, self.Y, self.rnd_idx)
        
    def get_Y(self):
        """ Get the last targets that were created/shuffled"""
        print('This feature is disabled. Please access generator.Y without shuffling.')
        if self.sequential or self.truncate:
            y_len = self.n_batches * self.batch_size
        else:
            y_len = len(self.Y)
        if self.val and (len(self.Y)!=y_len): print('not same length!')
        return np.array(self.Y_last_epoch, dtype=np.int32)[:y_len]
#        return np.array([x[0] for x in self.Y_last_epoch])
    
    def __next__(self):
        self.i +=1
        if self.step==self.n_batches:
            self.step = 0
            if self.random: self.randomize()
        if self.sequential: 
            return self.next_sequential()
        else:
            return self.next_normal()
        
    def next_normal(self):
        x_batch = np.array(self.X[self.step*self.batch_size:(self.step+1)*self.batch_size], dtype=np.float32)
        y_batch = np.array(self.Y[self.step*self.batch_size:(self.step+1)*self.batch_size], dtype=np.int32)

            
        self.step+=1
        if self.val:
            self.Y_last_epoch.extend(y_batch)
            return x_batch # for validation generator, save the new y_labels
        else:
            weights = np.ones(len(y_batch))
            for t in np.unique(np.argmax(y_batch,1)):
                weights[np.argmax(y_batch,1)==t] = self.c_weights[t]
            return (x_batch,y_batch, weights)  
        
    def next_sequential(self):
        x_batch = np.array([self.X[(seq * self.n_batches + self.step) % len(self.X)] for seq in range(self.batch_size)], dtype=np.float32)
        y_batch = np.array([self.Y[(seq * self.n_batches + self.step) % len(self.X)] for seq in range(self.batch_size)], dtype=np.int32)
        self.step+=1
        if self.val:
            self.Y_last_epoch.extend(y_batch)
            return x_batch # for validation generator, save the new y_labels
        else:
            return (x_batch,y_batch) 
        
#%%

def cv_rnn(data, targets, groups, modfun, rnn, epochs=250, folds=5, batch_size=512,
       val_batch_size=0, stop_after=0, name='', counter=0, plot = True):
    """
    Crossvalidation routinge for an RNN using extracted features on a basemodel
    :param rnns: list with the following:
                 [rnnfun, [layernames], seqlen, batch_size]
    :param ...: should be self-explanatory

    :returns results: dictionary with all RNN results
    """
    if val_batch_size == 0: val_batch_size = batch_size
    input_shape = list((np.array(data[0])).shape) #train_data.shape
    n_classes = targets.shape[1]
    
    
    
    gcv = GroupKFold(folds)
    results    = {modfun.__name__:[]}
    for lname in rnn[1]:
        results[lname] = []
    
    for i, idxs in enumerate(gcv.split(groups, groups, groups)):
        K.clear_session()
        print('-----------------------------')
        print('Starting fold {}: {}-{} at {}'.format(i,modfun.__name__, name, time.ctime()))
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
        g      = generator(train_data, train_target, batch_size, random=True)
        g_val  = generator(val_data, val_target, batch_size, val=True)
        g_test = generator(test_data, test_target, batch_size, val=True)
        cb     = Checkpoint(g_val, verbose=1, counter=counter, 
                 epochs_to_stop=stop_after, plot = plot)
        model.fit_generator(g, g.n_batches, verbose=0, epochs=epochs, callbacks=[cb], max_q_size=1)
        
        y_pred = model.predict_generator(g_test, g_test.n_batches, max_q_size=1)
        y_true = g_test.Y
        test_acc = accuracy_score(np.argmax(y_true,1),np.argmax(y_pred,1))
        test_f1  = f1_score(np.argmax(y_true,1),np.argmax(y_pred,1), average="macro")
        confmat = confusion_matrix(np.argmax(y_true,1),np.argmax(y_pred,1))
        results[modfun.__name__].append([cb.best_acc, cb.best_f1, test_acc, test_f1, confmat])
        print('ANN results: val acc/f1: {:.5f}/{:.5f}, test acc/f1: {:.5f}/{:.5f}'.format(cb.best_acc, cb.best_f1, test_acc, test_f1))
        
        rnn_modelfun, layernames, seq, rnn_bs = rnn
        
        for lname in layernames:
            train_data_extracted = get_activations(model, train_data, lname, batch_size)
            val_data_extracted   = get_activations(model, val_data,   lname, batch_size)
            test_data_extracted  = get_activations(model, test_data,  lname, batch_size)
            train_data_seq, train_target_seq = tools.to_sequences(train_data_extracted, train_target, seqlen=seq)
            val_data_seq, val_target_seq   = tools.to_sequences(val_data_extracted, val_target, seqlen=seq)
            test_data_seq, test_target_seq  = tools.to_sequences(test_data_extracted, test_target, seqlen=seq)
         
            rnn_shape  = list((np.array(train_data_seq[0])).shape)
            neurons = int(np.sqrt(rnn_shape[-1])*4)
            print('Starting RNN model with input from layer {}: {} at {}'.format(lname, rnn_shape, time.ctime()))
            rnn_model  = rnn_modelfun(rnn_shape, n_classes, layers=2, neurons=neurons, dropout=0.3)
            
            g      = generator(train_data_seq, train_target_seq, rnn_bs)
            g_val  = generator(val_data_seq, val_target_seq, rnn_bs, val=True)
            g_test = generator(test_data_seq, test_target_seq, rnn_bs, val=True)
            cb     = Checkpoint(g_val, verbose=1, counter=counter, 
                     epochs_to_stop=20, plot = plot)
            
            rnn_model.fit_generator(g, g.n_batches, epochs=epochs, verbose=0, callbacks=[cb])    
            y_pred = rnn_model.predict_generator(g_test, g_test.n_batches, max_q_size=1)
            y_true = g_test.Y
            val_acc = cb.best_acc
            val_f1  = cb.best_f1
            test_acc = accuracy_score(np.argmax(y_true,1),np.argmax(y_pred,1))
            test_f1  = f1_score(np.argmax(y_true,1),np.argmax(y_pred,1), average="macro")
            confmat = confusion_matrix(np.argmax(y_true,1),np.argmax(y_pred,1))
            results[lname].append([val_acc, val_f1, test_acc, test_f1, confmat])
            print('fold {}: val acc/f1: {:.5f}/{:.5f}, test acc/f1: {:.5f}/{:.5f}'.format(i,cb.best_acc, cb.best_f1, test_acc, test_f1))

            save_dict = {'1 Number':counter,
                         '2 Time':time.ctime(),
                         '3 CV':'{}/{}.'.format(i+1, folds),
                         '5 Model': lname,
                         '100 Comment': name,
                         '10 Epochs': epochs,
                         '11 Val acc': '{:.2f}'.format(val_acc*100),
                         '12 Val f1': '{:.2f}'.format(val_f1*100),
                         '13 Test acc':'{:.2f}'.format( test_acc*100),
                         '14 Test f1': '{:.2f}'.format(test_f1*100),
                         'Test Conf': str(confmat).replace('\n','')}
            tools.save_results(save_dict=save_dict)
        
        
        try:
            with open('{}_{}_{}_results.pkl'.format(counter,modelname,name), 'wb') as f:
                pickle.dump(results, f)
        except Exception as e:
            print("Error while saving results: ", e)
        sys.stdout.flush()
        
    return results

#%%
def test_model(data, targets, groups, testdata, modfun, epochs=250, batch_size=512,
       val_batch_size=0, stop_after=0, name='', counter=0, plot = True):
    """
    Train a model on the data
    
    :param ...: should be self-explanatory

    :returns results: (best_val_acc, best_val_f1, best_test_acc, best_test_f1)
    """
    return
#%%
def cv(data, targets, groups, modfun, epochs=250, folds=5, batch_size=512, val_batch_size=0, 
       stop_after=0, name='', counter=0, plot = True, balanced=False, weighted=False, log=False):
    """
    Crossvalidation routinge for training with a keras model.
    
    :param ...: should be self-explanatory

    :returns results: (best_val_acc, best_val_f1, best_test_acc, best_test_f1)
    """
    if val_batch_size == 0: val_batch_size = batch_size
    input_shape = list((np.array(data[0])).shape) #train_data.shape
    n_classes = targets.shape[1]
    
    print('Starting {}:{} at {}'.format(modfun.__name__, name, time.ctime()))
    
    global results
    results =[]
    gcv = GroupKFold(folds)
    
    for i, idxs in enumerate(gcv.split(groups, groups, groups)):
        K.clear_session()
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
        c_weights    = class_weight.compute_class_weight('balanced', np.unique(np.argmax(train_target,1)), np.argmax(train_target,1) )
        if log: c_weights = np.log2(c_weights+1)
        c_weights    = dict(zip(np.arange(5), c_weights)) if weighted else None
        
        
        model = modfun(input_shape, n_classes)
        modelname = model.name
        if balanced:
            g  = generator_balanced(train_data, train_target, batch_size)
        else:
            g  = generator(train_data, train_target, batch_size, random=True, class_weights=c_weights)
        g_val  = generator(val_data, val_target, batch_size, val=True)
        g_test = generator(test_data, test_target, batch_size, val=True)
        cb     = Checkpoint(g_val, verbose=1, counter=counter, 
                 epochs_to_stop=stop_after, plot = plot)
        model.fit_generator(g, g.n_batches, verbose=0, epochs=epochs, callbacks=[cb], max_q_size=1)
        predmodel = model
        
        
        y_pred = predmodel.predict_generator(g_test, g_test.n_batches, max_q_size=1)
        y_true = g_test.Y
        val_acc = cb.best_acc
        val_f1  = cb.best_f1
        test_acc = accuracy_score(np.argmax(y_true,1),np.argmax(y_pred,1))
        test_f1  = f1_score(np.argmax(y_true,1),np.argmax(y_pred,1), average="macro")
        
        confmat = confusion_matrix(np.argmax(y_true,1),np.argmax(y_pred,1))
        
        print('fold {}: val acc/f1: {:.5f}/{:.5f}, test acc/f1: {:.5f}/{:.5f}'.format(i,cb.best_acc, cb.best_f1, test_acc, test_f1))
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