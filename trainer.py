# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:45:06 2017

@author: Simon
"""
import os
import theano
import lasagne
import sklearn
import numpy as np
from tqdm import tqdm
from lasagne import layers as L
from theano import tensor as T
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time


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
    pred = T.argmax(network_output, axis=1)
    return theano.function([input_tensor], [pred, network_output])



def train(network_name, network, train_fn, val_fn, train_data, train_target, val_data, val_target,
                        test_data, test_target, batch_size= 64, epochs = 100, plotting=False):
    
    val_batch_size = batch_size*4
    per_iter = 0
    best_val_acc = 0.0
    tra_loss_lst = []
    tra_acc_lst = []
    tra_f1_lst = []
    val_loss_lst = []
    val_acc_lst = []  
    val_f1_lst = []
    n_batch_train = len(train_data)/batch_size # number of training mini-batches given the batch_size
    n_batch_val   = len(val_data)/val_batch_size
    n_batch_test  = len(test_data)/val_batch_size
    
    if plotting: plt.figure(figsize=(10, 5))
    
    # Main training loop
    for epoch in range(epochs):
        starttime = time.time()
        train_data, train_target = sklearn.utils.shuffle(train_data,train_target)
        
        # training        
        tra_losses = []
        tra_preds, tra_targs = [], []
        for b in tqdm(range(0, n_batch_train+1), leave=False, desc = 'Epoch {}/{}'.format(epoch+1, epochs+1)):
            X = train_data[b*batch_size:(b+1)*batch_size,:].astype(np.float32) # extract a mini-batch from x_train
            Y = train_target[b*batch_size:(b+1)*batch_size] # extract labels for the mini-batch
            loss, pred = train_fn(X.astype(np.float32), Y.astype(np.int32))
            tra_losses.append(loss)
            tra_preds.append(pred)
            tra_targs.append(Y)
            
        tra_preds = np.hstack(tra_preds)
        tra_targs = train_target #np.hstack(tra_targs)
        tra_f1  = f1_score(tra_preds, tra_targs, average='macro')  
        tra_acc = accuracy_score(tra_preds,tra_targs)
        
        tra_loss_lst.append(np.mean(tra_losses))
        tra_acc_lst.append(tra_acc)
        tra_f1_lst.append(tra_f1)
            
        
        # validation
        val_losses = []
        val_preds, val_targs = [], []
        for b in tqdm(range(0, n_batch_val+1), leave=False, desc = 'validation: '):
            X = val_data[b*val_batch_size:(b+1)*val_batch_size,:].astype(np.float32) # extract a mini-batch from x_train
            Y = val_target[b*val_batch_size:(b+1)*val_batch_size] # extract labels for the mini-batch
            loss, pred = val_fn(X.astype(np.float32), Y.astype(np.int32))
            val_losses.append(loss)
            val_preds.append(pred)
            val_targs.append(Y)
            
        val_preds = np.hstack(val_preds)
        val_targs = val_target #np.hstack(val_targs)
        val_f1 = f1_score(val_preds, val_targs, average='macro')  
        val_acc = accuracy_score(val_preds,val_targs)
            
            
        val_loss_lst.append(np.mean(val_losses))
        val_acc_lst.append(val_f1)   
        val_f1_lst.append(val_f1)
    
        #continue
        print('Epoch {}    Train {:.1f}/{:.1f}    Val {:.1f}/{:.1f}    eta: {:.1f} minutes'.format(epoch+1,tra_f1*100,tra_acc*100,val_f1*100,val_acc*100, per_iter*(epochs-epoch)))
        if val_f1 > best_val_acc:
            best_val_acc = val_f1
            # save network
            params = L.get_all_param_values(network)
            np.savez(os.path.join('./', network_name +'.npz'), params=params)
    
        # plot learning curves
        if plotting:
            tra_loss_plt, = plt.plot(range(len(tra_loss_lst)), tra_loss_lst, 'b')
            val_loss_plt, = plt.plot(range(len(val_loss_lst)), val_loss_lst, 'g')
            tra_acc_plt, = plt.plot(range(len(tra_acc_lst)), tra_acc_lst, 'm')
            val_acc_plt, = plt.plot(range(len(val_acc_lst)), val_acc_lst, 'r')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend([tra_loss_plt, val_loss_plt, tra_acc_plt, val_acc_plt], 
                        ['training loss', 'validation loss', 'training accuracy', 'validation accuracy'],
                        loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('Best validation f1= {:.2f}%'.format(100. * best_val_acc))
            plt.pause(0.01)
        per_iter = (time.time() - starttime) / 60
            
            
    npz = np.load(os.path.join('./', network_name+'.npz')) # load stored parameters
    L.set_all_param_values(network, npz['params']) # set parameters

    test_preds = []
    for b in tqdm(range(0, n_batch_test+1), leave=False, desc='Testing'):
        X = test_data[b*val_batch_size:(b+1)*val_batch_size,:].astype(np.float32) # extract a mini-batch from x_train
        Y = test_target[b*val_batch_size:(b+1)*val_batch_size] # extract labels for the mini-batch
        _, pred = val_fn(X.astype(np.float32), Y.astype(np.int32))
        test_preds.append(pred)
        
    final_f1  = f1_score(np.hstack(test_preds), test_target, average='macro')
    final_acc = accuracy_score(np.hstack(test_preds), test_target)

    
    return final_f1, final_acc








