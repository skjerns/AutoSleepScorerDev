# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:55:14 2017

@author: Simon
"""
import numpy as np
import keras
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import f1_score, accuracy_score

class Checkpoint(keras.callbacks.Callback):
    def __init__(self, validation_data, counter = 0, verbose=0, epochs_to_stop=15):
        super(Checkpoint, self).__init__()
        self.val_data = validation_data
        self.best_weights = None
        self.verbose = verbose
        self.counter = counter
        self.epochs_to_stop = epochs_to_stop
        

    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        self.val_f1 = []
        self.val_acc = []
        
        self.not_improved=0
        self.best_f1 = 0
        self.best_acc = 0
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        y_pred = self.model.predict(self.val_data[0], self.params['batch_size'], self.verbose)
        y_true = self.val_data[1]
        f1 = f1_score(np.argmax(y_pred,1),np.argmax(y_true,1), average="macro")
        acc = accuracy_score(np.argmax(y_pred,1),np.argmax(y_true,1))
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_f1.append(f1)
        self.val_acc.append(acc)
                    
        if f1 > self.best_f1:
            self.not_improved = 0
            self.best_f1 = f1
            self.best_acc = acc
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
            
        else:
            self.not_improved += 1
            if self.not_improved > self.epochs_to_stop and self.epochs_to_stop:
                print("No improvement after epoch {}".format(epoch))
                self.model.stop_training = True
        
    def on_train_end(self, logs={}):
        self.model.set_weights(self.best_weights)
        try:
            self.model.save(os.path.join('.','weights', str(self.counter) + self.model.name))
        except Exception as error:
            print("Got an error while saving model: {}".format(error))
        return
    
def ann(input_shape, n_classes, layers=2, neurons=300, dropout=0.35 ):
    """
    for working with extracted features
    """
    model = Sequential(name='ann')
    for l in range(layers):
        model.add(Dense (neurons, input_shape=input_shape, activation='elu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[keras.metrics.categorical_accuracy])
    return model


#%%
target = np.load('target.npy')
groups = np.load('groups.npy')
feats_eeg = np.load('feats_eeg.npy')# tools.feat_eeg(data[:,:,0])
feats_eog = np.load('feats_eog.npy')#tools.feat_eog(data[:,:,1])
feats_emg = np.load('feats_emg.npy')#tools.feat_emg(data[:,:,2])
feats = np.hstack([feats_eeg, feats_eog, feats_emg])

train_data = feats[15303:]
train_target = target[15303:]
val_data = feats[:15303]
val_target = target[:15303]
model = ann([37,], 5)

cb = Checkpoint([val_data,val_target], epochs_to_stop=100)
model.fit(train_data,train_target, 4048*10, epochs=500, verbose=2, callbacks=[cb])
