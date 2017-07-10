# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:09:09 2017

@author: Simon
"""
import os
import numpy as np
import keras
import keras.backend.tensorflow_backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import LSTM, Reshape,Permute, TimeDistributed
from keras.layers import MaxPooling2D, Conv2D, Conv1D, MaxPooling1D
from keras.optimizers import Adadelta, RMSprop, Adam
input_shape=[3000,1]
n_classes=5
#m=cnn3adam(input_shape,n_classes)
#%%

def cnn3adam_slim(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='cnn3adam')
    model.add(Conv1D (kernel_size = (50), filters = 32, strides=5, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (5), filters = 64, strides=1, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    model.add(Conv1D (kernel_size = (5), filters = 64, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense (250, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (250, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model

def cnn3adam(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='cnn3adam')
    model.add(Conv1D (kernel_size = (50), filters = 64, strides=5, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (5), filters = 128, strides=1, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D (kernel_size = (5), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Flatten())
    model.add(Dense (500, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (500, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model

def cnn3adam_filter(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='cnn3adam_filter')
    model.add(Conv1D (kernel_size = (50), filters = 128, strides=5, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (5), filters = 256, strides=1, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D (kernel_size = (5), filters = 300, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    model.add(Flatten(name='conv3'))
    model.add(Dense (1500, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization(name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense (1500, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization(name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax',name='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model

def cnn3adam_filter_l2(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='cnn3adam_filter')
    model.add(Conv1D (kernel_size = (50), filters = 128, strides=5, input_shape=input_shape, 
                      kernel_initializer='he_normal', activation='elu',kernel_regularizer=keras.regularizers.l2(0.005))) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (5), filters = 256, strides=1, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D (kernel_size = (5), filters = 300, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    model.add(Flatten(name='conv3'))
    model.add(Dense (1500, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization(name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense (1500, activation='elu', kernel_initializer='he_normal'))
    model.add(BatchNormalization(name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax',name='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam())
    return model


def ann(input_shape, n_classes, layers=2, neurons=80, dropout=0.35 ):
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


def largeann(input_shape, n_classes, layers=3, neurons=2000, dropout=0.35 ):
    """
    for working with extracted features
    """
#    gpu = switch_gpu()
#    with K.tf.device('/gpu:{}'.format(gpu)):
#        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
    model = Sequential(name='ann')
#    model.gpu = gpu
    for l in range(layers):
        model.add(Dense (neurons, input_shape=input_shape, activation='elu', kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[keras.metrics.categorical_accuracy])
    return model

#%% everyhing recurrent for ANN


def pure_rnn_do(input_shape, n_classes):
    """
    just replace ANN by RNNs
    """
    model = Sequential(name='pure_rnn_do')
    model.add(LSTM(80,return_sequences=True, input_shape=input_shape, recurrent_dropout=0.3))
    model.add(LSTM(80, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[keras.metrics.categorical_accuracy])
    return model

def ann_rnn(input_shape, n_classes):
    """
    for working with extracted features
    """
    model = Sequential(name='ann_rnn')
    model.add(TimeDistributed(Dense (80, activation='elu', kernel_initializer='he_normal'), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(TimeDistributed(Dense (80, activation='elu', kernel_initializer='he_normal')))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(LSTM(50))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[keras.metrics.categorical_accuracy])
    return model

def pure_rnn(input_shape, n_classes,layers=2, neurons=80, dropout=0):
    """
    just replace ANN by RNNs
    """
    model = Sequential(name='pure_rnn')
    model.add(LSTM(neurons, return_sequences=False if layers==1 else True, input_shape=input_shape, recurrent_dropout=dropout))
    for i in range(layers-1):
        model.add(LSTM(neurons, return_sequences=False if i==layers-2 else True, recurrent_dropout=dropout))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[keras.metrics.categorical_accuracy])
    return model

def pure_rnn_3(input_shape, n_classes):
    """
    just replace ANN by 3xRNNs
    """
    model = Sequential(name='pure_rnn')
    model.add(LSTM(80, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(80, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[keras.metrics.categorical_accuracy])
    return model



#%% training routine
def tsinalis(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 1, 15000, 1)
    """
    model = Sequential(name='Tsinalis')
    model.add(Conv2D (kernel_size = (1,200), filters = 20, input_shape=input_shape, activation='relu'))
    print(model.input_shape)
    print(model.output_shape)
    model.add(MaxPooling2D(pool_size = (1,20), strides=(1,10)))
    print(model.output_shape)
    model.add(Permute((3,2,1)))
    print(model.output_shape)
    model.add(Conv2D (kernel_size = (20,30), filters = 400, activation='relu'))
    print(model.output_shape)
    model.add(MaxPooling2D(pool_size = (1,10), strides=(1,2)))
    print(model.output_shape)
    model.add(Flatten())
    print(model.output_shape)
    model.add(Dense (500, activation='relu'))
    model.add(Dense (500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation = 'sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    return model


def cnn1d(input_shape, n_classes ):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 1)
    """
    model = Sequential(name='1D CNN')
    model.add(Conv1D (kernel_size = (50), filters = 150, strides=5, input_shape=input_shape, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    print(model.output_shape)
    model.add(Conv1D (kernel_size = (8), filters = 200, strides=2, input_shape=input_shape, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    print(model.output_shape)
    model.add(MaxPooling1D(pool_size = (10), strides=(2)))
    print(model.output_shape)
    
    model.add(Conv1D (kernel_size = (8), filters = 400, strides=2, input_shape=input_shape, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    print(model.output_shape)
    model.add(Flatten())
    model.add(Dense (700, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (700, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    return model



def rcnn(input_shape, n_classes):
    """
    Input size should be [batch, 1d, ch] = (XXX, 3000, 1)
    """
    model = Sequential(name='RCNN test')
    model.add(Conv1D (kernel_size = (200), filters = 20, batch_input_shape=input_shape, activation='elu'))
    model.add(MaxPooling1D(pool_size = (20), strides=(10)))
    model.add(Conv1D (kernel_size = (20), filters = 200, activation='elu'))
    model.add(MaxPooling1D(pool_size = (10), strides=(3)))
    model.add(Conv1D (kernel_size = (20), filters = 200, activation='elu'))
    model.add(MaxPooling1D(pool_size = (10), strides=(3)))
    model.add(Dense (512, activation='elu'))
    model.add(Dense (512, activation='elu'))
    model.add(Reshape((1,model.output_shape[1])))
    model.add(LSTM(256, stateful=True, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation = 'sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
    return model

def rnn_old(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 1, 15000, 1)
    """
    model = Sequential(name='Simple 1D CNN')
    model.add(keras.layers.LSTM(50, stateful=True, batch_input_shape=input_shape, return_sequences=False))
    model.add(Dense(n_classes, activation='sigmoid'))
    print(model.output_shape)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=[keras.metrics.categorical_accuracy])
    return model








#%% old models

def cnn1(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='no_MP_small_filters')
    model.add(Conv1D (kernel_size = (10), filters = 64, strides=2, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (10), filters = 64, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (10), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (10), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv1D (kernel_size = (10), filters = 150, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense (1024, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (1024, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
    return model
def cnn2(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='MP_small_filters')
    model.add(Conv1D (kernel_size = (10), filters = 64, strides=2, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D (kernel_size = (10), filters = 64, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D (kernel_size = (10), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())

    model.add(Flatten())
    model.add(Dense (500, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (500, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
    return model
def cnn3(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='mixture')
    model.add(Conv1D (kernel_size = (50), filters = 64, strides=5, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (5), filters = 128, strides=1, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())
    
    model.add(Conv1D (kernel_size = (5), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(MaxPooling1D())

    model.add(Flatten())
    model.add(Dense (500, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (500, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
    return model
def cnn4(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='large_kernel')
    model.add(Conv1D (kernel_size = (100), filters = 128, strides=10, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D (kernel_size = (100), filters = 128, strides=1, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (100), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense (768, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (768, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
    return model
def cnn5(input_shape, n_classes):
    """
    Input size should be [batch, 1d, 2d, ch] = (None, 3000, 3)
    """
    model = Sequential(name='very_large_kernel')
    model.add(Conv1D (kernel_size = (200), filters = 128, strides=3, input_shape=input_shape, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D (kernel_size = (200), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Conv1D (kernel_size = (200), filters = 128, strides=1, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv1D (kernel_size = (10), filters = 128, strides=2, kernel_initializer='he_normal', activation='elu')) 
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense (768, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense (768, activation='elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta())
    return model
m=cnn3adam_filter([3000,3],5)
print('loaded model.py')