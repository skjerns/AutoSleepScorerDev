# -*- coding: utf-8 -*-
"""
This is python 3 code
main script for training/classifying
"""
if not '__file__' in vars(): __file__= u'C:/Users/Simon/dropbox/Uni/Masterthesis/AutoSleepScorer/main.py'
import os
print(__name__)
if __name__== '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import gc; gc.collect()
import matplotlib
matplotlib.use('Agg')
import numpy as np
#import keras
#import keras_utils
#import time
#import tools
#import scipy
#import models
#import keras.backend as K
def main():
    global a
    a=2
    print(a)
    f1()
    
def f1():
    f2()
    print (a)

def f2():
    print (a)

main()