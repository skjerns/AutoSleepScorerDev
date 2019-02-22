# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:37:22 2017

@author: Simon
"""

import wfdb  # this package is used to download the EDFx database
import os
import sleeploader
import csv
import numpy as np
import tools
from tqdm import tqdm
from urllib.request import urlretrieve

datadir='edfx'



def download_edfx(datadir):

    
    edfxfiles = ['SC4001E0-PSG.edf', 'SC4001E0-PSG.edf.hyp', 'SC4002E0-PSG.edf', 'SC4002E0-PSG.edf.hyp', 'SC4011E0-PSG.edf',
                 'SC4011E0-PSG.edf.hyp', 'SC4012E0-PSG.edf', 'SC4012E0-PSG.edf.hyp', 'SC4021E0-PSG.edf', 'SC4021E0-PSG.edf.hyp',
                 'SC4022E0-PSG.edf', 'SC4022E0-PSG.edf.hyp', 'SC4031E0-PSG.edf', 'SC4031E0-PSG.edf.hyp', 'SC4032E0-PSG.edf',
                 'SC4032E0-PSG.edf.hyp', 'SC4041E0-PSG.edf', 'SC4041E0-PSG.edf.hyp', 'SC4042E0-PSG.edf', 'SC4042E0-PSG.edf.hyp',
                 'SC4051E0-PSG.edf', 'SC4051E0-PSG.edf.hyp', 'SC4052E0-PSG.edf', 'SC4052E0-PSG.edf.hyp', 'SC4061E0-PSG.edf',
                 'SC4061E0-PSG.edf.hyp', 'SC4062E0-PSG.edf', 'SC4062E0-PSG.edf.hyp', 'SC4071E0-PSG.edf', 'SC4071E0-PSG.edf.hyp',
                 'SC4072E0-PSG.edf', 'SC4072E0-PSG.edf.hyp', 'SC4081E0-PSG.edf', 'SC4081E0-PSG.edf.hyp', 'SC4082E0-PSG.edf',
                 'SC4082E0-PSG.edf.hyp', 'SC4091E0-PSG.edf', 'SC4091E0-PSG.edf.hyp', 'SC4092E0-PSG.edf', 'SC4092E0-PSG.edf.hyp',
                 'SC4101E0-PSG.edf', 'SC4101E0-PSG.edf.hyp', 'SC4102E0-PSG.edf', 'SC4102E0-PSG.edf.hyp', 'SC4111E0-PSG.edf',
                 'SC4111E0-PSG.edf.hyp', 'SC4112E0-PSG.edf', 'SC4112E0-PSG.edf.hyp', 'SC4121E0-PSG.edf', 'SC4121E0-PSG.edf.hyp',
                 'SC4122E0-PSG.edf', 'SC4122E0-PSG.edf.hyp', 'SC4131E0-PSG.edf', 'SC4131E0-PSG.edf.hyp', 'SC4141E0-PSG.edf',
                 'SC4141E0-PSG.edf.hyp', 'SC4142E0-PSG.edf', 'SC4142E0-PSG.edf.hyp', 'SC4151E0-PSG.edf', 'SC4151E0-PSG.edf.hyp',
                 'SC4152E0-PSG.edf', 'SC4152E0-PSG.edf.hyp', 'SC4161E0-PSG.edf', 'SC4161E0-PSG.edf.hyp', 'SC4162E0-PSG.edf',
                 'SC4162E0-PSG.edf.hyp', 'SC4171E0-PSG.edf', 'SC4171E0-PSG.edf.hyp', 'SC4172E0-PSG.edf', 'SC4172E0-PSG.edf.hyp',
                 'SC4181E0-PSG.edf', 'SC4181E0-PSG.edf.hyp', 'SC4182E0-PSG.edf', 'SC4182E0-PSG.edf.hyp', 'SC4191E0-PSG.edf',
                 'SC4191E0-PSG.edf.hyp', 'SC4192E0-PSG.edf', 'SC4192E0-PSG.edf.hyp']
    edfxfiles = [x for x in edfxfiles if x.endswith('.edf')] + [x for x in edfxfiles if x.endswith('.hyp')] #just sorting them.
    print ('Downloading EDFx. This will take some time (~1.8 GB).\nDownloading the files manually might be faster.')
    try:os.mkdir(datadir)
    except Exception as e: print('') if type(e) is FileExistsError else print(e)
    received = 0 
    progressloop  = tqdm(edfxfiles, desc='Downloading')
    for record in progressloop:
        if os.path.isfile(os.path.join(datadir,record)):
            progressloop.set_postfix_str('File {} already exists.'.format(record))
            continue
        urlretrieve('https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/' + record, os.path.join(datadir, record))
        received += os.path.getsize(os.path.join(datadir, record))
        progressloop.set_postfix_str('{:.0f}/{:.0f}MB'.format(received//1024/1024, 1868))
    
    
    
def convert_hypnograms(datadir):
    """
    This function is quite a hack to read the edf hypnogram as a byte array. 
    I found no working reader for the hypnogram edfs.
    """
    print('Converting hypnograms')
    files = [x for x in os.listdir(datadir) if x.endswith('.hyp')]
    for file in files:
        file = os.path.join(datadir,file)
        hypnogram = []
        with open(file, mode='rb') as f: # b is important -> binary
        
            raw_hypno = [x for x in str(f.read()).split('Sleep_stage_')][1:]
            for h in raw_hypno:
                stage  = h[0]
                repeat = int(h.split('\\')[0][12:])//30 # no idea if this also works on linux
                hypnogram.extend(stage*repeat)            
        with open(file[:-4] + '.csv', "w") as f:
            writer = csv.writer(f, lineterminator='\r')
            writer.writerows(hypnogram)
    

def truncate_eeg(sleepdataset):
    """
    Loads the previously saved EDFx database and truncates the eeg to remove the excessive non-bed time
    This is a lazy substitute for truncating with the lights-off markers.
    """
    sleep = sleepdataset

    data = sleep.data
    hypno = sleep.hypno
    new_data = []
    new_hypno = []
    for d, h in zip(data,hypno):
        if d.shape[0]%3000!=0: d = d[:len(d)-d.shape[0]%3000]
        d = d.reshape([-1,3000,3])
        if 9 in h:
            delete = np.where(h==9)[0]
            d = np.delete(d, delete, 0)
            h = np.delete(h, delete, 0)
        begin = np.where(h-np.roll(h,1)!=0)[0][0]-300
        end = np.where(h-np.roll(h,1)!=0)[0][-1]+30
    
        d = d[begin:end]
        h = h[begin:end]

        d = d.ravel()
        new_hypno.append(h)
        new_data.append(d*1000000)

    sleep.data = new_data
    sleep.hypno = new_hypno
    sleep.save_object()            
            
def prepare(datadir = 'edfx'):
    """
    Download, prepare and save the EDF database
    """
    
    download_edfx(datadir)
    convert_hypnograms(datadir)
    
    sleep = sleeploader.SleepDataset(datadir)
    sleep.load()
    truncate_eeg(sleep)
    return sleep


    
