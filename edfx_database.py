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
import re
from tqdm import tqdm
from pyedflib import highlevel
from urllib.request import urlretrieve, urlopen


datadir='edfx'



def download_edfx(datadir):

    response = urlopen('https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/')
    html = str(response.read())    
    edfxfiles = re.findall(r'(?<=<a href=")[^"]*', html)[1:]

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
    files = [x for x in os.listdir(datadir) if x.endswith('Hypnogram.edf')]
    for file in tqdm(files):
        file = os.path.join(datadir,file)
        
        hypnogram = []
        annot = highlevel.read_edf_header(file)['annotations']
        for bstart, blength, bstage in annot:
            length = int(blength.decode())
            stage = bstage.decode()
            if 'movement' in stage.lower(): stage='M'
            stage = [str(stage[-1])]*(length//30)
            hypnogram.extend(stage)
            
        csv_file = file.replace('-Hypnogram','')[:-5] + '0-PSG.csv'
        with open(csv_file, "w") as f:
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
    for d, h in tqdm(zip(data,hypno), total=len(data)):
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


    
