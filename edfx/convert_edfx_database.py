# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:37:22 2017

    Converts the EDFx database:
       (0. downloads the edfx database)
        1. converts .HYP files into .csv files, with one annotation for each second, annotations in numerical format
        2. extracts the necessary channels from the .EDF files
        3. truncates the EDF data so that the excessive WAKE periods are removed
        4. Saves everthing to a new folder
        

call prepare_edfx_database() to start.

@author: skjerns
"""

#import wfdb  # this package is used to download the EDFx database
import os
import csv
import numpy as np
from tqdm import tqdm
from urllib.request import urlretrieve
import os.path
import pyedflib
import warnings
import datetime


edfx_datadir = 'C:/edfx/' # path to the EDFX database
target_dir   = 'c:/edf_out/' # path where output file shsould be saved

def prepare_edfx_database(edfx_datadir, target_dir):
    """
    Converts the EDFx database:
       (0. downloads the edfx database)
        1. converts .HYP files into .csv files, with one annotation for each second
        2. extracts the necessary channels from the .EDF files
        3. truncates the EDF data so that the excessive WAKE periods are removed
        4. Saves everthing to a new folder
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # 0. download edfx
    download_edfx(edfx_datadir)
    
    # convert hypnograms to CSV
    hyp_files = [os.path.join(edfx_datadir,file) for file in os.listdir(edfx_datadir) if file.endswith('.hyp')]
    for hyp_file in tqdm(hyp_files, desc='converting hypnograms'):
        convert_hypnogram(hyp_file, target_dir = target_dir)
    
    # extract and truncate channels from EDF
    chs = ['EEG FPZ-CZ'] # extracting EMG channel will not work as it is in 1Hz (a bug?)
    edf_files = [os.path.join(edfx_datadir,file) for file in os.listdir(edfx_datadir) if file.endswith('.edf')]
    csv_files = [os.path.join(target_dir,file) for file in os.listdir(target_dir) if file.endswith('.csv')]

    for edf_file, csv_file in tqdm(list(zip(edf_files, csv_files)), desc='Truncating EDF files'):
        truncate_edf_file(edf_file, target_dir, hypnogram_csv=csv_file, chs=chs)
        
    print('All done, should be saved to {}'.format(target_dir))


def download_edfx(datadir):
    """
    Downloads the EDF database to the datadir.
    """
    
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
        urlretrieve('https://physionet.nlm.nih.gov/pn4/sleep-edfx/' + record, os.path.join(datadir, record))
        received += os.path.getsize(os.path.join(datadir, record))
        progressloop.set_postfix_str('{:.0f}/{:.0f}MB'.format(received//1024/1024, 1868))
    return True
    
    
def convert_hypnogram(hyp_file, target_dir=None):
    """
    This function is quite a hack to read the edf hypnogram as a byte array. 
    I found no working reader for the hypnogram edfs.
    
    converts stage names to numerical indicators
    
    It reads the .HYP file and converts it to a wished conversion scheme.
    One annotation per second it given as output
    """
    
    conversion = {'W':'0', '1':'1', '2':'2', '3':'3', '4':'4', 'R':'5', 'A':'6', 'M':'7', '?':'9'}

    hypnogram = []
    with open(hyp_file, mode='rb') as f: # b is important -> binary
        raw_hypno = [x for x in str(f.read()).split('Sleep_stage_')][1:]
        for h in raw_hypno:
            stage = h[0]
            if stage =='?': continue
            stage = conversion[stage]
            repeat = int(h.split('\\')[0][12:]) # no idea if this also works on linux
            hypnogram.extend([stage]*repeat)  
            
    output_file = os.path.join(target_dir, os.path.splitext(os.path.basename(hyp_file))[0])+'.csv'
    with open(output_file, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(hypnogram)
    return True
    


def truncate_edf_file(edf_file, target_dir, hypnogram_csv=None, chs = ['EEG Fpz-Cz']):
    """
    Reads the edf files, exctracts the prefered channels and
    truncates the excessive Wake periods
    
    :param edf_file: the source edf_file from the edfx database
    :param target_dir: where to save the output  files
    :param hypnogram_csv: a hypnogram csv for truncating excessive WAKE periods.
                          if False, no truncation will be performed and only channels will be extracted
    :param chs: The channels that should be extracted
    """

    # load EDF file
    # digital = True prevents any changes in data due to rounding
    data, signal_headers, header = read_pyedf(edf_file, ch_names=chs, digital=True, verbose=False) 
    sfreq = int(signal_headers[0]['sample_rate'])
    # should always be in 2D format [ch, data] 
    data = np.atleast_2d(data)
    
    # if wanted, we truncate the data
    if hypnogram_csv is not None:
        hypno = np.loadtxt(hypnogram_csv)
        data = data[..., :len(hypno)*sfreq]
        assert data.shape[1]/sfreq==len(hypno), 'edf/hypno be same length {}!={}'.format(data.shape[1]//sfreq, len(hypno))
        
        data, hypno = truncate_eeg_data(data, hypno)
        assert data.shape[1]//sfreq==len(hypno), 'edf/hypno be same length after trunc {}!={}'.format(data.shape[1]//sfreq, len(hypno))


        target_csv = os.path.join(target_dir, os.path.basename(hypnogram_csv))
        np.savetxt(target_csv, hypno, delimiter='\n', fmt='%d')
    
    targed_edf = os.path.join(target_dir, os.path.basename(edf_file))
    write_pyedf(targed_edf, data, signal_headers, header, digital=True) # digital=True to preserve data integrity
    return True
        

def truncate_eeg_data(data, hypno):
    """
    Loads the previously saved EDFx database and truncates the eeg to remove the excessive non-bed time
    This is a lazy substitute for truncating with the lights-off markers.
    Hypnogrma is assumed to be in seconds
    
    :param data: a numpy array with EEG data
    :param hypno: the hypnogram, a numpy array
    """
    begin = np.where(hypno-np.roll(hypno,1)!=0)[0][0]-3300
    end   = np.where(hypno-np.roll(hypno,1)!=0)[0][-1]+1800
    if end>len(hypno): end=len(hypno)
    new_data = data[...,begin*100:end*100]
    new_hypno = hypno[begin:end]
    return new_data, new_hypno




#%%
##### Functions necessary for reading and writing EDF files
    
def read_pyedf(edf_file, ch_nrs=None, ch_names=None, digital=False, verbose=True):
    """
    Reads an EDF file using pyEDFlib and returns the data, header and signalheaders
    
    :param edf_file: link to an edf file
    :param ch_nrs: The numbers of channels to read
    :param ch_names: The names of channels to read
    :returns: signals, signal_headers, header
    """      
    assert (ch_nrs is  None) or (ch_names is None), 'names xor numbers should be supplied'
    if ch_nrs is not None and not isinstance(ch_nrs, list): ch_nrs = [ch_nrs]
    if ch_names is not None and not isinstance(ch_names, list): ch_names = [ch_names]

    with pyedflib.EdfReader(edf_file) as f:
        # see which channels we want to load
        channels = [ch.upper() for ch in f.getSignalLabels()]
        if ch_names is not None:
            ch_nrs = []
            for ch in ch_names:
                if not ch.upper() in channels:
                    warnings.warn('{} is not in source file (contains {})'.format(ch.upper(), channels))
                    print('will be ignored.')
                else:    
                    ch_nrs.append(channels.index(ch.upper()))
        if ch_nrs is None: # no numbers means we load all
            n = f.signals_in_file
            ch_nrs = np.arange(n)
        # load headers, signal information and 
        header = f.getHeader()
        signal_headers = [f.getSignalHeaders()[c] for c in ch_nrs]
        n = len(ch_nrs)
        dtype = np.int if digital else np.float
        signals = np.zeros([n, f.getNSamples()[0]], dtype=dtype)
        for i,c in enumerate(tqdm(ch_nrs, desc='Reading Channels', disable=not verbose)):
           signals[i, :] = f.readSignal(c, digital=digital)
    assert len(signals)==len(signal_headers), 'Something went wrong, lengths of headers is not length of signals'
    assert all(np.mean(signals==0, 1)<1) # check if any rows are 0
    return  signals, signal_headers, header

def make_header(technician='', recording_additional='', patientname='',
                patient_additional='', patientcode= '', equipment= '',
                admincode= '', gender= '', startdate=None, birthdate= ''):
    
    assert startdate is None or isinstance(startdate, datetime), 'must be datetime or None, is {}: {}'.format(type(startdate), startdate)
    assert birthdate is '' or isinstance(birthdate, (datetime,str)), 'must be datetime or empty, is {}'.format(type(birthdate))
    if startdate is None: 
        now = datetime.now()
        startdate = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second)
        del now
    if isinstance(birthdate, datetime): birthdate = birthdate.strftime('%d %b %Y')
    local = locals()
    header = {}
    for var in local:
        if isinstance(local[var], datetime):
            header[var] = local[var]
        else:
            header[var] = str(local[var])
    return header

def write_pyedf(edf_file, signals, signal_headers, header=None, digital=False):
    """
    Write an edf file using pyEDFlib
    
    :param signals: The signals as a list of arrays or a ndarray
    :param signal_headers: a list with one signal header for each signal (see function `make_signal_header)
    :param header: a header (see make_header())
    :param digital: whether signals are presented digitally or in physical values
    """
    assert header is None or isinstance(header, dict), 'header must be dictioniary'
    assert isinstance(signal_headers, list), 'signal headers must be list'
    signals = np.atleast_2d(signals)
    assert len(signal_headers)==len(signals), 'signals and signal_headers must be same length'
    
    if header is None: header = make_header()
    
    n_channels = len(signals)
    
    with pyedflib.EdfWriter(edf_file, n_channels=n_channels) as f:  
        f.setSignalHeaders(signal_headers)
        f.setHeader(header)
        f.writeSamples(signals, digital=digital)
        
    return os.path.isfile(edf_file) and os.path.getsize(edf_file)>signals.shape[1]
