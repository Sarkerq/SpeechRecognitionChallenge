import tensorflow as tf
import scipy as sp
import numpy as np
import pandas as pd
import re
import hashlib
from sklearn.preprocessing import StandardScaler
import librosa    
import os
from pathlib import Path
import IPython.display as ipd
import cProfile

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from time import clock
import samplerate

"""Determines which data partition the file should belong to.

We want to keep files in the same training, validation, or testing sets even if new ones are added over time. This makes it less likely that testing samples will accidentally be reused in training when long runs are restarted for example. To keep this stability, a hash of the filename is taken and used to determine which set it should belong to. This determination only depends on the name and the set proportions, so it won't change as other files are added.

It's also useful to associate particular files as related (for example words spoken by the same person), so anything after 'nohash' in a filename is ignored for set determination. This ensures that 'bobby_nohash_0.wav' and 'bobby_nohash_1.wav' are always in the same set, for example.

Args: filename: File path of the data sample. validation_percentage: How much of the data set to use for validation. testing_percentage: How much of the data set to use for testing.

Returns: String, one of 'training', 'validation', or 'testing'. """ 

def which_set(filename, validation_percentage, testing_percentage):
    MAX_NUM_WAVS_PER_CLASS = 2**27 - 1 # ~134M 
    base_name = os.path.basename(filename) 
    # We want to ignore anything after 'nohash' in the file name when 
    # deciding which set to put a wav in, so the data set creator has a way of # grouping wavs that are close variations of each other. 
    hash_name = re.sub(r'nohash.*$', '', base_name) 
    # This looks a bit magical, but we need to decide whether this file should # go into the training, testing, or validation sets, and we want to keep 
    # existing files in the same set even if more files are subsequently 
    # added.   
    # To do that, we need a stable way of deciding based on just the file name 
    # itself, so we do a hash of that and then use that to generate a 
    # probability value that we use to assign it. 
    hash_name_hashed = hashlib.sha1(hash_name.encode("utf8")).hexdigest() 
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    #print(percentage_hash)
    if percentage_hash < validation_percentage: 
        result = 'validation' 
    elif percentage_hash < (testing_percentage + validation_percentage): 
        result = 'testing' 
    else: 
        result = 'training' 
    return result

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def get_train_test(train_audio_path, val_perc, portion):
    start = clock() 

    train_labels = os.listdir(train_audio_path)
    train_labels.remove('_background_noise_')

    labels_to_keep = ['yes', 'no', 'up', 'down', 'left',
                    'right', 'on', 'off', 'stop', 'go', 'silence']

    train_file_labels = dict()
    for label in train_labels:
        files = os.listdir(train_audio_path + '/' + label)
        for f in files:
            train_file_labels[label + '/' + f] = label

    train = pd.DataFrame.from_dict(train_file_labels, orient='index')
    train = train.reset_index(drop=False)
    train = train.rename(columns={'index': 'file', 0: 'folder'})
    train = train[['folder', 'file']]
    train = train.sort_values('file')
    train = train.reset_index(drop=True)
    def remove_label_from_file(label, fname):
        return fname[len(label)+1:]

    train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
    train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')
    test_perc = 0
    raw_train = []
    raw_dev = []
    i = 0
    for row in train[::portion].itertuples():
        i += 1
        folder = row[1]
        file = row[2]
        label = row[3]
        filename = folder + "/" + file
        which = which_set(f"{train_audio_path}/{filename}",val_perc,test_perc)
        sample_rate, samples = wavfile.read(train_audio_path + filename)

        if sample_rate != 8000:
            samples = samplerate.resample(samples, sample_rate/8000, 'sinc_best')
        #if len(samples) != 8000 : 
        #    continue
        std_samples = StandardScaler().fit_transform(samples.astype('float64').reshape(-1, 1)).reshape(-1,)
        freqs, times, spectrogram = log_specgram(std_samples, sample_rate)
        if which == 'training':
            raw_train.append((spectrogram, label))
        else:
            raw_dev.append((spectrogram,label))
        if i % 1000 == 0:
            print(f"{i} : {clock() - start} s")
        # if i == 5000:
        #     break    
    return raw_train, raw_dev




def get_test(test_audio_path, portion, part):
    start = clock() 

    train_file_labels =  os.listdir(test_audio_path)

    train = pd.DataFrame({'file':train_file_labels})
    train = train.reset_index(drop=False)
    train = train[['file']]
    train = train.sort_values('file')
    train = train.reset_index(drop=True)
    test_perc = 0
    raw_train = []
    i = 0
    length = train.shape[0]
    start_index = int((length * part) / portion)
    end_index = int((length * (part + 1)) / portion)

    for row in train[start_index:end_index].itertuples():
        i += 1
        filename = row[1]
        sample_rate, samples = wavfile.read(test_audio_path + filename)
        #if len(samples) != 8000 : 
        #    continue
        std_samples = StandardScaler().fit_transform(samples.astype('float64').reshape(-1, 1)).reshape(-1,)
        freqs, times, spectrogram = log_specgram(std_samples, sample_rate)
        raw_train.append( (filename,spectrogram) )
        if i % 1000 == 0:
            print(f"{i} : {clock() - start} s")
        # if i == 5000:
        #     break    
    return raw_train