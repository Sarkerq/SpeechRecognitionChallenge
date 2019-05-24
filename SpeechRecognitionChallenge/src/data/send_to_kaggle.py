import tensorflow as tf
import scipy as sp
import numpy as np
import pandas as pd
import re
import hashlib
from sklearn.preprocessing import StandardScaler

import os
from pathlib import Path
import IPython.display as ipd
import cProfile

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
%matplotlib inline


from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras.utils import to_categorical

from sklearn.preprocessing import OrdinalEncoder

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Input, Dense, LSTM

from tensorflow.python.keras.layers import Embedding, Dropout, Activation

import sys
from tensorflow.python.keras.models import load_model
def send_to_kaggle(model_path,parts):
    model = load_model(model_path)
    raw_train, raw_dev = get_train_test.get_train_test('../../data/raw/train/audio/',10,12)

    x_train = np.array([np.pad(x[0], ((0,99 - x[0].shape[0]),(0,0)), 'constant', constant_values=(0)) for x in raw_train])
    x_dev = np.array([np.pad(x[0], ((0,99 - x[0].shape[0]),(0,0)), 'constant', constant_values=(0)) for x in raw_dev])

    y_train = np.array([x[1] for x in raw_train])
    y_dev = np.array([x[1] for x in raw_dev])

    x_resh = x_train.reshape(-1, 99*161)

    scaler = StandardScaler()

    scaler.fit(x_resh)

    for i in range(0,parts):
        print (i)
        raw_test = get_train_test.get_test('../../data/raw/test/audio/',parts,i)

        test_fnames = np.array([x[0] for x in raw_test])

        x_test = np.array([np.pad(x[1], ((0,99 - x[1].shape[0]),(0,0)), 'constant', constant_values=(0)) for x in raw_test])

        x_test_scaled = scaler.transform(x_test.reshape(-1,99*161)).reshape(-1,99,161)

        x_pred = model.predict(x_test_scaled)

        x_best_pred = np.argmax(x_pred,axis=1)

        labels = [ 'down', 'go', 'left', 'no',  'off','on','right', 'stop','unknown', 'up', 'yes', ]

        f = lambda x: labels[x]
        f = np.vectorize(f)
        response = f(x_best_pred)

        total_response = np.concatenate([test_fnames.reshape(test_fnames.shape[0],-1),response.reshape(test_fnames.shape[0],-1)],axis = 1)

        f=open('resp.csv','ab')
        np.savetxt(f, total_response, delimiter=",", fmt='%s')
        f.close()