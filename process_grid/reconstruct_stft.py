import glob
import librosa
import pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.io as sio
import scipy
import h5py
from scipy.interpolate import interp1d

from helpers import process_align


sr = 25000
for i in range(10):
    name = 'train_pred_%d_pad' % i
    data = sio.loadmat('reconstructions/v10/%s.mat' % name)['X']
    pdb.set_trace()
    y_word      = librosa.istft(data)
    y_word = y_word
    librosa.output.write_wav('pred_test/%s.wav' % name, y_word, sr)
    break

