''' This code splits up CQT sentences in words and saves them as a dictionary in a .mat file'''
import glob
import librosa
import pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.io as sio
import scipy
import h5py
import math
from scipy.interpolate import interp1d

from helpers import process_align

tgt_length = 140 # This is the mean length of a word

def interp(cqt_word):
    '''Interpolate so we have a fixed size input for all words'''
    length = cqt_word.shape[1]
    x = np.arange(length)
    f = interp1d(x, cqt_word)

    xnew = np.zeros(tgt_length)
    tmp = np.arange(0, length - 1., (length - 1.)/(tgt_length - 1.))
    xnew[:tmp.shape[0]] = tmp
    if tmp.shape[0] != tgt_length:
        xnew[-1] = x[-1]

    return f(xnew)

def insert(t, key, val):
    if not t.has_key(key):
        t[key] = [val]
    else:
        t[key].append(val)

mytype = np.float32
nspks = 15 # Skip 33 and 34 (33 is weird)
Npad = 2 ** 15
cqt_time_bin = .002 # 2 ms
Fs = 25000.
lengths = []
scale = 1000. # Easier to learn in the scaled space.

for spk in range(13,nspks+1):
    skey = 'S%d' % spk
    print skey
    data = {}
    fnames = glob.glob('data/cqt_data/s%d/*.mat' % spk)

    i = 0
    for fname in fnames:
        if i % 100 == 0:
            print '\t%d' % i
        i = i + 1
        aname = fname.split('/')[-1][:-4]
        words = process_align('grid/data/all_align/s%d_align/%s.align' % (spk, aname))
        y = sio.loadmat(fname)['X']

        for (word, start, stop) in words:
            start = math.floor((start / Fs) / cqt_time_bin)
            stop = math.ceil((stop / Fs) / cqt_time_bin)
            cqt_word = interp(y[:,start:stop+1])
            insert(data, word, cqt_word)

    for word, cqt_word in data.items():
        data[word] = np.asarray(cqt_word)
        data[word] = data[word] = data.word[data.word < 0] = 0 # Threshold at 0 b/c ReLu (all close to 0 anyway)
        data[word] = scale * data[word]
    sio.savemat('data/cqt_data/data/s%d.mat' % spk, {'X':data})
