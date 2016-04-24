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

tgt_length = 140

def interp(cqt_word):
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

for spk in range(13,nspks+1):
    skey = 'S%d' % spk
    print skey
    #h5f = h5py.File('grid/stft_data/%s.h5' % skey, 'w')
    data = {}
    fnames = glob.glob('grid/cqt_shariq/s%d/*.mat' % spk)

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

            #cqtv = np.zeros((175, 1024))
            #cqtv[:,:140] = cqt_word
            #sio.savemat('test2/%s.mat' % word, {'X': cqtv})
            #pdb.set_trace()
            #lengths.append(stop - start)

    #if spk == 3:
        #lengths = np.asarray(lengths)
        #pdb.set_trace()
    for word, cqt_word in data.items():
        data[word] = np.asarray(cqt_word)
    sio.savemat('grid/cqt_shariq/data/s%d.mat' % spk, {'X':data})

#sio.savemat('grid/stft_data.mat', {'X': data}, do_compression=True)
#h5f = h5py.File('grid/stft_data.h5', 'w')
#for spk in data.keys():
    #spkg = h5f.create_group(spk)
    #for word, stftms in data[spk].items():
        #wordg = spkg.create_group(word)
        #wordg.create_dataset('X', data=stftms)
#h5f.close()
pdb.set_trace()
