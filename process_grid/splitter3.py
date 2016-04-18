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

tgt_length = 83

def interp(stftm, word, start, stop):
    length = stftm.shape[1]
    x = np.arange(length)
    f = interp1d(x, stftm)

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
nspks = 32 # Skip 33 and 34 (33 is weird)
Npad = 2 ** 15
for spk in range(1,nspks+1):
    skey = 'S%d' % spk
    print skey
    #h5f = h5py.File('grid/stft_data/%s.h5' % skey, 'w')
    data = {}
    fnames = glob.glob('grid/data/s%d/*.wav' % spk)

    i = 0
    for fname in fnames:
        if i % 100 == 0:
            print '\t%d' % i
        i = i + 1
        aname = fname.split('/')[-1][:-4]
        words = process_align('grid/data/all_align/s%d_align/%s.align' % (spk, aname))
        y, sr = librosa.load(fname, sr=25000)

        for (word, start, stop) in words:
            y_word = y[start:min(y.shape[0],stop)]
            cqtv = np.zeros((Npad,1), dtype=np.float16)
            cqtv[:y_word.shape[0],0] = y_word

            insert(data, word, cqtv)

    pdb.set_trace()
    sio.savemat('grid/cqt_shariq/p1/S%d.mat' % spk, {'X':data})

#sio.savemat('grid/stft_data.mat', {'X': data}, do_compression=True)
#h5f = h5py.File('grid/stft_data.h5', 'w')
#for spk in data.keys():
    #spkg = h5f.create_group(spk)
    #for word, stftms in data[spk].items():
        #wordg = spkg.create_group(word)
        #wordg.create_dataset('X', data=stftms)
#h5f.close()
pdb.set_trace()
