import glob
import librosa
import pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d

from helpers import process_align

tgt_length = 83

def interp(stftm, word, start, stop):
    length = stftm.shape[1]

    try:
        x = np.arange(length)
        f = interp1d(x, stftm)
    except Exception as e:
        pdb.set_trace()

    xnew = np.zeros(tgt_length)
    tmp = np.arange(0, length - 1., (length - 1.)/(tgt_length - 1.))
    xnew[:tmp.shape[0]] = tmp
    if tmp.shape[0] != tgt_length:
        xnew[-1] = x[-1]

    return f(xnew)

data = {}
lengths = []
for spk in range(1,34):
    print spk
    data['S%d' % spk] = {}
    fnames = glob.glob('grid/data/s%d/*.wav' % spk)

    i = 0
    for fname in fnames:
        if i > 1:
            break
        i = i + 1
        aname = fname.split('/')[-1][:-4]
        words = process_align('grid/data/all_align/s%d_align/%s.align' % (spk, aname))

        y, sr = librosa.load(fname, sr=25000)
        for (word, start, stop) in words:
            #print word, start, stop
            y_word = y[start:min(y.shape[0],stop)]
            stftm = librosa.stft(y_word, n_fft=175 * 2)
            stftm_interp = interp(stftm, word, start, stop)

            print stftm_interp.shape
            #librosa.display.specshow(stft_interp)
            #plt.show()
            stftm_interp.imag = np.zeros((stftm_interp.shape[0], stftm_interp.shape[1]))
            lengths.append(stftm_interp.shape[1])
            y_word      = librosa.istft(stftm_interp)
            librosa.output.write_wav('test/rtest_%d_%s.wav' % (spk, word), y_word, sr)
pdb.set_trace()
