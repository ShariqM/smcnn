import glob
import librosa
import pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from helpers import process_align

data = {}
for spk in range(1,34):
    data['S%d' % spk] = {}
    fnames = glob.glob('grid/data/s%d/*.wav' % spk)

    for fname in fnames:
        aname = fname.split('/')[-1][:-4]
        words = process_align('grid/data/all_align/s%d_align/%s.align' % (spk, aname))

        rosa = True
        if rosa:
            y, sr = librosa.load(fname, sr=25000)
            #y, sr = librosa.load(fname)
        else:
            sr, y = wavfile.read(fname)
        for (word, start, stop) in words:
            print word, start, stop
            y_word = y[start:min(y.shape[0],stop)]
            stft_matrix = librosa.stft(y_word)
            stft_matrix.imag = np.zeros((stft_matrix.shape[0], stft_matrix.shape[1]))
            y_word      = librosa.istft(stft_matrix)


            if rosa:
                librosa.output.write_wav('test/rtest_%s.wav' % word, y_word, sr)
            else:
                wavfile.write('test/wtest_%s.wav' % word, sr, y_word)
        pdb.set_trace()


        #stft_matrix = librosa.stft(y)
        #librosa.display.specshow(stft_matrix)
        #stft_matrix.imag = np.ones((stft_matrix.shape[0], stft_matrix.shape[1]))
        #plt.show()
        #y_hat       = librosa.istft(stft_matrix)
        #pdb.set_trace()
        #librosa.output.write_wav(fname + "_recon.wav", y_hat, sr)
