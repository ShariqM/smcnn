import librosa
import scipy.io as sio
import numpy as np
import pdb


fname = 'grid/data/S1/swih7s.wav'
Fs, x = sio.wavfile.read(fname)
y, sr = librosa.load(fname, sr=25000)
y2 = librosa.resample(y, sr, 16000, scale=True)
pdb.set_trace()

x = sio.loadmat('grid/stft_data/S1.mat')['X']

data = x['soon'][0]
D = librosa.logamplitude(np.abs(data)**2, ref_power=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.show()
pdb.set_trace()
