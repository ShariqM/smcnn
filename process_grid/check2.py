import matplotlib.pyplot as plt
import numpy as np
import librosa
import pdb
import scipy.io as sio

x = sio.loadmat('grid/cqt_shariq/p2/S1.mat')
x = sio.loadmat('grid/cqt_shariq/s1/bbaf2n.mat')['X']
plt.imshow(x,  aspect='auto',
            interpolation='nearest')
plt.show()
pdb.set_trace()

y, sr = librosa.load(librosa.util.example_audio_file())

D = librosa.logamplitude(np.abs(librosa.stft(y))**2, ref_power=np.max)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')
print 'showing'
plt.show()
