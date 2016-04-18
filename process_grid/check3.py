import matplotlib.pyplot as plt
import numpy as np
import librosa
import pdb
import scipy.io as sio

data = sio.loadmat('grid/cqt_shariq/data/s1.mat')['X']

s7 = data[0,0]['seven'][0]
cqtv = np.zeros((175,1024))
cqtv[:,:140] = s7
sio.savemat('test2/s7.mat', {'X': cqtv})
pdb.set_trace()
