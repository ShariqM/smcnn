''' Take the output of our neural network, make it correct size vector for the inverse cqt transformation'''
import scipy.io as sio
import numpy as np
import pdb

tgt_length = 140
scale_vals = 1000.

csz = 175
tsz = 1024
ver = 1

for name in ('train_pred', 'train_actual'):
    x = sio.loadmat('reconstructions/v%d/%s.mat' % (ver, name))['X1']

    batches = x.shape[0]
    batches = 10
    for i in range(batches):
        cqtv = np.zeros((csz, tsz))
        cqtv[:,:tgt_length] = x[i][0] / scale_vals
        sio.savemat('reconstructions/v%d/%s_%d_pad.mat' % (ver, name, i), {'X': cqtv})
