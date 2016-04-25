
import scipy.io as sio
import numpy as np
import pdb

# tgt_length = 83
tgt_length = 140
scale_vals = 1000.

# csz = 176
csz = 175
tsz = 1024
ver = 6

# for name in ('train_pred', 'test_pred', 'train_actual', 'test_actual'):
for name in ('train_pred', 'train_actual'):
    x = sio.loadmat('reconstructions/%s.mat' % name)['X1']
    # pdb.set_trace()

    batches = x.shape[0]
    batches = 10
    for i in range(batches):
        cqtv = np.zeros((csz, tsz))
        cqtv[:,:tgt_length] = x[i][0] / scale_vals
        sio.savemat('reconstructions/%s_%d_pad.mat' % (name, i), {'X': cqtv})
