
import scipy.io as sio
import numpy as np
import pdb

tgt_length = 135
scale_vals = 1000.

csz = 175
tsz = 1024

for name in ('s2_actual_v3', 's2_pred_v3'):
    x = sio.loadmat('reconstructions/%s.mat' % name)['X1']

    batches = x.shape[0]
    for i in range(batches):
        cqtv = np.zeros((csz, tsz))
        cqtv[:,:tgt_length] = x[i][0] / scale_vals
        sio.savemat('reconstructions/%s_%d_pad.mat' % (name, i), {'X': cqtv})
