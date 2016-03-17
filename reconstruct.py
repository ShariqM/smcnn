
import scipy.io as sio
import numpy as np
import pdb

tgt_length = 135
scale_vals = 1000.

csz = 175
tsz = 1024
name = 's2_actual_white'
x = sio.loadmat('reconstructions/%s.mat' % name)

cqtv = np.zeros((csz, tsz))
cqtv[:,:tgt_length] = x['X1'][0] / scale_vals
sio.savemat('reconstructions/%s_pad.mat' % name, {'X': cqtv})
pdb.set_trace()
