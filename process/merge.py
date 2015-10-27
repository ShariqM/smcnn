import numpy as np
import scipy.io as io
import glob
import sys

fnames = glob.glob('timit/TRAIN/DR*.mat')
import pdb

nexamples = 1000
result = np.zeros((nexamples,1024,175))
i = 0
for fname in fnames:
    for j in range(0,10):
        print i
        a = io.loadmat(fname)
        b = a['data']
        c = b['X'][0][0]
        result[i][:][:] = c[:,(j*1024):(j+1)*1024].T
        i = i + 1
        if i >= nexamples:
            io.savemat('timit/TRAIN/%d.mat' % nexamples, {'X':result})
            sys.exit(0)

