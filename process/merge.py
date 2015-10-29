import numpy as np
import scipy.io as io
import glob
import sys

fnames = glob.glob('timit/TRAIN/DR*.mat')
import pdb

nexamples = 2000
result = np.zeros((nexamples,1024,175))
i = 0
for fname in fnames:
    phn_fname =
    for j in range(0,10):
        a = io.loadmat(fname)
        b = a['data']
        c = b['X'][0][0]
        result[i][:][:] = c[:,(j*1024):(j+1)*1024].T
        i = i + 1
        if i % 100 == 0:
            print i
        if i == nexamples:
            print 'start save'
            io.savemat('timit/TRAIN/process/data_%d.mat' % i, {'X':result})
            print 'saved at %d' % i
            sys.exit(0)

