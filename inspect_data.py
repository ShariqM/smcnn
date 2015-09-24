import numpy
import scipy.io
import pdb

def get_data(name):
    direc = 'timit/TRAIN/'
    data = scipy.io.loadmat(direc + name + '.mat')['data'][0][0][0]
    pdb.set_trace()
    return data

name = 'DR1_FCJF0'
data = get_data(name)

name = 'DR1_FDAW0'
data2 = get_data(name)
pdb.set_trace()
