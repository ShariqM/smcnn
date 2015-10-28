import numpy
import scipy.io
import pdb

def get_data(name):
    direc = 'timit/TRAIN/'
    data = scipy.io.loadmat(direc + name + '.mat')
    pdb.set_trace()
    data = scipy.io.loadmat(direc + name + '.mat')['data'][0][0][0]
    return data

name = 'process/2000'
data = get_data(name)
print data.shape

pdb.set_trace()

name = 'DR1_FCJF0'
data = get_data(name)
print data.shape


name = 'DR1_FDAW0'
data2 = get_data(name)

import matplotlib.pyplot as plt
import numpy as np

plt.imshow(data[:,0:3000])
plt.tight_layout()
plt.show()

'''
grid = np.random.random((10,10))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

ax1.imshow(grid, extent=[0,100,0,1])
ax1.set_title('Default')

ax2.imshow(grid, extent=[0,100,0,1], aspect='auto')
ax2.set_title('Auto-scaled Aspect')

ax3.imshow(grid, extent=[0,100,0,1], aspect=100)
ax3.set_title('Manually Set Aspect')

plt.tight_layout()
plt.show()
'''
