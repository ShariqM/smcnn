import scipy.io as io
import pdb
from helpers import get_char2vec
import numpy as np


characters = [' ', '!', '"', "'", ',', '-', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

data = io.loadmat('timit/TRAIN/CHAR/DR3_MADC0_CHAR.mat')['X']

g = data[1,:,:]
pdb.set_trace()
sentence = ''
for i in range(g.shape[1]):
    sentence += characters[np.argwhere(g[:,i]==1)[0][0]]
    if sentence[-1] == '.': # End of sentence
        break
print sentence
