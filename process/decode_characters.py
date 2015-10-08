# Just a test that encode is working properly
import scipy.io as io
import pdb
from helpers import get_char2vec
import numpy as np


characters = [' ', '!', '"', "'", ',', '-', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

name = 'DR3_MADC0_CHAR.mat'
#name = 'DR1_FCJF0_CHAR.mat'
data = io.loadmat('timit/TRAIN/CHAR/%s' % name)['X']

for q in range(10):
    g = data[5,:,:]
    sentence = ''
    for i in range(g.shape[1]):
        sentence += characters[np.argwhere(g[:,i]==1)[0][0]]
        if sentence[-1] == '.': # End of sentence
            break
    print sentence
    break
