import numpy as np
import glob
import pdb
from helpers import cleanup_line, get_char2vec
import scipy.io as io

sentences_per_speaker = 10
longest_sentence = 80

characters = [' ', '!', '"', "'", ',', '-', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
numchars = len(characters)

char2vec = get_char2vec(characters)

def get_mfname(fname):
    path = fname.split('/')
    return 'timit/TRAIN/CHAR/%s_%s_CHAR.mat' % (path[2], path[3])

def make_files():
    mfilenames = set()
    fnames = glob.glob('timit/TRAIN/*/*/*.TXT')
    for fname in fnames:
        mfilename = get_mfname(fname)
        mfilenames.add(mfilename)
        data = np.zeros((sentences_per_speaker, numchars, longest_sentence))
        io.savemat(mfilename, {'X':data})
    io.savemat('mfilenames.mat', {'mfilenames':list(mfilenames)})

#make_files()
mfilenames = io.loadmat('mfilenames.mat')['mfilenames']
sent_idx = {}
for mfilename in mfilenames:
    sent_idx[mfilename] = 0

fnames = glob.glob('timit/TRAIN/*/*/*.TXT')
for fname in fnames:
    mfilename = get_mfname(fname)
    data = io.loadmat(mfilename)['X']
    idx = sent_idx[mfilename]

    f = open(fname, 'r')
    line = cleanup_line(f.readline())
    for i in range(len(line)):
        data[idx, :, i] = char2vec[line[i]]

    io.savemat(mfilename, {'X':data})
    sent_idx[mfilename] = idx + 1
    f.close()
