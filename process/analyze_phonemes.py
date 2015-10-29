import numpy as np
import glob
import json
from helpers import cleanup_line

# fnames = glob.glob('timit/TRAIN/*/*/*.PHN')
fnames = glob.glob('timit/TRAIN/DR1/*/*.PHN')

phnset = set()
max_clen = -1
for fname in fnames:
    f = open(fname, 'r')

    line = f.readline()
    while line != '':
        phn = line.split(' ')[2][:-1] # Remove Time stamps and newline
        phnset.add(phn)
        line = f.readline()
    f.close()

phonemes = list(phnset)
phonemes.sort()
print 'All Phonemes:', phonemes
numphonemes = len(phonemes)
print 'Number of unique phonemes: ', numphonemes
