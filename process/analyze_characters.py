import numpy as np
import glob
import json
from helpers import cleanup_line

fnames = glob.glob('timit/TRAIN/*/*/*.TXT')

charset = set()
max_clen = -1
for fname in fnames:
    f = open(fname, 'r')
    line = cleanup_line(f.readline())
    if len(line) > max_clen:
        max_clen = len(line)
    for c in line:
        charset.add(c)
    f.close()

characters = list(charset)
characters.sort()
print 'All Characters:', characters
numchars = len(charset)
print 'Number of unique characters: ', numchars
print 'Longest sentence (in # of chars): ', max_clen
