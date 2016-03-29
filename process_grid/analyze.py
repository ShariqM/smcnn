import numpy as np
import glob
import pdb

def process_align(fname):
    f = open(fname, 'r')

    x = f.readline() # Skip the first one
    x = f.readline()
    results = []
    while x != '':
        split = x.split(' ')
        start, stop = float(split[0]), float(split[1])
        results.append((split[2], start, stop))
    f.close()

    return results


fnames = glob.glob("grid/data/all_align/*/*.align")
longest = 0
for fname in fnames:
    results = process_align(fname)

    first_start = results[0][0]
    last_start  = results[-1][0]

    if last_start - first_start > longest:
        longest = last_start - first_start
        long_fname = fname

print (long_fname, longest)
