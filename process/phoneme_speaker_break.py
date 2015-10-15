import numpy as np
import glob
import json
from helpers import cleanup_line
import scipy.io as io
import pdb
from math import floor, ceil

fnames = glob.glob('timit/TRAIN/*/*/*.PHN')
fnames = glob.glob('timit/TRAIN/DR1/FCJF0/*.PHN')
F  = 32768 # Number of freq extracted, == 2.048 seconds @ 16kHz
FP = 1024  # Number of cqt features per WAV file
F_B = 0.0 + F / FP # Frequencies per CQT bins

# Returns the (beginning, end) frequencies in the cqt vector
def get_window(fname):
    txtfname = fname[:-3] + 'TXT'
    f = open(txtfname, 'r')
    total = int(f.readline().split(' ')[1])
    gap = (total - F) / 2.
    f.close()
    return gap, total - gap

mfilenames = glob.glob('timit/TRAIN/*.mat')
sent_idx = {}
for mfilename in mfilenames:
    sent_idx[mfilename] = 0


collect = {}
def update_dict(dialect, speaker):
    if not collect.has_key(dialect):
        collect[dialect] = {}
    if not collect[dialect].has_key(speaker):
        collect[dialect][speaker] = {}

def update_dict_2(phn):
    if not collect[dialect][speaker].has_key(phn):
        collect[dialect][speaker][phn] = []

for fname in fnames:
    path = fname.split('/')
    dialect, speaker = path[2], path[3]

    mfilename = 'timit/TRAIN/%s_%s.mat' % (dialect, speaker)
    idx = sent_idx[mfilename]
    data = io.loadmat(mfilename)['data'][0][0][0]

    update_dict(dialect, speaker)
    bfreq, efreq = get_window(fname)

    f = open(fname, 'r')
    while True:
        x = f.readline()
        if x == '':
            break
        cqt_vect = np.zeros((175, 1024))
        pstart, pstop, phn = x.split(' ')
        phn = phn[:-1]
        if phn == 'h#':
            continue
        pstart = int(pstart)
        pstop  = int(pstop)
        update_dict_2(phn)

        # If this Phoneme starts and ends in the window save it
        if pstart > bfreq and pstop < efreq:
            sbin = floor((pstart-bfreq) / F_B)
            ebin = ceil((pstop-bfreq) / F_B)

            b,e = FP * idx + sbin, FP * idx + ebin
            l = e - b
            collect[dialect][speaker][phn].append(data[:,b:e])

    sent_idx[mfilename] = idx + 1
    f.close()



best_key = 'G'
longest = -1
for phn, data in collect[dialect][speaker].items():
    if len(data) > longest:
        best_key = phn
        longest = len(data)
#print best_key, longest

#for phn,data in collect[dialect][speaker].items():
    #collect[dialect][speaker][phn] =  {'%d' % i: data[i] for i in range(len(data))}
#
#pdb.set_trace()
#io.savemat('timit/TRAIN/PHN_SPK/phn_spk.mat', collect)

sdict = {}
for phn,data in collect[dialect][speaker].items():
    for i in range(len(data)):
        sdict['%s_%s_%s_%d' % (dialect, speaker, phn, i)] = data[i]

io.savemat('timit/TRAIN/PHN_SPK/phn_spk.mat', sdict)

#for phn,data in collect[dialect][speaker].items():
    #sfile = 'timit/TRAIN/PHN_SPK/%s_%s_%s' % (dialect, speaker, phn)
    #dict_version = {'%d' % i: data[i] for i in range(len(data))}
    #io.savemat(sfile, {'G': dict_version})
#
#g = io.loadmat(sfile)
#pdb.set_trace()
