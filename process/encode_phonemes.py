import numpy as np
from helpers import get_x2vec
import glob
import scipy.io as io
import pdb
from math import floor, ceil

phonemes = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

speakers = ['FCJF0', 'FDAW0', 'FDML0', 'FECD0', 'FETB0', 'FJSP0', 'FKFB0', 'FMEM0', 'FSAH0', 'FSJK1', 'FSMA0', 'FTBR0', 'FVFB0', 'FVMH0', 'MCPM0', 'MDAC0', 'MDPK0', 'MEDR0', 'MGRL0', 'MJEB1', 'MJWT0', 'MKLS0', 'MKLW0', 'MMGG0', 'MMRP0', 'MPGH0', 'MPGR0', 'MPSW0', 'MRAI0', 'MRCG0', 'MRDD0', 'MRSO0', 'MRWS0', 'MTJS0', 'MTPF0', 'MTRR0', 'MWAD0', 'MWAR0']

nphonemes = len(phonemes)
nspeakers = len(speakers)

phn2vec = get_x2vec(phonemes)
spk2vec = get_x2vec(speakers)


# fnames = glob.glob('timit/TRAIN/*/*/*.PHN')
fnames = glob.glob('timit/TRAIN/DR1/*/*.PHN') # Let's just try dialect 1 for now
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

nexamples = 379
data = np.zeros((nexamples, 1024, 175))
phn_class = np.zeros((nexamples, 1024, nphonemes))
spk_class = np.zeros((nexamples, nspeakers))
spkset = set()

for i, fname in zip(range(len(fnames)), fnames):
    path = fname.split('/')
    dialect, speaker = path[2], path[3]
    spkset.add(speaker)

    mfilename = 'timit/TRAIN/%s_%s.mat' % (dialect, speaker)
    idx = sent_idx[mfilename]
    sent_idx[mfilename] = idx + 1
    tmp = io.loadmat(mfilename)['data'][0][0][0]
    data[i] = tmp[:,(idx*FP):((idx+1)*FP)].T

    bfreq, efreq = get_window(fname)

    f = open(fname, 'r')
    while True:
        x = f.readline()
        if x == '':
            break
        cqt_vect = np.zeros((175, 1024))
        pstart, pstop, phn = x.split(' ')
        phn = phn[:-1]
        pstart = int(pstart)
        pstop  = int(pstop)

        # If this phoneme ends before the beg freq or starts after end freq skip
        if pstop < bfreq or pstart > efreq:
            continue

        sbin = 0 if pstart < bfreq else floor((pstart-bfreq) / F_B)
        ebin = FP if pstop > efreq else ceil((pstop-bfreq) / F_B)
        # print (phn,sbin,ebin)
        phn_class[i][sbin:ebin][:] = phn2vec[phn]

    spk_class[i] = spk2vec[speaker]
    print i
    f.close()

i = nexamples
io.savemat('timit/TRAIN/process/DR1_data_%d.mat' % i, {'X':data})
io.savemat('timit/TRAIN/process/DR1_phn_%d.mat' % i, {'X':phn_class})
io.savemat('timit/TRAIN/process/DR1_spk_%d.mat' % i, {'X':spk_class})
# io.savemat(
# speakers = list(spkset)
# speakers.sort()
# print 'All Speakers:', speakers
# numspeakers = len(speakers)
# print 'Number of unique speakers: ', numspeakers
