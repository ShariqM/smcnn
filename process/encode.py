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
h_star_idx = phonemes.index('h#')

# fnames = glob.glob('timit/TRAIN/*/*/*.PHN')
fnames = glob.glob('timit/TRAIN/DR1/*/*.PHN') # Let's just try dialect 1 for now
F  = 32768 # Number of freq extracted, == 2.048 seconds @ 16kHz
FP = 1024  # Number of cqt features per WAV file
F_B = 0.0 + F / FP # Frequencies per CQT bins
cqt_features = 175
timepoints   = 1024
SA_per_spk   = 10.

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
    sent_idx[mfilename] = (0, 0)

total_examples = 380


train_prop = 8 # Number between (1-10) (10 sentences per speaker
train_examples = train_prop * nspeakers
test_examples  = (SA_per_spk - train_prop) * nspeakers

trainset = np.zeros((train_examples, timepoints, cqt_features))
testset  = np.zeros((test_examples,  timepoints, cqt_features))

phoneme_time = False
# tr_phn_label = np.zeros(train_examples, timepoints)) Later?
# te_phn_label = np.zeros(test_examples,  timepoints))

tr_spk_label = np.zeros(train_examples)
te_spk_label = np.zeros(test_examples)

tr_spk_to_idx = np.zeros((nspeakers, train_prop))
te_spk_to_idx = np.zeros((nspeakers, SA_per_spk - train_prop))

spkset = set()

tr_j = 0
te_j = 0
for i, fname in zip(range(len(fnames)), fnames):
    if i % 20 == 0:
        print 'INDEX=%d/%d' % (i, total_examples)

    path = fname.split('/')
    dialect, speaker = path[2], path[3]
    spkset.add(speaker)
    spi = speakers.index(speaker)
    assert spi != -1

    mfilename = 'timit/TRAIN/%s_%s.mat' % (dialect, speaker)
    s_tr_j, s_te_j = sent_idx[mfilename]
    idx = s_tr_j + s_te_j
    data = io.loadmat(mfilename)['data'][0][0][0]

    if s_tr_j < train_prop:
        trainset[tr_j] = data[:,(idx*FP):((idx+1)*FP)].T
        tr_spk_label[s_tr_j] = spi + 1 # Ugh lua indexing
        tr_spk_to_idx[spi][s_tr_j] = tr_j

        sent_idx[mfilename] = (s_tr_j + 1, s_te_j)
        tr_j = tr_j + 1
    else:
        testset[te_j] = data[:,(idx*FP):((idx+1)*FP)].T
        te_spk_label[s_te_j] = spi + 1 # Ugh lua indexing
        te_spk_to_idx[spi][s_te_j] = te_j

        sent_idx[mfilename] = (s_tr_j, s_te_j + 1)
        te_j = te_j + 1

    if not phoneme_time:
        continue

    f = open(fname, 'r')
    bfreq, efreq = get_window(fname)
    while True:
        x = f.readline()
        if x == '':
            break
        cqt_vect = np.zeros((cqt_features, timepoints))
        pstart, pstop, phn = x.split(' ')
        phn = phn[:-1]
        pstart = int(pstart)
        pstop  = int(pstop)

        # If this phoneme ends before the beg freq or starts after end freq skip
        if pstop < bfreq or pstart > efreq:
            continue

        sbin = 0 if pstart < bfreq else floor((pstart-bfreq) / F_B)
        ebin = FP if pstop > efreq else ceil((pstop-bfreq) / F_B)

        pi = phonemes.index(phn)
        phn_label[i][sbin:ebin] = pi + 1 # Ugh lua indexing

    # f.close()

print (tr_j, te_j)
# print (nspeakers)
io.savemat('timit/TRAIN/process/DR1_trainset.mat',  {'X':trainset})
io.savemat('timit/TRAIN/process/DR1_testset.mat',   {'X':testset})
io.savemat('timit/TRAIN/process/DR1_tr_spk.mat', {'X':tr_spk_label})
io.savemat('timit/TRAIN/process/DR1_te_spk.mat', {'X':te_spk_label})
io.savemat('timit/TRAIN/process/DR1_tr_spk_to_idx.mat', {'X':tr_spk_to_idx})
io.savemat('timit/TRAIN/process/DR1_te_spk_to_idx.mat', {'X':te_spk_to_idx})
# io.savemat('timit/TRAIN/process/DR1_phn_%d.mat' % i, {'X':phn_label})

# speakers = list(spkset)
# speakers.sort()
# print 'All Speakers:', speakers
# numspeakers = len(speakers)
# print 'Number of unique speakers: ', numspeakers
