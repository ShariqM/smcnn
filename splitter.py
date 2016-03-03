import numpy as np
import scipy.io as sio
import pdb

def get_alignment(speaker, fname):
    words = {}
    sz = 1024
    tsamples = 51200

    f = open('../GRID/data/all_align/s%d_align/%s.align' % (speaker, fname), 'r')
    x = f.readline()
    offset = 0
    while x != '':
        split = x.split(' ')
        start, stop = float(split[0]), float(split[1])
        word = split[2][:-2]
        nstart, nstop = int(sz * start/tsamples), int(sz * stop/tsamples)
        if not offset:
            offset = nstop
        else:
            if word == 'sil':
                break
            #print  "'%s': (%d, %d)," % (word, nstart - offset, nstop - offset)
            #print  nstop - nstart
            words[word] = (nstart - offset, nstop - offset)
        x = f.readline()
    f.close()
    return words


for speaker in (1,2):
    fc = open('metadata/s%d_catalog.txt' % speaker)
    x = fc.readline()
    i = 1
    while x != '':
        fname = x[:-1]
        words = get_alignment(speaker, fname)
        print words

        mname = 'grid/grid/s%d/mat/s%d_%d.mat' % (speaker, speaker, i)
        cqtv = sio.loadmat(mname)['X']

        for word, (start,stop) in words.items():
            cqt_word = np.zeros((cqtv.shape[0], cqtv.shape[1]))
            cqt_word[:,start:stop] = cqtv[:, start:stop]

        x = fc.readline()
        i = i + 1
    fc.close()

