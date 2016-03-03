import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import pdb

tgt_length = 135

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
            words[word] = (nstart - offset, min(sz, nstop - offset))
        x = f.readline()
    f.close()
    return words

def insert(t, key, val):
    if not t.has_key(key):
        t[key] = [val]
    else:
        t[key].append(val)

def interp(cqtv, word, words):
    start, stop = words[word]
    cqt_word   = np.zeros((cqtv.shape[0], cqtv.shape[1]))
    cqt_word[:,start:stop] = cqtv[:, start:stop]
    cqt_word_i = np.zeros((cqtv.shape[0], cqtv.shape[1]))

    diff = stop - start

    try:
        x = np.arange(diff)
        f = interp1d(x, cqtv[:,start:stop])
    except Exception as e:
        pdb.set_trace()

    xnew = np.zeros(tgt_length)
    tmp = np.arange(0, diff - 1., (diff - 1.)/(tgt_length - 1.))
    xnew[:tmp.shape[0]] = tmp
    if tmp.shape[0] != tgt_length:
        xnew[-1] = x[-1]

    #cqt_word_i[:, start:start+tgt_length] = f(xnew)
    #sio.savemat('tmp/%s' % (word), {'X':cqt_word})
    #sio.savemat('tmp/%s_interp' % (word), {'X':cqt_word_i})
    return f(xnew)

stats = {}
data = {'S1': {}, 'S2': {}}
for speaker in (1,2):
    skey = 'S%d' % speaker
    fc = open('metadata/s%d_catalog.txt' % speaker)
    x = fc.readline()
    i = 1
    while x != '':
        fname = x[:-1]
        words = get_alignment(speaker, fname)
        mname = 'grid/grid/s%d/mat/s%d_%d.mat' % (speaker, speaker, i)
        cqtv = sio.loadmat(mname)['X']

        for word, (start,stop) in words.items():
            cqt_word = interp(cqtv, word, words)
            insert(data[skey], word, cqt_word)
            insert(stats, '', 0. +  stop - start)

        x = fc.readline()
        i = i + 1

    for word in data[skey].keys():
        data[skey][word] = np.asarray(data[skey][word])
    fc.close()

sio.savemat('grid/words/data2.mat', {'X': data})
