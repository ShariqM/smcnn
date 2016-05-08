import glob
import numpy as np

fnames = glob.glob('*/*.align')

diffs = []
ok = 0
bad = 0
for fname in fnames:
    f = open(fname, 'r')
    x = f.readline()
    sz = 1024
    tsamples = 51200

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
            diffs.append(nstop - nstart)
            diff = nstop - nstart
            if diff < 75 or diff > 300:
                bad = bad + 1
                #print 'bad', bad
            else:
                ok = ok + 1
                #print 'ok', ok
        x = f.readline()
print "Min: %f, Max: %f, Mean:%f, Std: %f" % \
    (np.min(diffs), np.max(diffs), np.mean(diffs), np.std(diffs))
