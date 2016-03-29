import pdb
def process_align(fname):
    f = open(fname, 'r')

    x = f.readline() # Skip the first one
    x = f.readline()
    results = []
    offset = -1
    while x != '':
        split = x.split(' ')
        start, stop = int(split[0]), int(split[1])
        offset = start if offset == -1 else offset
        results.append((split[2][:-2], start - offset, stop - offset))
        x = f.readline()
    f.close()

    return results[:-1] # Don't need Sil at end


