import pdb
def process_align(fname, apply_offset=True):
    f = open(fname, 'r')

    x = f.readline() # Skip the first one (empty)
    x = f.readline()
    results = []
    offset = 0
    first_start = 0
    while x != '':
        split = x.split(' ')
        start, stop = int(split[0]), int(split[1])
        word = split[2][:-2]
        offset = start if apply_offset and offset == 0 else offset
        results.append((word, start - offset, stop - offset))
        x = f.readline()
        first_start = start if first_start == 0  else first_start
        last_start = start

        #if stop - start > 64384 and word != 'sil':
            #print (fname, word, stop-start)
    f.close()

    #print first_start
    l = (last_start - first_start) / 25000.0 * 16000.0
    if l > 65384:
       print l

    return results[:-1] # Don't need Sil at end


