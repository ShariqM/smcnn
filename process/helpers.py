import numpy as np

def get_x2vec(x):
    x2vec = {}
    for i in range(len(x)):
        vec = np.zeros(len(x))
        vec[i] = 1
        x2vec[x[i]] = vec
    return x2vec

def cleanup_line(line):
    words = line.split(' ')[2:] # Remove Time stamps
    words[-1] = words[-1][:-1] # Remove Newline character
    return ' '.join(words).lower()
