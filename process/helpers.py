import numpy as np

def get_char2vec(characters):
    char2vec = {}
    for i in range(len(characters)):
        vec = np.zeros(len(characters))
        vec[i] = 1
        char2vec[characters[i]] = vec
    return char2vec

def cleanup_line(line):
    words = line.split(' ')[2:] # Remove Time stamps
    words[-1] = words[-1][:-1] # Remove Newline character
    return ' '.join(words).lower()
