Convolutional Neural Network for Voice Conversion / Sound Morphing

by
Shariq Mobin & Joan Bruna

# Files

# Getting the data ready
    1) Download the GRID wav files
        - cd data/wavs/
        - ./download
    2) Compute the CQT representation using Matlab
        - Start matlab
        - from smcnn, cd grid/wav_to_cqt
        - run 'script_fwdcqt'
        - This will take awhile... (maybe 30 hours if you do all 34 speakers)
    3) Cut up the CQT sentences into words
        - From smcnn, python data/cqt_sent_to_word.py

    The data is now ready for training.
