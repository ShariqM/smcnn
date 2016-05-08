Generative Adversarial Convolutional Neural Network for Voice Conversion / Sound Morphing

by
Shariq Mobin & Joan Bruna

# Files
    1) models/cnn.lua
        - Specifies the architecture for the encoder, decoder, and classifier
    2) convert.lua
        - This runs the network
        - 'th convert.lua -type cuda'

# CQT Code
    All the matlab code in data/wav_to_cqt/cqt_functions was taken from:
       - https://github.com/joanbruna/scattnmf

# Getting the data ready
    1) Download the GRID wav files
        - cd data/wavs/
        - ./download
    2) Download the GRID align files
        - cd data/aligns/
        - ./download
    2) Compute the CQT representation using Matlab
        - Start matlab
        - from smcnn, cd grid/wav_to_cqt
        - run 'script_fwdcqt'
        - This will take awhile... (maybe 30 hours if you do all 34 speakers)
    3) Cut up the CQT sentences into words
        - From smcnn, python data/cqt_sent_to_word.py

    The data is now ready for training.

# Evaluating the results
    1) Save the output of the network
        - th convert.lua -type cuda -init_from cv/net_analogy_0.50.t7 -pred_save
    2) Prepare the CQT vectors
        - python prepare_for_invqt.py
    3) Compute the invcqt to get the WAV
        - Start matlab
        - from smcnn, cd grid/wav_to_qt
        - run 'script_invcqt'
        - reconstructed WAVs will be in reconstructions/v1/*
