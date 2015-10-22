
dofile('build_pdata.lua')
data = matio.load('timit/TRAIN/CHAR/DR1_%s_CHAR.mat' % speaker)['X']
char_vecs = data[{1,{},{}}]
