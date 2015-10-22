nexamples = 1024 -- SA1
speaker = 'FCJF0'

trainset = torch.Tensor(nexamples, cqt_features)

testset = torch.Tensor(nexamples, cqt_features)

data = matio.load('timit/TRAIN/DR1_%s.mat' % speaker)['data']
trainset[{{1, nexamples}}] = data['X'][{{}, {1, nexamples}}] -- 175x10240
testset[{{1, nexamples}}] = data['X'][{{}, {nexamples + 1, nexamples * 2}}] -- 175x10240

-- require 'image'
-- testset:div(testset:mean())
-- image.save('junk/trainset.png', testset:transpose(1,2))
