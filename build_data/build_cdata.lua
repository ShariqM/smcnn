nspeakers = 10
speakers = {'FCJF0', 'FDAW0', 'FDML0', 'FECD0', 'FETB0', 'FJSP0', 'FKFB0', 'FMEM0', 'FSAH0', 'FSJK1'}
nexamples = 1000

trainset = {}
trainset['data'] = torch.DoubleTensor(nexamples, 175)
trainset['label'] = torch.ByteTensor(nexamples)

testset = {}
testset['data'] = torch.DoubleTensor(nexamples, 175)
testset['label'] = torch.ByteTensor(nexamples)

nex_per = nexamples/nspeakers
for i, speaker in ipairs(speakers) do
    data = matio.load('timit/TRAIN/DR1_%s.mat' % speaker)['data']
    start, stop = (i-1) * nex_per + 1, i * nex_per
    -- print (start, stop)
    trainset.data[{{start, stop}}] = data['X'][{{}, {1,100}}] -- 175x10240
    trainset.label[{{start, stop}}] = i

    testset.data[{{start, stop}}] = data['X'][{{}, {101,200}}] -- 175x10240
    testset.label[{{start, stop}}] = i
    -- print (speaker)
end
-- debug.debug()

setmetatable(trainset,
    {__index = function(t, i)
                   return {t.data[i], t.label[i]}
               end}
);

function trainset:size() -- SGD.train() needs this
    return self.data:size(1)
end

