require 'torch'
require 'nn'
local matio = require 'matio'
matio.use_lua_strings = true
-- local dbg = require("debugger")

-- SETUP
cmd = torch.CmdLine()
cmd:option('-type', 'double', 'type: double | float | cuda')
opt = cmd:parse(arg or {})


data_ready = true
-- DATA
if data_ready then -- Use this until we preprocess the TIMIT data set
    -- testset = torch.load('timit-test.t7')

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
    -- debug.debug()
else
    trainset = {}
    -- 28 Speakers * 1000 samples, 1 channel, 16kHZ (upper bound on phoneme?)
    trainset['data'] = torch.Tensor(1000 * 28, 1, 16000)
    trainset['label'] = 1 -- TODO
end

if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   trainset.data = trainset.data:cuda()
   testset.data = testset.data:cuda()
end

dofile 'dnn.lua'

-- LOSS
criterion = nn.ClassNLLCriterion()

-- CUDA
if opt.type == 'cuda' then
   net:cuda()
   criterion:cuda()
end

print ("Before (train):", net:forward(trainset.data[{{1},{}}]))
print ("Before (test):", net:forward(testset.data[{{80},{}}]))

-- TRAIN
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.5
trainer.maxIteration = 10 -- just do 5 epochs of training.
trainer:train(trainset)

print ("After (train):", net:forward(trainset.data[{{1},{}}]))
print ("After (test):", net:forward(testset.data[{{1},{}}]))


-- TEST
