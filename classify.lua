require 'torch'
require 'nn'
matio = require 'matio'
matio.use_lua_strings = true

-- SETUP
cmd = torch.CmdLine()
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-load', 'false', 'load pre-trained neural network')
cmd:option('-net', 'dnn', 'type: dnn | cnn | lstm')
opt = cmd:parse(arg or {})

-- Load the Training and Test Set
dofile('build_data.lua')

if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   trainset.data = trainset.data:cuda()
   testset.data = testset.data:cuda()
end

if opt.load then
    net = torch.load('nets/%s.bin' % opt.net)
    print 'hi'
else
    dofile ('%s.lua' % opt.net)
end

-- LOSS
criterion = nn.ClassNLLCriterion()

-- CUDA
if opt.type == 'cuda' then
   net:cuda()
   criterion:cuda()
end

print ("Before (train):", net:forward(trainset.data[{{1},{}}]))
print ("Correct: (train)", trainset.label[1])

print ("Before (test):", net:forward(testset.data[{{80},{}}]))
print ("Correct (test):", testset.label[80])

-- TRAIN
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.05
for i=1,1 do
    trainer.maxIteration = 50 -- just do 5 epochs of training.
    trainer:train(trainset)
    torch.save('net.bin', net)
    print ('Saved %d' % i)
end

print ("After (train):", net:forward(trainset.data[{{1},{}}]))
print ("After (test):", net:forward(testset.data[{{1},{}}]))


-- TEST
