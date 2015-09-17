require 'torch'
require 'nn'

-- SETUP
cmd = torch.CmdLine()
cmd:option('-type', 'double', 'type: double | float | cuda')
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- DATA
if true then -- Use this until we preprocess the TIMIT data set
    trainset = {}
    -- 28 Speakers * 1000 samples, 1 channel, 16kHZ (upper bound on phoneme?)
    trainset['data'] = torch.Tensor(1000 * 28, 1, 16000)
    trainset['label'] = :w
else
    trainset = torch.load('timit-train.t7')
    testset = torch.load('timit-test.t7')
end


-- NETWORK
num_conv_layers = 2
filt_sizes = {5, 5}
pool_sizes = {2,2}
channels = {6,16}
channels[0] = 1 -- Input Channels (from data)

net = nn.Sequential()
for i = 1, num_conv_layers, do
    net:add(nn.SpatialConvolution(channels[i-1], channels[i], filt_sizes[i][0], filt_sizes[i][1])) -- MM?
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(pool[i][0], pool[i][1]))
end

neurons = channels[num_conv_layers] * result_size
-- TODO
net:add(nn.View(neurons))
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 10))
net:add(nn.LogSoftMax())

print('Lenet5\n' .. net:__tostring());

-- LOSS


-- CUDA
if opt.type == 'cuda' then
   net:cuda()
   -- criterion:cuda() TODO
end

-- TEST

