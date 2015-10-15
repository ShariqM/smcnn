require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'math'
require 'os'
require 'TemporalAvgPooling'
-- local LSTM = require 'lstm'
local LSTM = require 'lstm'
matio = require 'matio'
matio.use_lua_strings = true
local model_utils=require 'model_utils'

-- SETUP
local cmd = torch.CmdLine()
cmd:option('-type',       'double', 'type: double | float | cuda')
cmd:option('-load_net', false,  'load pre-trained neural network')

    -- General
cmd:option('-max_epochs', 1000, 'number of full passes through the training data')
cmd:option('-batch_size',1, 'number of sequences to train on in parallel')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-notrain', false,'Don\'t train network')

local opt = cmd:parse(arg or {})

-- CUDA
if opt.type == 'float' then
    print('==> switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
    print('==> switching to CUDA')
    require 'cunn'
    torch.setdefaulttensortype('torch.FloatTensor')
end


-- Load the Training and Test Set
cqt_features = 175
dofile('build_rdata.lua')


-- Network
function get_clone(net)
    -- Write and Read the network in order to copy it
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    local params, grad_params             = net:parameters()
    local clone_params, clone_grad_params = clone:parameters()
    for i = 1, #params do
        clone_params[i]:set(params[i])
        -- Don't copy the gradients otherwise they wipe eachother out
    end

    collectgarbage()
    return clone
end

 -- Parameters
filt_size = {5, 5}
nchannels = {cqt_features, 100, 50, 20}
poolsize = 10
inpsize = 20

input = nn.Identity()()
input2 = nn.Identity()()

 -- Architecture
function new_encoder()
    enc1 = nn.TemporalConvolution(nchannels[1], nchannels[2], filt_size[1])(input)
    enc2 = nn.ReLU()(enc1)
    enc3 = nn.TemporalConvolution(nchannels[2], nchannels[3], filt_size[2])(enc2)
    enc4 = nn.ReLU()(enc3)

    pool = nn.TemporalAvgPooling(poolsize, nchannels[3])(enc4)

    return enc4, pool -- FIXME return pool
end

enc, out = new_encoder()
net = nn.gModule({input}, {out})
clone = get_clone(net)

i1 = nn.Identity()()
i2 = nn.Identity()()
distance = nn.PairwiseDistance(1)({i1, i2})
stable = nn.gModule({i1, i2}, {distance})

criterion = nn.HingeEmbeddingCriterion(1) -- Argument is Margin What should this be?

x = torch.rand(inpsize, cqt_features)
y = torch.rand(inpsize, cqt_features)

-- Use a typical generic gradient update function
function gradUpdate(net, x, y, criterion, learningRate)
    local pred = net:forward(x)
    local err = criterion:forward(pred, y)
    local gradCriterion = criterion:backward(pred, y)
    net:zeroGradParameters()
    net:backward(x, gradCriterion)
    net:updateParameters(learningRate)
end

for i = 1, 1000  do
    -- gradUpdate(stable, {x, y}, 1, criterion, 0.01)
    lx = net:forward(x) -- 1x50 (averaged over time points for 50 channels)
    ly = clone:forward(y)
    pred = stable:forward({lx, ly}) -- Computes norm and gets a scalar
    -- print ('disiance', pred)
    print (pred[1])

    err = criterion:forward(pred, 1) --- Scalar
    grad_criterion = criterion:backward(pred, 1) -- == 1 ?
    stable:zeroGradParameters()
    net:zeroGradParameters()
    grad_distance = stable:backward({lx,ly}, grad_criterion) -- 2 1x50 vectors
    grad_net = net:backward(x, grad_distance[1]) -- Don't care about other gradient?
    net:updateParameters(1e-7)

    -- print('distance', stable:forward({lx, ly}))
end



--[[
distance = nn.PairwiseDistance(1)({net(), clone()}) -- Is this right?
stable = nn.gModule({input, input2}, {distance})

criterion = nn.HingeEmbeddingCriterion(1) -- Argument is Margin What should this be?

x = torch.rand(cqt_features)
y = torch.rand(cqt_features)

-- Use a typical generic gradient update function
function gradUpdate(net, x, y, criterion, learningRate)
    local pred = net:forward(x)
    local err = criterion:forward(pred, y)
    local gradCriterion = criterion:backward(pred, y)
    net:zeroGradParameters()
    net:backward(x, gradCriterion)
    net:updateParameters(learningRate)
end

for i = 1, 10 do
   gradUpdate(stable, {x, y}, 1, criterion, 0.01)
   print(stable:forward({x, y})[1])
end

--]]
