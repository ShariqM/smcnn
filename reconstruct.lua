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
 -- Parameters
filt_size = {5, 5}
nchannels = {100, 50, 20}
poolsize = 10

 -- Architecture
input = nn.Identity()()

function new_encoder()
    enc1 = nn.TemporalConvolution(cqt_features, nchannels[1], filt_size[1])(input)
    enc2 = nn.ReLU()(enc1)
    enc3 = nn.TemporalConvolution(nchannels[1], nchannels[2], filt_size[2])(enc2)
    enc4 = nn.ReLU()(enc3)

    pool = nn.TemporalAvgPooling(poolsize, nchannels[2], 0)(enc4)

    return enc4, enc4 -- FIXME return pool
end

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

enc, out = new_encoder()
net = nn.gModule( {input}, {out})
clone = get_clone(net)

criterion = nn.HingeEmbeddingCriterion(1) -- Argument is Margin What should this be?
