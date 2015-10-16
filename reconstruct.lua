require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'math'
require 'os'
require 'TemporalAvgPooling'
require 'helpers'
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
learningRate = 1e-7
filt_sizes = {5, 5}
nchannels = {cqt_features, 100, 50, 20}
poolsize = 10
inpsize = 20

    -- Architecture
input_x1 = nn.Identity()()
input_x2 = nn.Identity()()

enc_x1, pool_x1 = new_encoder(input_x1, nchannels, filt_sizes, poolsize)
enc_x2, pool_x2 = new_encoder(input_x2, nchannels, filt_sizes, poolsize)
tie_weights(enc_x1, enc_x2)
output_x1 = new_decoder(enc_x1, nchannels, filt_sizes)

dist = nn.PairwiseDistance(1)({pool_x1, pool_x2})
snet = nn.gModule({input_x1, input_x2}, {output_x1, dist})

hinge = nn.HingeEmbeddingCriterion(1)
mse   = nn.MSECriterion()

-- Train
x1 = torch.Tensor(inpsize, cqt_features)
x2 = torch.Tensor(inpsize, cqt_features)
narrow_x1 = get_narrow_x(x1, filt_sizes)

for i = 1, 10 do
    gradUpdate(snet, {x1,x2}, {narrow_x1,1}, hinge, mse, learningRate)
    print ('Distance', snet:forward({x1,x2})[2][1])
    print ('Distance2', snet:forward({x2,x2})[2][1])
end