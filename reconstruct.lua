require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'math'
require 'os'
require 'TemporalAvgPooling'
require 'helpers'

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
iterations = 1000
learningRate = 2e-5
filt_sizes = {5, 5}
nchannels = {cqt_features, 175, 125}
poolsize = 10
inpsize  = 26

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

function get_comp_lost(filt_sizes, mult) -- mult - Pass 1 for encoding, 2 for encode&decoe
    sum = 0
    for i, f in pairs(filt_sizes) do
        sum = sum + mult * (f - 1) -- 2 * for backwards pass
    end
    return sum
end

function get_desired_hinge(x1, filt_sizes, poolsize)
    sum = get_comp_lost(filt_sizes, 1)
    return 1 + torch.floor((x1:size()[1] - poolsize - sum)/(poolsize/2))
end

function get_narrow_x(x1, filt_sizes)
    sum = get_comp_lost(filt_sizes, 2)
    start = torch.floor(sum / 2)
    return x1:narrow(1, start, x1:size()[1] - sum)
end

-- Train
x1 = trainset['ix'][0]:transpose(1,2)
x2 = trainset['ix'][1]:transpose(1,2)
narrow_x1 = get_narrow_x(x1, filt_sizes)
hinge_size = get_desired_hinge(x1, filt_sizes, poolsize)
hinge_signal = torch.Tensor(hinge_size):fill(1)
assert (snet:forward({x2,x2})[2][1] == 0) -- Distance between identitical x == 0

for i = 1, iterations do
    print ('Distance', snet:forward({x1,x2})[2][1])
    gradUpdate(snet, {x1,x2}, {narrow_x1,hinge_signal}, hinge, mse, learningRate)
end
