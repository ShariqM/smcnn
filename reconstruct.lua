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

-- Parameters
cqt_features = 175
iterations = 100
learningRate = 1e-4
filt_sizes = {5, 5}
nchannels = {cqt_features, 175, 125}
poolsize = 10

-- Load the Training and Test Set
dofile('build_rdata.lua')

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
for i = 1, 100 do
    idx_1 = math.random(1, #ts.all)
    x1, x1_phn, x1_speaker, x1_len = unpack(ts['all'][idx_1])

    idx_2 = ts.hs[x1_len][math.random(1, #ts.hs[x1_len])]
    x2, x2_phn, x2_speaker, x2_len = unpack(ts['all'][idx_2])

    narrow_x1 = get_narrow_x(x1, filt_sizes)
    hinge_signal = torch.Tensor(x1_len):fill(toInt(x1_phn == x2_phn))
    print ('Compare', x1_phn, x2_phn, hinge_signal[1])

    print ('Start-Distance', snet:forward({x1,x2})[2][1])
    for j = 1, iterations do
        gradUpdate(snet, {x1,x2}, {narrow_x1,hinge_signal}, hinge, mse, learningRate)
    end
    print ('End-Distance', snet:forward({x1,x2})[2][1])
end

