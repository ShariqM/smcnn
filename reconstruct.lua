require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'math'
require 'os'
require 'TemporalAvgPooling'
require 'PairwiseBatchDistance'
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

dist = nn.PairwiseBatchDistance(1)({pool_x1, pool_x2})
snet = nn.gModule({input_x1, input_x2}, {output_x1, dist})

hinge = nn.HingeEmbeddingCriterion(1)
mse   = nn.MSECriterion()

math.randomseed(os.time()) -- Have to do this to get diff numbers ...
-- Train
max_length = 90 -- Must be poolsize more than max_lenght build_rdata (FIXME so ugly)
batch_size = 2
for i = 1, 100 do
    x1_batch = torch.Tensor(batch_size, max_length, cqt_features)
    x2_batch = torch.Tensor(batch_size, max_length, cqt_features)
    max_pool = get_out_length_2(max_length, filt_sizes, poolsize)

    hinge_signals = torch.Tensor(batch_size, max_pool):fill(1)
    end_length = max_length - get_comp_lost(filt_sizes, 2)
    reconstruct_signal = torch.Tensor(batch_size, end_length, cqt_features)

    for k = 1, batch_size do
        idx_1 = math.random(1, #ts.all)
        x1, x1_phn, x1_speaker, x1_tlen = unpack(ts['all'][idx_1])

        idx_2 = math.random(1, #ts.all)
        x2, x2_phn, x2_speaker, x2_tlen = unpack(ts['all'][idx_2])

        -- slen = math.min(x1:size()[1], x2:size()[1])
        tlen = math.min(x1_tlen, x2_tlen)
        slen = 8 + poolsize + (tlen - 1) * ((poolsize)/2) -- Ugly FIXME
        x1_batch[{k, {1,slen}, {}}] = x1[{{1,slen}, {}}]
        x2_batch[{k, {1,slen}, {}}] = x2[{{1,slen}, {}}]

        if x1_phn ~= x2_phn then
            hinge_signals[{k,{1,tlen}}]:fill(-1)
        end
        reconstruct_signal[{k,{},{}}] = x1_batch[{k, {1,end_length}, {}}]

    end

    idx_1 = math.random(1, #ts.all)
    x1, x1_phn, x1_speaker, x1_len = unpack(ts['all'][idx_1])
--
    idx_2 = ts.hs[x1_len][math.random(1, #ts.hs[x1_len])]
    x2, x2_phn, x2_speaker, x2_len = unpack(ts['all'][idx_2])

    narrow_x1 = get_narrow_x(x1, filt_sizes)
    hinge_signal = torch.Tensor(x1_len):fill(toInt(x1_phn == x2_phn))
    -- print ('Compare', x1_phn, x2_phn, hinge_signal[1])


    print ('Start-Distance', snet:forward({x1_batch,x2_batch})[2])
    -- print ('Start-Distance', snet:forward({x1,x2})[2][1])
    for j = 1, iterations do
        gradUpdate(snet, {x1_batch,x2_batch}, {reconstruct_signal, hinge_signals}, hinge, mse, learningRate)
        -- gradUpdate(snet, {x1,x2}, {narrow_x1,hinge_signal}, hinge, mse, learningRate)
    end
    print ('End-Distance', snet:forward({x1_batch,x2_batch})[2])

    -- print ('Start-Distance', snet:forward({x1,x2})[2][1])
    -- for j = 1, iterations do
        -- gradUpdate(snet, {x1,x2}, {narrow_x1,hinge_signal}, hinge, mse, learningRate)
    -- end
    -- print ('End-Distance', snet:forward({x1,x2})[2][1])
end

