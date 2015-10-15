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
learningRate = 1e-7
filt_sizes = {5, 5}
nchannels = {cqt_features, 100, 50, 20}
poolsize = 10
inpsize = 40

input_x1 = nn.Identity()()
input_x2 = nn.Identity()()

function new_encoder(input)
    enc1 = nn.TemporalConvolution(nchannels[1], nchannels[2], filt_sizes[1])(input)
    enc2 = nn.ReLU()(enc1)
    enc3 = nn.TemporalConvolution(nchannels[2], nchannels[3], filt_sizes[2])(enc2)
    enc4 = nn.ReLU()(enc3)

    pool = nn.TemporalAvgPooling(poolsize, nchannels[3])(enc4)

    return enc4, pool
end

function new_decoder(input)
    dec1 = nn.TemporalConvolution(nchannels[3], nchannels[2], filt_sizes[2])(input)
    dec2 = nn.ReLU()(dec1)
    dec3 = nn.TemporalConvolution(nchannels[2], nchannels[1], filt_sizes[1])(dec2)
    dec4 = nn.ReLU()(dec3)
    return dec4
end

function tie(input_x1, x1, input_x2, x2)
    px1 = x1
    px2 = x2
    while true do -- Walk the children copying the parameters as you go
        params_px1 = px1.data.module:parameters()
        params_px2 = px2.data.module:parameters()

        if params_px1 then
            for i=1, #params_px1 do
                params_px2[i]:set(params_px1[i])
            end
        end

        if #px1.children == 0 then
            break
        end
        px1 = px1.children[1]
        px2 = px2.children[1]
    end
end

enc_x1, pool_x1 = new_encoder(input_x1)
enc_x2, pool_x2 = new_encoder(input_x2)
tie(input_x1, enc_x1, input_x2, enc_x2)
output_x1 = new_decoder(enc_x1)

distance = nn.PairwiseDistance(1)({pool_x1, pool_x2})
snet = nn.gModule({input_x1, input_x2}, {output_x1, distance})

hinge = nn.HingeEmbeddingCriterion(1)
mse   = nn.MSECriterion()

function gradUpdate(net, x, y, hinge, mse, learningRate)
    output_x1, pred = unpack(net:forward(x))

    merr = mse:forward(output_x1, y[1])
    herr = hinge:forward(pred, y[2])

    gradMSE   = mse:backward(output_x1, y[1])
    gradHinge = hinge:backward(pred, y[2])

    net:zeroGradParameters()
    net:backward(x, {gradMSE, gradHinge})
    net:updateParameters(learningRate)
end

function get_narrow_x(x1, filt_sizes)
    sum = 0
    for i, f in pairs(filt_sizes) do
        sum = sum + 2 * (f - 1) -- 2 * for backwards pass
    end
    start = torch.floor(sum / 2)
    print (sum, start, sum)
    return x1:narrow(1, start, x1:size()[1] - sum)
end

x1 = torch.Tensor(inpsize, cqt_features)
x2 = torch.Tensor(inpsize, cqt_features)
narrow_x1 = get_narrow_x(x1, filt_sizes)

for i = 1, 10 do
    gradUpdate(snet, {x1,x2}, {narrow_x1,1}, hinge, mse, learningRate)
    print ('Distance', snet:forward({x1,x2})[2][1])
    print ('Distance2', snet:forward({x2,x2})[2][1])
end
