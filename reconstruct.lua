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

    mem:close()
    collectgarbage()
    return clone
end

 -- Parameters
learningRate = 1e-7
filt_size = {5, 5}
nchannels = {cqt_features, 100, 50, 20}
poolsize = 10
inpsize = 20

input_x1 = nn.Identity()()
input_x2 = nn.Identity()()

function new_encoder(input)
    enc1 = nn.TemporalConvolution(nchannels[1], nchannels[2], filt_size[1])(input)
    enc2 = nn.ReLU()(enc1)
    enc3 = nn.TemporalConvolution(nchannels[2], nchannels[3], filt_size[2])(enc2)
    enc4 = nn.ReLU()(enc3)

    pool = nn.TemporalAvgPooling(poolsize, nchannels[3])(enc4)

    return enc4, pool
end

function new_decoder(input)
    dec1 = nn.TemporalConvolution(nchannels[3], nchannels[2], filt_size[2])(input)
    dec2 = nn.ReLU()(dec1)
    dec3 = nn.TemporalConvolution(nchannels[2], nchannels[1], filt_size[1])(dec2)
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
-- inet = nn.gModule({input_x1}, {output_x1})
-- snet = nn.gModule({input_x1, input_x2}, {distance})

hinge = nn.HingeEmbeddingCriterion(1)
mse   = nn.MSECriterion()

function gradUpdate(net, x, y, hinge, mse, learningRate)
    output_x1, pred = unpack(net:forward(x))
    -- output_x1 = net:forward(x)
    -- pred = net:forward(x)

    merr = mse:forward(output_x1, y[1])
    herr = hinge:forward(pred, y[2])

    gradMSE   = mse:backward(output_x1, y[1])
    gradHinge = hinge:backward(pred, y[2])

    net:zeroGradParameters()
    net:backward(x, {gradMSE, gradHinge})
    -- net:backward(x, gradHinge)
    -- net:backward(x, gradMSE)
    net:updateParameters(learningRate)
end

x1 = torch.rand(inpsize, cqt_features)
x2 = torch.rand(inpsize, cqt_features)

for i = 1, 10 do
    hack_out_x1 = torch.rand(inpsize - 4 * (filt_size[1] - 1), cqt_features) -- FIXME
    gradUpdate(snet, {x1,x2}, {hack_out_x1,1}, hinge, mse, learningRate)
    -- gradUpdate(snet, x1, {hack_out_x1,1}, hinge, mse, learningRate)
    print ('Distance', snet:forward({x1,x2})[2][1])
    print ('Distance2', snet:forward({x2,x2})[2][1])
    -- print ('Distance', snet:forward(x1))
    -- print ('Distance2', snet:forward(x1))
    -- print ('Distance', snet:forward({x1,x2})[1])
    -- print ('Distance2', snet:forward({x1,x1})[1])
end
