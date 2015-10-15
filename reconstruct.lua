require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'math'
require 'os'
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
dofile('build_pdata.lua')



-- Network
 -- Parameters
filt_size = [5, 5]
nchannels = [100, 50, 20]
poolsize = 10

 -- Architecture
net = nn.Sequential()

input = nn.Identity()()
enc1 = nn.TemporalConvolution(cqt_features, nchannels[1], filt_size[1])(input)
enc2 = nn.ReLU()(enc1)
enc3 = nn.TemporalConvolution(nchannels[1], nchannels[2], filt_size[2])(enc2)
enc4 = nn.ReLU()(enc3)

pool = nn.TemporalAvgPooling(poolsize, nchannels[2])(enc4)

dec1 = nn.TemporalConvolution(nchannels[2], nchannels[1], filt_size[2])(enc4)
dec2 = nn.ReLU()(dec1)
dec3 = nn.TemporalConvolution(nchannels[1], cqt_features, filt_size[1])(dec2)
out = nn.ReLU()(dec3)

net = nn.gModule( {input}, { out, pool } )

criterion = nn.HingeEmbeddingCriterion(1) -- Argument is Margin What should this be?

-- do fwd/bwd and return loss, grad_params
function feval(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()

    -- TODO BATCHES
    local start = torch.random(trainset:size()[1] - opt.seq_length - 1) -- Extra 1 for prediction
    start = 0

    ------------------- forward pass -------------------
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    -- local predictions = {}           -- output prediction
    local predictions = torch.Tensor(cqt_features, opt.seq_length)
    local loss = 0

    -- fset = torch.add(trainset, 1)

    for t=1,opt.seq_length do
        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        -- lstm_c[t], lstm_h[t] = unpack(protos.lstm:forward{trainset[{start+t,{}}], lstm_c[t-1], lstm_h[t-1]})
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{trainset[{start+t,{}}], lstm_c[t-1], lstm_h[t-1]})
        predictions[{{},t}] = clones.output[t]:forward(lstm_h[t])
        -- loss = loss + clones.criterion[t]:forward(predictions[t], trainset[{t,{}}]) -- Test
        -- loss = loss + clones.criterion[t]:forward(predictions[t], fset[{t,{}}]) -- Test
        loss_t = clones.criterion[t]:forward(predictions[{{},t}], trainset[{start+t+1,{}}])
        loss = loss + loss_t
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss
        -- local doutput_t = clones.criterion[t]:backward(predictions[t], fset[{t,{}}]) -- Test
        -- local doutput_t = clones.criterion[t]:backward(predictions[t], trainset[{t,{}}]) -- Test
        local doutput_t = clones.criterion[t]:backward(predictions[{{},t}], trainset[{start+t+1,{}}])
        -- Two cases for dloss/dh_t:
        --   1. h_T is only used once, (not to the next LSTM timestep).
        --   2. h_t is used twice, for the prediction and for the next step. To obey the
        --      multivariate chain rule, we add them.
        if t == opt.seq_length then
            assert(dlstm_h[t] == nil)
            dlstm_h[t] = clones.output[t]:backward(lstm_h[t], doutput_t)
        else
            dlstm_h[t]:add(clones.output[t]:backward(lstm_h[t], doutput_t))
        end

        -- backprop through LSTM timestep
        print (t, trainset:size())
        dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
            {trainset[{start+t,{}}], lstm_c[t-1], lstm_h[t-1]},
            {dlstm_c[t], dlstm_h[t]}
        ))
    end

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end



-- optimization stuff
local losses = {}
local optim_state = {learningRate = 1e-3}
local iterations = opt.max_epochs
local start = os.time()
for i = 1, iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, gradnorm = %6.4e, elapsed=%d",
                i, loss[1], loss[1] / opt.seq_length, grad_params:norm(), os.difftime(os.time(), start)))
    end
end


