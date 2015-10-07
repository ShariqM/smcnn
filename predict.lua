require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'os'
local LSTM = require 'lstm'
matio = require 'matio'
matio.use_lua_strings = true
local model_utils=require 'model_utils'

-- SETUP
local cmd = torch.CmdLine()
cmd:option('-type',       'double', 'type: double | float | cuda')
cmd:option('-load_net', false,  'load pre-trained neural network')

    -- RNN Specific
cmd:option('-rnn_size',    175,     'size of LSTM internal state')
cmd:option('-seq_length',  32,      'number of timesteps to unroll to')

    -- General
cmd:option('-max_epochs',1000, 'number of full passes through the training data')
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
dofile('build_pdata.lua')
cqt_features = 175

-- Network
local protos = {}
local params, grad_params
local clones = {}
if opt.load_net then
    protos = torch.load('nets/lstm.bin')
    clones = torch.load('nets/lstm_clones.bin')
    params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.output)
else
    protos = {}
    -- protos.embed = nn.Sequential():add(nn.Linear(cqt_features, opt.rnn_size)) -- Maybe?
    protos.lstm = LSTM.lstm(opt.rnn_size)
    protos.output = nn.Sequential():add(nn.Linear(opt.rnn_size, cqt_features))
    protos.criterion = nn.MSECriterion()
    params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.output)
    params.uniform(-0.08, 0.08) -- Hmmm...

    for name,proto in pairs(protos) do
        print('cloning '..name)
        clones[name] = model_utils.clone_T_times(proto, opt.seq_length)
    end
end


-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate_h = initstate_c:clone()

-- Put everything on the GPU
if opt.type == 'cuda' then
    for name,proto in pairs(protos) do
        print('cudafying '..name)
        -- proto:cuda()
    end
end

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()

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
    local predictions = torch.Tensor(opt.seq_length, 175)
    local loss = 0

    -- fset = torch.add(trainset, 1)

    for t=1,opt.seq_length do
        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{trainset[{start+t,{}}], lstm_c[t-1], lstm_h[t-1]})
        predictions[t] = clones.output[t]:forward(lstm_h[t])
        -- loss = loss + clones.criterion[t]:forward(predictions[t], trainset[{t,{}}]) -- Test
        -- loss = loss + clones.criterion[t]:forward(predictions[t], fset[{t,{}}]) -- Test
        loss_t = clones.criterion[t]:forward(predictions[t], trainset[{start+t+1,{}}])
        -- print (loss_t)
        loss = loss + loss_t
    end

    matio.save('predictions/test.mat', predictions)

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss
        -- local doutput_t = clones.criterion[t]:backward(predictions[t], fset[{t,{}}]) -- Test
        -- local doutput_t = clones.criterion[t]:backward(predictions[t], trainset[{t,{}}]) -- Test
        local doutput_t = clones.criterion[t]:backward(predictions[t], trainset[{start+t+1,{}}])
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
local optim_state = {learningRate = 1e-5}
local iterations =  opt.max_epochs
local start = os.time()
for i = 1, iterations do
    if opt.notrain then -- Messy
        break
    end

    feval(params, optim_state)
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        print 'Saving LSTM'
        torch.save('nets/lstm.bin', protos)
        torch.save('nets/lstm_clones.bin', clones)
    end

    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e, elapsed=%d",
                i, loss[1], loss[1] / opt.seq_length, grad_params:norm(), os.difftime(os.time(), start)))
    end
end

-- Reset
-- local gc = torch.zeros(opt.batch_size, opt.rnn_size)
-- local gh = initstate_c:clone()

-- Test
print("*** --- ***           *** --- ***")
print("*** --- *** Test Time *** --- ***")
print("*** --- ***           *** --- ***")
function test(params_)
    if params_ ~= params then
        params:copy(params_)
    end
    grad_params:zero()


    local loss = 0
    -- local lstm_c = initstate_c
    -- local lstm_h = initstate_h
    -- local lstm_c = {[0]=gc} -- internal cell states of LSTM
    -- local lstm_h = {[0]=gh} -- output values of LSTM
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    print (initstate_c)


    for t=1, opt.seq_length do
        x = t
        -- lstm_c, lstm_h = unpack(clones.lstm[x]:forward{trainset[{t,{}}], lstm_c, lstm_h}) -- 1 ok?
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[x]:forward{trainset[{t,{}}], lstm_c[t-1], lstm_h[t-1]})
        local prediction = clones.output[x]:forward(lstm_h[t])
        -- local prediction = clones.output[x]:forward(lstm_h)
        local loss_t = loss + clones.criterion[x]:forward(prediction, trainset[{t+1,{}}])
        loss = loss + loss_t
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f", t, loss_t, loss / t))
    end
end


test(params)
