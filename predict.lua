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
cmd:option('-load',       'false',  'load pre-trained neural network')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')

    -- RNN Specific
cmd:option('-rnn_size',    175,     'size of LSTM internal state')
cmd:option('-seq_length',  64,      'number of timesteps to unroll to')

    -- General
cmd:option('-max_epochs',20000, 'number of full passes through the training data')
cmd:option('-batch_size',1, 'number of sequences to train on in parallel')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-seed',123,'torch manual random number generator seed')

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

-- TODO Load nn

-- Network
local protos = {}
-- protos.embed = nn.Sequential():add(nn.Linear(cqt_features, opt.rnn_size)) -- Maybe?
protos.lstm = LSTM.lstm(opt.rnn_size)
protos.output = nn.Sequential():add(nn.Linear(opt.rnn_size, cqt_features))
protos.criterion = nn.MSECriterion()
local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.output)
params.uniform(-0.08, 0.08) -- Hmmm...

local clones = {}
    -- clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
for name,proto in pairs(protos) do
    print('cloning '..name)
    clones[name] = model_utils.clone_T_times(proto, opt.seq_length)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
-- Put everything on the GPU
if opt.type == 'cuda' then
    for name,proto in pairs(protos) do
        print('cudafying '..name)
        proto:cuda()
    end
end
local initstate_h = initstate_c:clone()


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
    ------------------- forward pass -------------------
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- output prediction
    local loss = 0

    fset = torch.add(trainset, 1)

    for t=1,opt.seq_length do
        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{trainset[{t,{}}], lstm_c[t-1], lstm_h[t-1]})
        predictions[t] = clones.output[t]:forward(lstm_h[t])
        -- loss = loss + clones.criterion[t]:forward(predictions[t], trainset[{t,{}}]) -- Test
        -- loss = loss + clones.criterion[t]:forward(predictions[t], fset[{t,{}}]) -- Test
        loss = loss + clones.criterion[t]:forward(predictions[t], trainset[{t+1,{}}])
    end

    ------------------ backward pass -------------------
    -- complete reverse order of the above
    local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
    local dlstm_h = {}                                  -- output values of LSTM
    for t=opt.seq_length,1,-1 do
        -- backprop through loss
        -- local doutput_t = clones.criterion[t]:backward(predictions[t], fset[{t,{}}]) -- Test
        -- local doutput_t = clones.criterion[t]:backward(predictions[t], trainset[{t,{}}]) -- Test
        local doutput_t = clones.criterion[t]:backward(predictions[t], trainset[{t+1,{}}])
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
            {trainset[{t,{}}], lstm_c[t-1], lstm_h[t-1]},
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
local iterations = opt.max_epochs
start = os.time()
for i = 1, iterations do
    local _, loss = optim.adagrad(feval, params, optim_state)
    losses[#losses + 1] = loss[1]

    if i % opt.save_every == 0 then
        torch.save(opt.savefile, protos)
    end
    if i % opt.print_every == 0 then
        print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e, elapsed=%d",
                i, loss[1], loss[1] / opt.seq_length, grad_params:norm(), os.difftime(os.time(), start)))
    end
end
