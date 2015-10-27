
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'PairwiseBatchDistance'

matio = require 'matio'
matio.use_lua_strings = true
local model_utils=require 'model_utils'
local LSTM = require 'models.lstm'
local RNN = require 'models.rnn'
local TimitBatchLoader = require 'TimitBatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a speech auto-encoding model')
cmd:text()
cmd:text('Options')
-- data
-- model params
cmd:option('-rnn_size', 176, 'size of LSTM internal state')
cmd:option('-num_layers', 3, 'number of layers in the LSTM')
cmd:option('-model', 'rnn', 'lstm or rnn')
cmd:option('-pool_size', 2, 'Pool window on hidden state') -- Need to work in stride
-- optimization
cmd:option('-iters',100,'iterations per epoch')
cmd:option('-learning_rate',1e-8,'learning rate')
cmd:option('-learning_rate_decay',0.90,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every',200,'Save every $1 iterations')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-type',       'double', 'type: double | float | cuda')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- CUDA
if opt.type == 'float' then
    print('==> switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
    print('==> switching to CUDA')
    require 'cunn'
    torch.setdefaulttensortype('torch.FloatTensor') -- Not sure why I do this
end

-- Load the Training and Test Set
cqt_features = 175
local loader = TimitBatchLoader.create(cqt_features, opt.batch_size, opt.seq_length)

local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos

    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false
else

    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(cqt_features, opt.rnn_size, opt.pool_size,
                               opt.num_layers, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(cqt_features, opt.rnn_size, opt.pool_size,
                             opt.num_layers, opt.dropout)
    end
    protos.criterion_rct = nn.MSECriterion()
    protos.criterion_stb = nn.MSECriterion() -- L2 ok?
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.type == 'cuda' then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then table.insert(init_state, h_init:clone()) end
end

-- Put everything on the GPU
if opt.type == 'cuda' then
    for name,proto in pairs(protos) do
        print('cudafying '..name)
        proto:cuda()
    end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
if do_random_init then
    params:uniform(-0.08, 0.08) -- small uniform numbers
    -- params:uniform(-1e-5, 1e-5)
end


-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx then
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
            end
        end
    end
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_T_times(proto, opt.seq_length, not proto.parameters)
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state) -- Not sure why we need clone...
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    x = loader:next_batch()
    if opt.type == 'cuda' then x = x:float():cuda() end -- Ship to GPU | Need Float?

    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local pool_state = {}
    local reconstructions = {}
    local loss = 0

    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[{{},t,{}}], unpack(rnn_state[t-1])}

        rnn_state[t] = {}
        for i=1, #init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

        pool_state[t] = lst[#lst]
        -- print (pool_state[t]:size())
        reconstructions[t] = lst[#lst - 1] -- second to last element is the reconstruction

        loss = loss + clones.criterion_rct[t]:forward(reconstructions[t], x[{{},t,{}}])

        prev_pool = pool_state[t]
        if t > 3 then
            prev_pool = pool_state[t-1]
        end
        loss = loss + clones.criterion_stb[t]:forward(prev_pool, pool_state[t]) -- TODO no err erly
    end
    -- print ('Mean Pool', pool_state[opt.seq_length]:mean(4):mean() / x[{{},opt.seq_length, {}}]:mean(1):mean())
    t = opt.seq_length
    -- reconstructions[t]:fill(0)
    -- reconstructions[t]:fill(1e-8)
    -- reconstructions[t] = reconstructions[t] + x[{{},t,{}}]
    -- print (reconstructions[t][1]:size())
    -- print (
    -- r = reconstructions[t][1]
    -- o = x[{{},t,{}}][1]
    -- print (string.format('%s: %.2fdB', 'Recons:', -10 * math.log10(math.pow((o - r):norm(), 2) / (math.pow(o:norm(), 2)))))
    -- r = reconstructions[t][opt.batch_size]
    -- o = x[{{},t,{}}][opt.batch_size]
    -- print (string.format('%s: %.2fdB', 'Recons:', -10 * math.log10(math.pow((o - r):norm(), 2) / (math.pow(o:norm(), 2)))))
    print (string.format('%s: %.2fdB', 'Recons:', -10 * math.log10(math.pow((x[{{},t,{}}] - reconstructions[t]):norm(),2)/(math.pow(x[{{},t,{}}]:norm(), 2)))))
    -- print ('SNR Recons:', -10 * math.log10(((x[{{},t,{}}] - reconstructions[t]):norm())/(x[{{},t,{}}]:norm())))

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion_rct[t]:backward(reconstructions[t], x[{{},t,{}}])

        prev_pool = pool_state[t]
        if t > 3 then
            prev_pool = pool_state[t-1]
        end
        doutput2_t = clones.criterion_stb[t]:backward(prev_pool, pool_state[t])
        doutput2_t:fill(0) -- FIXME No Influence for now

        table.insert(drnn_state[t], doutput_t)
        table.insert(drnn_state[t], doutput2_t)
        local dlst = clones.rnn[t]:backward({x[{{},t,{}}], unpack(rnn_state[t-1])}, drnn_state[t])

        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- grad_params:div(opt.seq_length) -- this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil
for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if (i % opt.save_every == 0 or i == iterations) then
        local savefile = string.format('%s/lm_%s_epoch%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        -- break -- halt
    end
end
