require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'helpers'

matio = require 'matio'
matio.use_lua_strings = true
local model_utils=require 'model_utils'
local CNN2 = require 'models.CNN2'
local Difference = require 'models.difference'
local GridSpeechBatchLoader = require 'GridSpeechBatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a speech conversion model')
cmd:text()
cmd:text('Options')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-iters',200,'iterations per epoch')

cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-batch_size',64,'number of sequences to train on in parallel')
cmd:option('-dropout',0,'dropout for regularization, used after each CNN hidden layer. 0 = no dropout')

cmd:option('-save_pred',false,'Save prediction')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-learning_rate',1,'learning rate')
cmd:option('-learning_rate_decay',0.98,'learning rate decay')
cmd:option('-learning_rate_decay_after',20,'in number of epochs, when to start decaying the learning rate')

cmd:option('-max_epochs',200,'number of full passes through the training data')

cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every',100,'Save every $1 iterations')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-seed',9415,'torch manual random number generator seed')
opt = cmd:parse(arg)

torch.manualSeed(opt.seed)

-- CUDA
if opt.type == 'float' then
    print('==> switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
    print('==> switching to CUDA')
    require 'cunn'
    torch.setdefaulttensortype('torch.FloatTensor') -- Not sure why I do this
end

-- cqt_features = 175
-- timepoints = 135
cqt_features = 176
timepoints = 83
local loader = GridSpeechBatchLoader.create(cqt_features, timepoints, opt.batch_size)

nspeakers = 2
init_params = true
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    encoder = checkpoint.encoder
    decoder = checkpoint.decoder
    init_params = false
else
    encoder = CNN2.encoder(cqt_features, timepoints)
    decoder = CNN2.decoder(cqt_features, timepoints)
    debug.debug()
end

diffnet = Difference.diff()
criterion = nn.MSECriterion()


local net_params, net_grads = encoder:parameters()
itorch.image(encoder:get(1).weight)

-- CUDA
if opt.type == 'cuda' then
   encoder:cuda()
   decoder:cuda()
   diffnet:cuda()
   criterion:cuda()
end


-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(encoder, decoder)
nparams = params:nElement()
print('number of parameters in the model: ' .. nparams)

if init_params then
    params:normal(-1/torch.sqrt(nparams), 1/torch.sqrt(nparams))
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

first = true
last_grad_params = params -- Dummy
last_params = params -- Dummy

function feval(p)
    if p ~= params then
        params:copy(p)
    end
    grad_params:zero()

    local perf = false
    local timer = torch.Timer()
    sAwX, sBwX, sAwY, sBwY = unpack(loader:next_batch(train))

    if perf then print (string.format("Time 1: %.3f", timer:time().real)) end

    if opt.type == 'cuda' then
        sAwX = sAwX:float():cuda()
        sBwX = sBwX:float():cuda()
        sAwY = sAwY:float():cuda()
        sBwY = sBwY:float():cuda()
    end

    if perf then print (string.format("Time 1.5: %.3f", timer:time().real)) end

    rsAwX = encoder:forward(sAwX)
    rsBwX = encoder:forward(sBwX)
    rsAwY = encoder:forward(sAwY)
    if perf then print (string.format("Time 1.75: %.3f", timer:time().real)) end
    diff  = diffnet:forward({rsAwX, rsBwX})
    if perf then print (string.format("Time 2: %.3f", timer:time().real)) end

    sBwY_pred = decoder:forward({diff, rsAwY})
    if perf then print (string.format("Time 3: %.3f", timer:time().real)) end

    local loss = criterion:forward(sBwY, sBwY_pred)
    if perf then print (string.format("Time 4: %.3f", timer:time().real)) end

    doutput = criterion:backward(sBwY_pred, sBwY)
    if opt.save_pred then matio.save('reconstructions/s2_actual_v6.mat', {X1=sBwY:float()}) end
    if opt.save_pred then matio.save('reconstructions/s2_pred_v6.mat', {X1=sBwY_pred:float()}) end
    diff_out, rsAwY_out = unpack(decoder:backward({diff, rsAwY}, doutput))

    rsAwX_out, rsBwX_out = unpack(diffnet:backward({rsAwX, rsBwX}, diff_out))
    if perf then print (string.format("Time 4.5: %.3f", timer:time().real)) end

    sAwY_out = encoder:backward(sAwY, rsAwY_out)
    sBwX_out = encoder:backward(sBwX, rsBwX_out)
    sAwX_out = encoder:backward(sAwX, rsAwX_out) -- Check gradients add?
    if perf then print (string.format("Time 5: %.3f", timer:time().real)) end

    if not first then
        print (string.format("L:%.5f", torch.norm(grad_params - last_grad_params) / torch.norm(params - last_params)))
    end
    first = false
    last_params:copy(params)
    print ("Params", torch.norm(params))
    -- last_grad_params:copy(grad_params)
    return loss, grad_params
end


function test(p)
    if p ~= params then
        params:copy(p)
    end
    grad_params:zero()

    sAwX, sBwX, sAwY, sBwY = unpack(loader:next_test_batch())

    if opt.type == 'cuda' then
        sAwX = sAwX:float():cuda()
        sBwX = sBwX:float():cuda()
        sAwY = sAwY:float():cuda()
        sBwY = sBwY:float():cuda()
    end

    rsAwX = encoder:forward(sAwX)
    rsBwX = encoder:forward(sBwX)
    rsAwY = encoder:forward(sAwY)
    diff  = diffnet:forward({rsAwX, rsBwX})

    sBwY_pred = decoder:forward({diff, rsAwY})

    local loss = criterion:forward(sBwY, sBwY_pred)
    print(string.format("TEST LOSS - loss=%.5f", loss))
end

local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil
local optim_state = {learningRate = opt.learning_rate}

for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
    print (i)
    local _, loss = optim.sgd(feval, params, optim_state)
    local time = timer:time().real

    loss = loss[1]

    -- exponential learning rate decay
    if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if (i % opt.save_every == 0 or i == iterations) then
        local savefile = string.format('%s/net_analogy_%.2f.t7', opt.checkpoint_dir, epoch)
        print('saving checkpoint to ' .. savefile)
        checkpoint = {}
        checkpoint.encoder = encoder
        checkpoint.decoder = decoder
        torch.save(savefile, checkpoint)
        print('saved checkpoint to ' .. savefile)
    end


    -- handle early stopping if things are going really bad
    if loss ~= loss then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss end
end
