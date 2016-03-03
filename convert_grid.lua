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
-- local GridSpeechBatchLoader = require 'GridSpeechBatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a speech conversion model')
cmd:text()
cmd:text('Options')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-iters',400,'iterations per epoch')
cmd:option('-learning_rate',2e-2,'learning rate')
cmd:option('-learning_rate_decay',0.98,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')

cmd:option('-max_epochs',200,'number of full passes through the training data')
cmd:option('-batch_size', 8,'number of sequences to train on in parallel')
cmd:option('-dropout',0,'dropout for regularization, used after each CNN hidden layer. 0 = no dropout')

cmd:option('-print_every',200,'how many steps/minibatches between printing out the loss')
cmd:option('-test_every',1000,'Run against the test set every $1 iterations')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-learning_rate',3e-4,'learning rate')
cmd:option('-learning_rate_decay',0.98,'learning rate decay')
cmd:option('-learning_rate_decay_after',20,'in number of epochs, when to start decaying the learning rate')

cmd:option('-max_epochs',200,'number of full passes through the training data')

cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every',200,'Save every $1 iterations')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
opt = cmd:parse(arg)

-- CUDA
if opt.type == 'float' then
    print('==> switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
    print('==> switching to CUDA')
    require 'cunn'
    torch.setdefaulttensortype('torch.FloatTensor') -- Not sure why I do this
end

cqt_features = 175
timepoints = 135
--local loader = GridSpeechBatchLoader.create(cqt_features, timepoints, opt.batch_size)

nspeakers = 2
init_params = false
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    encoder = checkpoint.encoder
    decoder = checkpoint.decoder
    init_params = false
else
    encoder = CNN2.encoder(timepoints)
    decoder = CNN2.decoder(timepoints)
end
diffnet = Difference.diff()
print ('D')

criterion = nn.MSECriterion()

-- CUDA
if opt.type == 'cuda' then
   encoder:cuda()
   decoder:cuda()
   criterion:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(encoder)
nparams = params:nElement()
print('number of parameters in the model: ' .. nparams)

if init_params then
    params:normal(-1/torch.sqrt(nparams), 1/torch.sqrt(nparams))
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

function feval(p)
    if p ~= params then
        params:copy(p)
    end
    grad_params:zero()

    local timer = torch.Timer()
    sApX, sBpX, sApY, sBpY = unpack(loader:next_grid_batch(train))

    -- print (string.format("Time 1: %.3f", timer:time().real))

    if opt.type == 'cuda' then
        sApX = sApX:float():cuda()
        sBpX = sBpX:float():cuda()
        sApY = sApY:float():cuda()
        sBpY = sBpY:float():cuda()
    end

    rsApX = encoder:forward(sApX)
    rsBpX = encoder:forward(sBpX)
    diff  = diffnet:forward(sApX, sBpX)
    rsApY = encoder:forward(sBpY)

    sBpY_pred = decoder:forward(diff, rsApY)

    local loss = criterion:forward(sBpY, sBpY_pred)

    doutput = criterion:backward(sBpY, sBpY_pred)
    diff_out, rsApY_out = decoder:backward(doutput)
    rsApX_out, rsBpX_out = diffnet:backward(diff_out)
    sApY_out = encoder:backward(rsApY_out)
    sApX_out = encoder:backward(rsApX_out) -- Check gradients add?
    sBpX_out = encoder:backward(rsBpX_out)

    return loss, grad_params
end

local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil
local optim_state = {learningRate = opt.learning_rate}

for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
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

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), loss=%.5f, grad norm = %.3f, time/batch = %.4fs", i, iterations, epoch, loss , grad_params:norm(), time))
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
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
end
