require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'helpers'

local model_utils=require 'model_utils'
local CNN = require 'models.cnn'
local TimitBatchLoader = require 'TimitBatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a speech classificaiton model')
cmd:text()
cmd:text('Options')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-iters',40,'iterations per epoch')
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-max_epochs',100,'number of full passes through the training data')

cmd:option('-print_every',20,'how many steps/minibatches between printing out the loss')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
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
local loader = TimitBatchLoader.create(cqt_features)

nspeakers = 38
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    cnn = checkpoint.model
    batch_size = checkpoint.batch_size
else
    cnn, batch_size = unpack(CNN.cnn_localmin(nspeakers))
end
criterion = nn.ClassNLLCriterion()

-- CUDA
if opt.type == 'cuda' then
   cnn:cuda()
   criterion:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(cnn)
-- params:uniform(-0.08, 0.08) -- small uniform numbers
print('number of parameters in the model: ' .. params:nElement())

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    x, spk_class = unpack(loader:next_spk())
    labels = torch.Tensor(batch_size):fill(spk_class)

    if opt.type == 'cuda' then x = x:float():cuda() end -- Ship to GPU
    if opt.type == 'cuda' then labels = labels:float():cuda() end -- Ship to GPU

    pred = cnn:forward(x)
    -- pred[{{1,42163},36}]:fill(0) -- No loss
    -- pred[{{1,42163},36}]:fill(0) -- No loss
    -- print ('Mean 36', torch.mean(torch.exp(pred[{{},36}]))) -- Should be approaching 1
    local loss = criterion:forward(pred, labels)
    local doutput = criterion:backward(pred, labels)
    cnn:backward(x, doutput)
    -- print (grad_params:norm())
    -- norm_val = grad_params:norm()/10
    -- for i=1,params:nElement() do
        -- grad_params[i] = grad_params[i]/norm_val
    -- end
    -- print (grad_params:norm())
    -- grad_params = grad_params * 1000

    return loss, grad_params
end

train_losses = {}
local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local train_loss = 0

for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
    local _, loss = optim.sgd(feval, params, optim_state) -- Works better than adagrad or rmsprop
    local time = timer:time().real

    train_loss =  train_loss + loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
        train_loss = 0
    end

    if (i % opt.save_every == 0 or i == iterations) then
        local savefile = string.format('%s/cnn_epoch%.2f.t7', opt.checkpoint_dir, epoch)
        print('saving checkpoint to ' .. savefile)
        checkpoint = {}
        checkpoint.model= cnn
        checkpoint.batch_size = batch_size
        torch.save(savefile, checkpoint)
    end


    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
end
