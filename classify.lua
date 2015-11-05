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
cmd:option('-learning_rate',4e-2,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')

cmd:option('-batch_size',1,'number of sequences to train on in parallel')
cmd:option('-max_epochs',100,'number of full passes through the training data')

cmd:option('-print_every',2,'how many steps/minibatches between printing out the loss')
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
total_tlength = 1024
local loader = TimitBatchLoader.create(cqt_features, total_tlength, opt.batch_size)

nspeakers = 38
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    cnn = checkpoint.model
else
    cnn = unpack(CNN.cnn_simple(nspeakers, opt.batch_size))
    -- dummy_cnn, batch_size = unpack(CNN.cnn_simple(nspeakers, opt.batch_size))
    -- cnn, batch_size = unpack(CNN.cnn_localmin(nspeakers))
    -- cnn, batch_size = unpack(CNN.cnn(nspeakers, false))
    -- dummy_cnn, batch_size = unpack(CNN.cnn(nspeakers, true))
    -- cnn, batch_size = unpack(CNN.cnn_original(nspeakers, false))
    -- dummy_cnn, batch_size = unpack(CNN.cnn_original(nspeakers, true))
    -- cnn, batch_size = unpack(CNN.cnn_original(nspeakers))
    -- cnn, batch_size = unpack(CNN.cnn_localmin(nspeakers))
    -- cnn, batch_size = unpack(CNN.cnn_localmin(nspeakers))
end
criterion = nn.ClassNLLCriterion()

-- CUDA
if opt.type == 'cuda' then
   cnn:cuda()
   -- dummy_cnn:cuda()
   criterion:cuda()
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(cnn)
dummy_params, garbage = model_utils.combine_all_parameters(dummy_cnn)
dummy_params:fill(1e-1)
nparams = params:nElement()
print('number of parameters in the model: ' .. nparams)
-- params:normal(-1/torch.sqrt(nparams), 1/torch.sqrt(nparams))
params:uniform(-0.08, 0.08) -- small uniform numbers

function pool_plot()
    -- print ('try plot')
    dist = torch.Tensor(opt.seq_length)
    for i=1,opt.seq_length do
        dist[i] = (pool_state[i] - pool_state[1]):norm()
    end
    gnuplot.title('Norm of Pool state X_t vs Pool State X_1')
    gnuplot.xlabel('Time (t)')
    gnuplot.ylabel('Norm')
    -- gnuplot.axis({1, opt.seq_length, 1, opt.rnn_size * 5})
    gnuplot.axis({1, opt.seq_length, 1, 1000})
    gnuplot.plot(dist)
    -- print ('plotted')
end

local pred
local mean_sum = 0
local plot_time
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    local timer = torch.Timer()
    x, spk_labels = unpack(loader:next_spk())
    if opt.type == 'cuda' then x = x:float():cuda() end -- Ship to GPU

    pred   = cnn:forward(x)
    -- print ('Time 1: ', timer:time().real)
    num_per_batch = pred:size()[1] / opt.batch_size
    batch_spk_labels = torch.Tensor(pred:size()[1])
    mean_sum_batch = 0
    for b=1, opt.batch_size do
        for n=1, num_per_batch do
            batch_spk_labels[(b-1) * num_per_batch + n] = spk_labels[b]
        end
        mean_sum_batch = mean_sum_batch + torch.mean(torch.exp(pred[{{1,b*num_per_batch},spk_labels[b]}]))
    end
    mean_sum = mean_sum + mean_sum_batch / opt.batch_size

    if plot_time == true then
        gnuplot.imagesc(x[{1,1,{},{}}]:transpose(1,2), 'color')
        heatmap = torch.Tensor(cqt_features, total_tlength )
        cqt_size = 25
        time_size = 16
        local g = 1
        for t=1, total_tlength/time_size do
            for c=1, cqt_features/cqt_size do
                c_sidx = (c-1) * cqt_size + 1
                c_eidx = c * cqt_size
                t_sidx = (t-1) * time_size + 1
                t_eidx = t * time_size
                heatmap[{{c_sidx, c_eidx}, {t_sidx, t_eidx}}] = torch.exp(pred[{g,spk_labels[1]}])
                g = g + 1
                -- heatmap[{{c_sidx, c_eidx}, {t_sidx, t_eidx}}] = torch.uniform(0,1)
            end
        end
        -- q = torch.Tensor(2,2)
        -- q[{1,1}] = 1
        -- q[{1,2}] = 2
        -- q[{2,1}] = 3
        -- q[{2,2}] = 4
        -- gnuplot.imagesc(q)
        -- debug.debug()
        image = x[{1,1,{},{}}]:transpose(1,2)

        threshold = image:mean() + (image:std()/2)
        for c=1, cqt_features do
            for t=1, total_tlength do
                if image[{c,t}] > threshold then
                    heatmap[{c,t}] = -1
                end
            end
        end

        gnuplot.figure(1)
        gnuplot.imagesc(heatmap, 'color')
        debug.debug()

        -- gnuplot.figure(2)
        -- gnuplot.imagesc(x[{1,1,{},{}}]:transpose(1,2), 'color')
        -- print ('Time 2: ', timer:time().real)
    end

    if opt.type == 'cuda' then batch_spk_labels = batch_spk_labels:float():cuda() end -- Ship to GPU
    local loss = criterion:forward(pred, batch_spk_labels)
    -- print ('Time 3: ', timer:time().real)

    -- energy = dummy_cnn:forward(x)
    -- print ('Time 2: ', timer:time().real)
    -- weights = energy / energy:max()
    -- pred[{{1,42163},36}]:fill(0) -- No loss
    -- pred[{{1,42163},36}]:fill(0) -- No loss

    -- threshold = 0.8
    -- correct_vector = torch.Tensor(nspeakers):fill(-100)
    -- correct_vector[spk_class] = 0
    -- c = 0
    -- tot_prob = 0
    -- print (spk_class)

    -- for b=1, batch_size do
        -- if weights[b] < threshold then
            -- pred[b] = correct_vector
        -- else
            -- c = c + 1
            -- tot_prob = tot_prob + torch.exp(pred[{b,spk_class}])
        -- end
    -- end
    -- print ('Time 3: ', timer:time().real)
    -- print ('mean prob for thresholding: ', tot_prob / c)


    local doutput = criterion:backward(pred, batch_spk_labels)
    -- print (doutput:size())
    cnn:backward(x, doutput)
    -- print (grad_params:norm())
    -- norm_val = grad_params:norm()
    -- x = grad_params:clone()
    -- x = x/norm_val
    -- for i=1,params:nElement() do
        -- grad_params[i] = grad_params[i]/norm_val
    -- end
    -- print (torch.dist(x, grad_params))
    -- print ('Time 4: ', timer:time().real)
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

    if train_loss < 0.3 then
        plot_time = true
    end

    -- exponential learning rate decay
    if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- if true or i == 1 or i % opt.print_every == 0 then
    if i % opt.print_every == 0 then
        -- print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
        print(string.format("%d/%d (epoch %.3f), mean=%.2f, train_loss = %6.8f, grad norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, mean_sum / opt.print_every, train_loss, grad_params:norm(), time))
        train_loss = 0
        mean_sum = 0
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
