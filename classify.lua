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
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-iters',400,'iterations per epoch')
cmd:option('-learning_rate',8e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')

cmd:option('-batch_size', 32,'number of sequences to train on in parallel')
cmd:option('-max_epochs',200,'number of full passes through the training data')

cmd:option('-print_every',200,'how many steps/minibatches between printing out the loss')
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

plot_threshold = 188000
cqt_features = 175
timepoints = 1024
local loader = TimitBatchLoader.create(cqt_features, timepoints, opt.batch_size)

nspeakers = 38
init_params = false
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    cnn = checkpoint.model
    dummy_cnn = checkpoint.dummy_model
    init_params = false
else
    cnn       = CNN.cnn(nspeakers, false)
    dummy_cnn = CNN.cnn(nspeakers, true)
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
if init_params then
    params:normal(-1/torch.sqrt(nparams), 1/torch.sqrt(nparams))
    params:uniform(-0.08, 0.08) -- small uniform numbers
end

function heat_plot(image)
    heatmap = torch.Tensor(cqt_features, timepoints)
    cqt_size = 25
    time_size = 32
    local g = 1
    for t=1, timepoints/time_size do
        for c=1, cqt_features/cqt_size do
            c_sidx = (c-1) * cqt_size + 1
            c_eidx = c * cqt_size
            t_sidx = (t-1) * time_size + 1
            t_eidx = t * time_size
            if weights[g] == 1 then
                heatmap[{{c_sidx, c_eidx}, {t_sidx, t_eidx}}] = torch.exp(pred[{g,spk_labels[1]}])
            else
                heatmap[{{c_sidx, c_eidx}, {t_sidx, t_eidx}}] = -0.5
            end
            g = g + 1
        end
    end

    threshold = image:mean()
    for c=1, cqt_features do
        for t=1, timepoints do
            if image[{c,t}] > threshold then
                heatmap[{c,t}] = -1
            end
        end
    end

    gnuplot.figure(1)
    gnuplot.pngfigure('results/heatmap1.png')
    gnuplot.title(string.format('Heatmap of CNN for Speaker 1 (NSpeakers=%d)', opt.batch_size))
    gnuplot.xlabel('Time (t)')
    gnuplot.ylabel('CQT')

    gnuplot.imagesc(heatmap, 'color')
    gnuplot.plotflush()

    gnuplot.figure(2)
    gnuplot.pngfigure('results/image1.png')
    gnuplot.title('Original Image')
    gnuplot.xlabel('Time (t)')
    gnuplot.ylabel('CQT')
    gnuplot.imagesc(image, 'color')
    gnuplot.plotflush()
    debug.debug()
end

loader:setup_weights(dummy_cnn, opt.type == 'cuda')

x, spk_labels, weights = unpack(loader:next_spk())
diag_weights = torch.diag(weights):float()

if opt.type == 'cuda' then x = x:float():cuda() end -- Ship to GPU
if opt.type == 'cuda' then weights = weights:float():cuda() end -- Ship to GPU


local mean_sum = 0
local plot_time
function feval(p)
    if p ~= params then
        params:copy(p)
    end
    grad_params:zero()

    -- local timer = torch.Timer()
    -- x, spk_labels, weights = unpack(loader:next_spk())
    -- diag_weights = torch.diag(weights):float()

    -- if opt.type == 'cuda' then x = x:float():cuda() end -- Ship to GPU
    -- if opt.type == 'cuda' then weights = weights:float():cuda() end -- Ship to GPU
        -- This was taking 0.03 seconds for some reason (X only .007 even though it's bigger)
    -- if opt.type == 'cuda' then diag_weights = diag_weights:cuda() end -- Ship to GPU

    pred   = cnn:forward(x)
    num_per_batch = pred:size()[1] / opt.batch_size
    batch_spk_labels = torch.Tensor(pred:size()[1])
    mean_sum_batch = 0
    for b=1, opt.batch_size do
        for n=1, num_per_batch do
            batch_spk_labels[(b-1) * num_per_batch + n] = spk_labels[b]
        end

        -- Take the mean of the proabilities in the windows we care about
        sidx, eidx = (b-1)*num_per_batch + 1, b*num_per_batch
        probabilities = torch.exp(pred[{{sidx, eidx}, spk_labels[b]}])
        relevant = torch.cmul(weights[{{sidx,eidx}}], probabilities):float()

        relevant = relevant:index(1, torch.squeeze(torch.nonzero(relevant)))
        mean_sum_batch = mean_sum_batch + torch.mean(relevant)
    end
    mean_sum = mean_sum + mean_sum_batch / opt.batch_size

    if plot_time == true then
        print ('show')
        heat_plot(x[{1,1,{},{}}]:transpose(1,2))
    end

    if opt.type == 'cuda' then batch_spk_labels = batch_spk_labels:float():cuda() end -- Ship to GPU
    local loss = criterion:forward(pred, batch_spk_labels)

    doutput = criterion:backward(pred, batch_spk_labels):float()
    -- if opt.type == 'cuda' then doutput = doutput:float():cuda() end
    doutput = diag_weights * doutput
    if opt.type == 'cuda' then doutput = doutput:float():cuda() end
    cnn:backward(x, doutput)
    -- print ('Time 8: ', timer:time().real)
    -- print ('')

    return loss, grad_params
end

train_losses = {}
local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil
local optim_state = {learningRate = opt.learning_rate}
local train_loss = 0

for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
    local _, loss = optim.sgd(feval, params, optim_state) -- Works better than adagrad or rmsprop
    local time = timer:time().real

    train_loss =  train_loss + loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    if i > plot_threshold then
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
        print(string.format("%d/%d (epoch %.3f), mean=%.2f, train_loss = %.3f, grad norm = %.3f, time/batch = %.4fs", i, iterations, epoch, mean_sum / opt.print_every, train_loss, grad_params:norm(), time))
        train_loss = 0
        mean_sum = 0
    end

    if (i % opt.save_every == 0 or i == iterations) then
        local savefile = string.format('%s/cnn_epoch%.2f.t7', opt.checkpoint_dir, epoch)
        print('saving checkpoint to ' .. savefile)
        checkpoint = {}
        checkpoint.model= cnn
        checkpoint.dummy_model = dummy_cnn
        torch.save(savefile, checkpoint)
    end


    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
end
