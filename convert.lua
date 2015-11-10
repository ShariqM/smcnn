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
local CNN = require 'models.cnn'
local SpeechBatchLoader = require 'SpeechBatchLoader'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a speech classificaiton model')
cmd:text()
cmd:text('Options')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:option('-iters',400,'iterations per epoch')
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

plot_threshold = 18800993289148120
cqt_features = 175
timepoints = 1024
local loader = SpeechBatchLoader.create(cqt_features, timepoints, opt.batch_size)

nspeakers = 38
assert (string.len(opt.init_from) > 0)
print('loading an Network from checkpoint ' .. opt.init_from)
local checkpoint = torch.load(opt.init_from)
cnn = checkpoint.model
dummy_cnn = checkpoint.dummy_model
init_params = false

criterion = nn.ClassNLLCriterion()
criterion_mse = nn.MSECriterion()

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

loader:setup_grid_weights(dummy_cnn, opt.type == 'cuda')

local mean_sum = 0
function feval_transform(x)
    pred   = cnn:forward(x)

    num_per_batch = pred:size()[1]
    batch_spk_labels = torch.Tensor(pred:size()[1]):fill(tgt)
    mean_sum_batch = 0

    -- Take the mean of the proabilities in the windows we care about
    sidx, eidx = 1, num_per_batch
    probabilities = torch.exp(pred[{{sidx, eidx}, tgt}])
    relevant = torch.cmul(weights[{{sidx,eidx}}], probabilities):float()

    relevant = relevant:index(1, torch.squeeze(torch.nonzero(relevant)))
    mean_sum = mean_sum + torch.mean(relevant)

    if opt.type == 'cuda' then batch_spk_labels = batch_spk_labels:float():cuda() end -- Ship to GPU
    local loss = criterion:forward(pred, batch_spk_labels)

    doutput = criterion:backward(pred, batch_spk_labels):float()
    if opt.type == 'cuda' then doutput = doutput:float():cuda() end
    doutput = torch.cmul(block_weights, doutput)
    dinput = cnn:backward(x, doutput)
    return loss, dinput
end

train_losses = {}
local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil
local optim_state = {learningRate = opt.learning_rate}
local train_loss = 0

local dInput = torch.Tensor(1, 1, timepoints, cqt_features)
if opt.type == 'cuda' then dInput = dInput:float():cuda() end

src = 2
tgt = 1
sz = 50
x, spk_labels, weights , idx= unpack(loader:get_grid_src(true, src, tgt))
x_orig = x:clone()
block_weights = torch.expand(torch.reshape(weights, weights:size()[1], 1), weights:size()[1], nspeakers)
if opt.type == 'cuda' then x = x:float():cuda() end -- Ship to GPU
if opt.type == 'cuda' then weights = weights:float():cuda() end -- Ship to GPU
if opt.type == 'cuda' then block_weights = block_weights:cuda() end -- Ship to GPU


for i=1, 100 do
    _, loss = optim.sgd(feval_transform, x, optim_state)
    if i == 1 then
        print (string.format("%d) Mean Error %.3f Loss: %.3f", i, 1 - mean_sum, loss[1]))
    end
    if i % opt.print_every == 0 then
        spk = matio.save(string.format('converted/s%d_%d.mat', src, idx), x_orig:float())
        spk = matio.save(string.format('converted/s%d_%d_to_s%d_%d_i=%d.mat', src, idx, tgt, idx, i), x:float())
        print (x:mean(), params:mean())
        print (string.format("%d) Mean Error %.3f Loss: %.3f", i, 1 - (mean_sum/opt.print_every), loss[1]))
        mean_sum = 0
    end
end
