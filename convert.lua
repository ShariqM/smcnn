require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

matio = require 'matio'
matio.use_lua_strings = true
local model_utils=require 'model_utils'
local CNN = require 'models.cnn'
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
cmd:option('-batch_size',128,'number of sequences to train on in parallel')
cmd:option('-dropout',0.1,'dropout for regularization, used after each CNN hidden layer. 0 = no dropout')
cmd:option('-log',false,'Log the probabilities of correct answers')

cmd:option('-save_pred',false,'Save prediction')
cmd:option('-run_test',false,'Run test set')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-dont_save', false, 'Stop checkpointing')
cmd:option('-learning_rate',1e-5,'learning rate')
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

cqt_features = 175
timepoints = 140
local loader = GridSpeechBatchLoader.create(cqt_features, timepoints,
                                            opt.batch_size, opt.compile_test)

init_params = true
if string.len(opt.init_from) > 0 then
    print('loading a Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    encoder = checkpoint.encoder
    decoder = checkpoint.decoder
    classify = checkpoint.classify
    init_params = false
else
    if opt.compile_test then
        classify = CNN.test_adv_classifier(cqt_features, timepoints, opt.dropout)
        opt.dont_save = true
    else
        print 'Loading network'
        encoder  = CNN.encoder(cqt_features, timepoints, opt.dropout)
        decoder  = CNN.decoder(cqt_features, timepoints, opt.dropout)
        classify = CNN.adv_classifier(cqt_features, timepoints, opt.dropout)
        print 'Networks created'
    end
end

diffnet = Difference.diff()
criterion = nn.ClassNLLCriterion()

-- CUDA
if opt.type == 'cuda' then
   encoder:cuda()
   decoder:cuda()
   classify:cuda()
   diffnet:cuda()
   criterion:cuda()
end

-- put the above things into one flattened parameters tensor
gen_params, gen_grad_params = model_utils.combine_all_parameters(encoder, decoder)
nparams = gen_params:nElement()
print('number of parameters in the generative     model: ' .. nparams)

disc_params, disc_grad_params = model_utils.combine_all_parameters(classify)
nparams = disc_params:nElement()
print('number of parameters in the discriminative model: ' .. nparams)

if init_params then
    gen_params:uniform(-0.08, 0.08) -- small uniform numbers
    disc_params:uniform(-0.08, 0.08) -- small uniform numbers
end

function gen_feval(p)
    if p ~= gen_params then
        gen_params:copy(p)
    end
    gen_grad_params:zero()

    local perf = false
    local timer = torch.Timer()
    sAwX, sBwX, sAwY, sBwY, spk_labels, word_labels = unpack(loader:next_batch(train))

    if perf then print (string.format("Time 1: %.3f", timer:time().real)) end

    if opt.type == 'cuda' then
        sAwX = sAwX:float():cuda()
        sBwX = sBwX:float():cuda()
        sAwY = sAwY:float():cuda()
        sBwY = sBwY:float():cuda()

        spk_labels  =  spk_labels:float():cuda()
        word_labels = word_labels:float():cuda()
    end

    if perf then print (string.format("Time 1.5: %.3f", timer:time().real)) end

    -- Forward
    rsAwX = encoder:forward(sAwX)
    rsBwX = encoder:forward(sBwX)
    rsAwY = encoder:forward(sAwY)
    if perf then print (string.format("Time 1.75: %.3f", timer:time().real)) end
    diff  = diffnet:forward({rsAwX, rsBwX})
    if perf then print (string.format("Time 2: %.3f", timer:time().real)) end

    local sBwY_pred = decoder:forward({diff, rsAwY})
    -- print ('Decode out:', sBwY_pred:size())
    -- debug.debug()
    if perf then print (string.format("Time 3: %.3f", timer:time().real)) end

    spk_pred, word_pred = unpack(classify:forward(sBwY_pred))
    if opt.log then
        print "GEN - Generative Distribution"
        for i=opt.batch_size-8,opt.batch_size do
            sprob = torch.exp(spk_pred)[{i,spk_labels[i]}]
            wprob = torch.exp(word_pred)[{i,word_labels[i]}]
            print (string.format("\tP(S=%d)=%.2f || P(W)=%.2f", spk_labels[i], sprob, wprob))
        end
    end

    local loss = criterion:forward(spk_pred, spk_labels)
    loss = loss + criterion:forward(word_pred, word_labels)
    if perf then print (string.format("Time 4: %.3f", timer:time().real)) end

    if opt.save_pred then
        matio.save('reconstructions/train_actual.mat', {X1=sBwY:float()})
        matio.save('reconstructions/train_pred.mat', {X1=sBwY_pred:float()})
        if not opt.run_test then os.exit() end
    end

    -- Backward
    doutput_spk  = criterion:backward(spk_pred,  spk_labels):float()
    doutput_word = criterion:backward(word_pred, word_labels):float()
    if opt.type == 'cuda' then doutput_spk  = doutput_spk:cuda() end
    if opt.type == 'cuda' then doutput_word = doutput_word:cuda() end

    doutput = classify:backward(x, {doutput_spk, doutput_word})
    if opt.type == 'cuda' then doutput = doutput:cuda() end
    diff_out, rsAwY_out = unpack(decoder:backward({diff, rsAwY}, doutput))

    rsAwX_out, rsBwX_out = unpack(diffnet:backward({rsAwX, rsBwX}, diff_out))
    if perf then print (string.format("Time 4.5: %.3f", timer:time().real)) end

    sAwY_out = encoder:backward(sAwY, rsAwY_out)
    sBwX_out = encoder:backward(sBwX, rsBwX_out)
    sAwX_out = encoder:backward(sAwX, rsAwX_out) -- Check gradients add?
    if perf then print (string.format("Time 5: %.3f", timer:time().real)) end

    return loss, gen_grad_params
end

function disc_feval(p)
    if p ~= disc_params then
        disc_params:copy(p)
    end
    disc_grad_params:zero()

    local perf = false
    local timer = torch.Timer()
    x, spk_labels, word_labels = unpack(
                            loader:next_adv_class_batch(opt.type == 'cuda'))

    if perf then print (string.format("Time 1: %.3f", timer:time().real)) end

    if opt.type == 'cuda' then
        x = x:float():cuda()
        spk_labels  =  spk_labels:float():cuda()
        word_labels = word_labels:float():cuda()
    end

    if perf then print (string.format("Time 2: %.3f", timer:time().real)) end

    spk_pred, word_pred = unpack(classify:forward(x))
    if opt.log then
        print "DISC - True Distribution"
        for i=1,4 do
            sprob = torch.exp(spk_pred)[{i,spk_labels[i]}]
            wprob = torch.exp(word_pred)[{i,word_labels[i]}]
            print (string.format("\tP(S=%d)=%.2f || P(W)=%.2f", spk_labels[i], sprob, wprob))
        end
        print "DISC - Generative Distribution"
        for i=opt.batch_size-4,opt.batch_size do
            sprob = torch.exp(spk_pred)[{i,spk_labels[i]}]
            wprob = torch.exp(word_pred)[{i,word_labels[i]}]
            print (string.format("\tP(S=%d)=%.2f || P(W)=%.2f", spk_labels[i], sprob, wprob))
        end
    end

    -- debug.debug()
    if perf then print (string.format("Time 3: %.3f", timer:time().real)) end
    local loss = criterion:forward(spk_pred, spk_labels)
    loss = loss + criterion:forward(word_pred, word_labels)
    if perf then print (string.format("Time 4: %.3f", timer:time().real)) end

    doutput_spk  = criterion:backward(spk_pred,  spk_labels):float()
    doutput_word = criterion:backward(word_pred, word_labels):float()
    if opt.type == 'cuda' then doutput_spk  = doutput_spk:cuda() end
    if opt.type == 'cuda' then doutput_word = doutput_word:cuda() end

    classify:backward(x, {doutput_spk, doutput_word})

    return loss, disc_grad_params
end

local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
-- local optim_state = {learningRate = opt.learning_rate, momentum=0.5}
local optim_state = {learningRate = opt.learning_rate}

opt_disc = true
opt_gen  = true
local disc_loss = -1.0
local gen_loss  = -1.0
local threshold_1 = 0.85
local threshold_2 = 0.95

for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
    if opt_gen then
        _, gen_loss = optim.sgd(gen_feval, gen_params, optim_state)
        gen_loss = gen_loss[1]
    end

    if opt_disc then
        _, disc_loss = optim.sgd(disc_feval, disc_params, optim_state)
        disc_loss = disc_loss[1]
    end

    local time = timer:time().real

    if i == 1 or i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), disc_loss = %6.8f, gen_loss=%6.8f, time/batch = %.4fs", i, iterations, epoch, disc_loss, gen_loss, time))
    end

    if not opt.dont_save and (i % opt.save_every == 0 or i == iterations) then
        local savefile = string.format('%s/net_analogy_%.2f.t7', opt.checkpoint_dir, epoch)
        print('saving checkpoint to ' .. savefile)
        checkpoint = {}
        checkpoint.encoder = encoder
        checkpoint.decoder = decoder
        checkpoint.classify = classify
        torch.save(savefile, checkpoint)
        print('saved checkpoint to ' .. savefile)
    end

    if disc_loss < threshold_1 * gen_loss then
        opt_disc = false
        opt_gen = true
    end

    if disc_loss > threshold_2 * gen_loss then
        opt_disc = true
        opt_gen = false
    end

    -- handle early stopping if things are going really bad
    if gen_loss ~= gen_loss or disc_loss ~= disc_loss then
        print('loss is NaN.  This usually indicates a bug.')
        break -- halt
    end
end
