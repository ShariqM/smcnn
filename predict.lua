require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
local LSTM = require 'lstm'
matio = require 'matio'
matio.use_lua_strings = true
local model_utils=require 'model_utils'

-- SETUP
local cmd = torch.CmdLine()
cmd:option('-type',       'double', 'type: double | float | cuda')
cmd:option('-load',       'false',  'load pre-trained neural network')
cmd:option('-rnn_size',    256,     'size of LSTM internal state')
cmd:option('-seq_length',  16,      'number of timesteps to unroll to')
cmd:option('-net',        'dnn',    'type: dnn | cnn')
local opt = cmd:parse(arg or {})

-- Load the Training and Test Set
dofile('build_data.lua')

-- CUDA
if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   trainset.data = trainset.data:cuda()
   testset.data = testset.data:cuda()
end

-- TODO Load nn

-- Network
lstm = LSTM.lstm(opt.rnn_size)
criterion =  nn.MSECriterion()
params, grads = model_utils.combine_all_parameters(lstm) -- Not sure what this does...
params.uniform(-0.08, 0.08) -- Hmmm...

local clone = model_utils.clone_T_times(lstm, opt.seq_length)
