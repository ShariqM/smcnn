local RNN = {}

function RNN.rnn(input_size, rnn_size, pool_size, n, dropout)

  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  local pool
  local identity_hack = false
  for L = 1,n do

    local prev_h = inputs[L+1]
    if L == 1 then
      x = inputs[1]
      input_size_L = input_size
    else
      x = outputs[(L-1)]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    i2h = nn.Linear(input_size_L, rnn_size)(x)
    h2h = nn.Linear(rnn_size, rnn_size)(prev_h)

    local next_h
    if identity_hack then
        local params, grads = i2h.data.module:parameters()
        params[1]:set(torch.eye(rnn_size))
        params[2]:set(torch.zeros(rnn_size))

        params, grads = h2h.data.module:parameters()
        params[1]:set(torch.zeros(rnn_size, rnn_size))
        params[2]:set(torch.zeros(rnn_size))
        next_h = nn.CAddTable(){i2h, h2h}
    else
        -- local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})
        next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h})
    end

    -- if L == math.floor((n-1)/2) then -- Middle Layer
    if L == 1 then
      reshape_h = nn.Reshape(1,1,rnn_size)(next_h)
      pool      = nn.SpatialLPPooling(1, 2, pool_size, 1, 2)(reshape_h)
    end

    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  local h2o = nn.NormLinear(rnn_size, input_size)(top_h)
  table.insert(outputs, h2o)

  if identity_hack then
      local params, grads = h2o.data.module:parameters()
      params[1]:set(torch.eye(rnn_size))
      params[2]:set(torch.zeros(rnn_size))
  end

  table.insert(outputs, pool) -- Last ouput

  return {nn.gModule(inputs, outputs), i2h, h2h, h2o}
end

return RNN
