local RNN = {}

function RNN.rnn(input_size, rnn_sizes, pool_size, dropout)

  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  local n = tableLength(rnn_sizes) -- Num layers
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  rnn_sizes[0] = input_size

  local x
  local outputs = {}
  local pool
  for L = 1,n do

    local prev_h = inputs[L+1]
    if L == 1 then
      x = inputs[1]
    else
      x = outputs[(L-1)]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
    end

    -- RNN tick
    i2h = nn.Linear(rnn_sizes[L-1], rnn_sizes[L])(x)
    h2h = nn.Linear(rnn_sizes[L], rnn_sizes[L])(prev_h)

    local next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h})

    if L == 1 + math.floor((n-1)/2) then -- Middle Layer
      reshape_h = nn.Reshape(1,1,rnn_sizes[L])(next_h)
      pool      = nn.SpatialLPPooling(1, 2, pool_size, 1, 2)(reshape_h)
    end

    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  local h2o = nn.NormLinear(rnn_sizes[n], input_size)(top_h)
  table.insert(outputs, h2o)

  table.insert(outputs, pool) -- Last ouput

  return {nn.gModule(inputs, outputs), i2h, h2h, h2o}
end

return RNN
