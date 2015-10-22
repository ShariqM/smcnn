local RNN = {}

function RNN.rnn(input_size, rnn_size, n, dropout)

  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  local pool
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
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})
    if L == math.floor((n-1)/2) then
        -- reshape_h = nn.Reshape(1,5,rnn_size/5)(next_h)
        -- pool = reshape_h
        -- pool = nn.SpatialLPPooling(1, 2, rnn_size/5, 5)(reshape_h)
        reshape_h = nn.Reshape(1,1,rnn_size)(next_h)
        pool = nn.SpatialLPPooling(1, 2, rnn_size, 1)(reshape_h)
    end

    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local x_hat = nn.Linear(rnn_size, input_size)(top_h)
  table.insert(outputs, x_hat)

  table.insert(outputs, pool)

  return nn.gModule(inputs, outputs)
end

return RNN

