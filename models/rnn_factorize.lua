local RNN = {}

function RNN.rnn(input_size, rnn_sizes, spk_size, lens, pool_size, dropout)

  nspeakers, nphonemes = unpack(lens)
  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  local n = tableLength(rnn_sizes) -- Num layers
  local pool_layer = 1 + math.floor((n-1)/2) -- Middle layer

  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
    if L == pool_layer then
        table.insert(inputs, nn.Identity()()) -- Extra input
    end
  end

  local phn_size = rnn_sizes[pool_layer] - spk_size
  rnn_sizes[0] = input_size

  local x
  local outputs = {}
  local next_hs
  local next_hp
  local softmax_s
  local softmax_p

  local hi_idx = 2 -- hidden input index
  for L = 1,n do
    if L == 1 then
      x = inputs[1]
    else
      if L-1 == pool_layer then -- if previous layer was pool then have to concat it for input
        -- x = nn.Concat(2)({next_hs, next_hp})
        x = 'garbage'
      else
        x = outputs[(L-1)]
      end
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
    end

    -- RNN tick
    if L == pool_layer then
      i2hs = nn.Linear(rnn_sizes[L-1], spk_size)(x)
      i2hp = nn.Linear(rnn_sizes[L-1], phn_size)(x)

      local prev_hs = inputs[hi_idx] -- Last input?
      local prev_hp = inputs[hi_idx+1]
      hi_idx = hi_idx + 2

      hs2hs = nn.Linear(spk_size, spk_size)(prev_hs)
      hp2hp = nn.Linear(phn_size, phn_size)(prev_hp)

      next_hs = nn.ReLU()(nn.CAddTable(){i2hs, hs2hs})
      next_hp = nn.ReLU()(nn.CAddTable(){i2hp, hp2hp})
      table.insert(outputs, next_hs)
      table.insert(outputs, next_hp)

      reshape_hs = nn.Reshape(1,1,spk_size)(next_hs)
      reshape_hp = nn.Reshape(1,1,phn_size)(next_hp)
      pool_hs    = nn.SpatialLPPooling(1, 2, pool_size, 1, 2)(reshape_hs)
      pool_hp    = nn.SpatialLPPooling(1, 2, pool_size, 1, 2)(reshape_hp)

      breshape_hs = nn.Reshape(spk_size/pool_size)(pool_hs)
      breshape_hp = nn.Reshape(phn_size/pool_size)(pool_hp)

      rep_s      = nn.Linear(spk_size/pool_size, nspeakers)(breshape_hs)
      rep_p      = nn.Linear(phn_size/pool_size, nphonemes)(breshape_hp)
      softmax_s  = nn.LogSoftMax()(rep_s)
      softmax_p  = nn.LogSoftMax()(rep_p)
    else
      local prev_h = inputs[hi_idx]
      hi_idx = hi_idx + 1

      if L == 3 then
        is2h = nn.Linear(spk_size, rnn_sizes[L])(next_hs)
        ip2h = nn.Linear(phn_size, rnn_sizes[L])(next_hp)
        i2h  = nn.CAddTable(){is2h, ip2h}
      else
        i2h = nn.Linear(rnn_sizes[L-1], rnn_sizes[L])(x)
      end

      h2h = nn.Linear(rnn_sizes[L], rnn_sizes[L])(prev_h)

      local next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h})
      table.insert(outputs, next_h)
    end
  end

    -- Broken for 1 layer
  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

  local h2o = nn.NormLinear(rnn_sizes[n], input_size)(top_h)
  table.insert(outputs, h2o)

  -- Softmax's at the end
  table.insert(outputs, softmax_s)
  table.insert(outputs, softmax_p)

  return {nn.gModule(inputs, outputs), i2h, h2h, h2o}
end

return RNN
