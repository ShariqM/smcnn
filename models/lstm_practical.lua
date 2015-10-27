-- adapted from: wojciechz/learning_to_execute on github

local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm(cqt_features, rnn_size)
    local x = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(cqt_features, rnn_size)(x)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    comp = nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
    -- graph.dot(tmp.fg, 'LSTM', 'lstm.png') -- Really complicated...
    -- for i,node in ipairs(comp.forwardnodes) do
        -- print ('i', i)
        -- print (node.data)
        -- print ('in', node.data.input)
        -- print ('Out', node.data.nSplitOutputs)
    -- end
    -- debug.debug()

    return comp

end

return LSTM
