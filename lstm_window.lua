-- adapted from: wojciechz/learning_to_execute on github

local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm_window(rnn_size)
    local x      = nn.Identity()()
    local CVEC   = nn.Identity()() -- NumUnique by NumChars(U) (36x80?)
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()
    local prev_k = nn.Identity()()

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(rnn_size, rnn_size)(x)
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

    -- CODE FOR WINDOW
    h2ah = nn.Linear(rnn_size, 1)(prev_h)
    h2bh = nn.Linear(rnn_size, 1)(prev_h)
    h2kh = nn.Linear(rnn_size, 1)(prev_h)
    a    = nn.Exp(h2ah)
    b    = nn.Exp(h2bh)
    k    = nn.Exp(h2kh) + prev_k -- Shift in space

    -- Does negative work?
    phi  = nn.Mul(a, nn.Exp(nn.Mul(-b, nn.Square(k - u)))) -- FIXME Need vector but have scalar...?
    w    = nn.DotProductc(CVEC, Phi)

    tmp = nn.gModule({x, prev_c, prev_h, prev_w, prev_k}, {next_c, next_h, next_w, next_k})
    -- graph.dot(tmp.fg, 'LSTM', 'lstm.png') -- Really complicated...
    return tmp
end

return LSTM
