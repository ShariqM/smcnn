-- adapted from: wojciechz/learning_to_execute on github

local LSTM = {}

-- Creates one timestep of one LSTM
function LSTM.lstm_window(rnn_size, char_vec_size)
    -- LSTM CELL CODE
    local x      = nn.Identity()()
    local prev_c = nn.Identity()()
    local prev_h = nn.Identity()()
    local prev_w = nn.Identity()() -- Window input

    function new_input_sum()
        -- transforms input
        local i2h            = nn.Linear(rnn_size, rnn_size)(x)
        -- transforms previous timestep's output
        local h2h            = nn.Linear(rnn_size, rnn_size)(prev_h)
        -- transforms previous timesteps window function
        local w2h            = nn.Linear(char_vec_size, rnn_size)(prev_w)
        return nn.CAddTable()({i2h, h2h, w2h})
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

    -- WINDOW FUNCTION CODE
    local vec_of_ones = nn.Identity()() -- Hacky, input a constant vector of 1's to
    local vec_inc_u   = nn.Identity()() -- Hacky, descending from U to 1 elements
    local char_vecs   = nn.Identity()() -- NumUnique by NumChars(U) (36x80?)
    local prev_k      = nn.Identity()()
    h2ah = nn.Linear(rnn_size, 1)(prev_h)
    h2bh = nn.Linear(rnn_size, 1)(prev_h)
    h2kh = nn.Linear(rnn_size, 1)(prev_h)
    a    = nn.Exp()(h2ah)
    b    = nn.Exp()(h2bh)
    -- k    = nn.Add()({nn.Exp()(h2kh), prev_k}) -- Shift in space
    k       = nn.CAddTable()({nn.Exp()(h2kh), prev_k}) -- Does this work for scalar

    avec = nn.DotProduct()({vec_of_ones, a})  -- Ux1 = UX1 * 1x1
    bvec = nn.DotProduct()({vec_of_ones, b})
    kvec = nn.DotProduct()({vec_of_ones, k})

    tmp     = nn.CSubTable()({k, vec_inc_u})
    Phi_sq  = nn.Square()(nn.CSubTable()({kvec, vec_inc_u}))
    Phi_exp = nn.Exp()(nn.CMulTable()({bvec, Phi_sq})) -- XXX Didn't do neg, neg weights? XXX
    Phi     = nn.CMulTable()({avec, Phi_exp})
    next_w  = nn.DotProduct()({char_vecs, Phi}) -- AxU * Ux1 = Ax1

    local next_k = k
    comp = nn.gModule({x, prev_c, prev_h, prev_w, prev_k, char_vecs,
                      vec_of_ones, vec_inc_u},
                      {next_c, next_h, next_w, next_k})
    -- graph.dot(comp.fg, 'LSTM', 'lstm.png') -- Really complicated...
    return comp
end

return LSTM
