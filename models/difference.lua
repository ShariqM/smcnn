-- Simply Compute the difference of the input tensors
local Difference = {}

function Difference.diff()
    local x = nn.Identity()()
    local y = nn.Identity()()

    local out = nn.CSubTable(){x, y}

    return nn.gModule({x,y}, {out})
end


return Difference

