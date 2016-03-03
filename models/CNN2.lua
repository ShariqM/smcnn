
-- NETWORK

local CNN2 = {}

psz = 2 -- Pool Size
csz = 4 -- Conv Size
nchannels = {1,16}
dim = 10
full_sizes = {-1, 2048, 1024}

function CNN2.encoder(cqt_features, timepoints)
    local x = nn.Identity()()

    local cqt_size  = cqt_features
    local time_size = timepoints

    local curr = x
    for i=1, #nchannels - 1 do
        local conv = nn.SpatialConvolution(nchannels[i],nchannels[i+1],csz,csz)(curr)
        local relu = nn.ReLU()(conv)
        local pavg = nn.SpatialAveragePooling(psz,psz,psz,psz)(relu)
        curr = pavg

        cqt_size  = cqt_size - (csz - 1)
        cqt_size  = cqt_size / psz
        time_size = time_size - (csz - 1)
        time_size = time_size / psz
    end

    full_sizes[1] = nchannels[#nchannels] * cqt_size * time_size

    curr = nn.View(full_sizes[1])(curr)

    for i=1, #full_sizes - 2 do
        local full = nn.Linear(full_sizes[i], full_sizes[i+1])(curr)
        local relu = nn.ReLU()(full)
        curr = relu
    end

    i = #full_sizes - 1
    local out = nn.Linear(full_sizes[i], full_sizes[i+1])(curr)

    return nn.gModule({x}, {out})
end

function CNN2.decoder(cqt_features, timepoints)
    local A = nn.Identity()()
    local B = nn.Identity()()

    -- Deep Combiner
    i = #full_sizes
    local Afull = nn.Linear(full_sizes[i], full_sizes[i])(A)
    local Arelu = nn.ReLU()(Afull)

    local Bfull = nn.Linear(full_sizes[i], full_sizes[i])(B)
    local Brelu = nn.ReLU()(Bfull)

    local full = nn.CAddTable()({Arelu, Brelu})

    -- Decode it
    local curr = full
    for i=#full_sizes-1, 2, -1 do
        local full = nn.Linear(full_sizes[i+1], full_sizes[i])(curr)
        local relu = nn.ReLU()(full)
        curr = relu
    end

    i = 1
    curr = nn.Linear(full_sizes[i+1], full_sizes[i])(curr)
    curr = nn.View(nchannels[#nchannels -1], dim)(curr)

    for i=#nchannels-1, 1, -1 do
        local sus = nn.SpatialUpSamplingNearest(psz)(curr)
        local relu = nn.ReLU()(sus)
        local conv = nn.SpatialConvolution(nchannels[i+1],nchannels[i],csz,csz)(relu)
        curr = conv
    end

    local out = curr

    return nn.gModule({A, B}, {curr})
end

return CNN2
