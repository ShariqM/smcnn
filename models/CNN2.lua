
-- NETWORK

local CNN2 = {}

--176x83
--[[
hpsz = 3 -- Height Pool Size
wpsz = 3 -- Width Pool Size
csz = 5 -- Conv Size
ssz = 1 -- Stride Size

nchannels = {1,8,64}
full_sizes = {-1, 2048, 2048}
view_height = 17
view_width  = 7
-- view_width  = 7

usz = 3 -- Height Upsample Size
-- usz = 2 -- Width Upsample Size
]]--

-- 175x140
hpsz = 3 -- Height Pool Size
wpsz = 3 -- Width Pool Size
csz = 5 -- Conv Size
ssz = 1 -- Stride Size

nchannels = {1,8,64}
full_sizes = {-1, 2048, 2048}
view_height = 17
view_width  = 13
-- view_width  = 7

usz = 3 -- Height Upsample Size
-- usz = 2 -- Width Upsample Size

-- 175x140
--[[
hpsz = 3 -- Height Pool Size
wpsz = 3 -- Width Pool Size
csz = 5 -- Conv Size
ssz = 1 -- Stride Size

nchannels = {1,8,64}
full_sizes = {-1, 4096, 4096}
view_height = 17
view_width  = 13
-- view_width  = 7

usz = 3 -- Height Upsample Size
-- usz = 2 -- Width Upsample Size
]]--


function CNN2.encoder(cqt_features, timepoints, dropout)
    local x = nn.Identity()()

    local curr = x
    for i=1, #nchannels - 1 do
        local conv = nn.SpatialConvolution(nchannels[i],nchannels[i+1],csz,csz,ssz,ssz)(curr)
        local relu = nn.ReLU()(conv)
        local pavg = nn.SpatialAveragePooling(hpsz,wpsz,hpsz,wpsz)(relu)
        curr = pavg
        if dropout then
            curr = nn.Dropout(dropout)(curr)
        end
    end

    full_sizes[1] = nchannels[#nchannels] * view_height * view_width
    print (full_sizes[1])
    curr = nn.View(full_sizes[1])(curr)

    for i=1, #full_sizes - 2 do
        local full = nn.Linear(full_sizes[i], full_sizes[i+1])(curr)
        local relu = nn.ReLU()(full)
        curr = relu
        if dropout then
            curr = nn.Dropout(dropout)(curr)
        end
    end

    i = #full_sizes - 1
    local out = nn.Linear(full_sizes[i], full_sizes[i+1])(curr)

    return nn.gModule({x}, {out})
end

function CNN2.decoder(cqt_features, timepoints, dropout)
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
        if dropout then
            curr = nn.Dropout(dropout)(curr)
        end
    end
    curr = nn.Linear(full_sizes[2], full_sizes[1])(curr)
    print ('full', full_sizes[1])

    curr = nn.View(nchannels[#nchannels], view_height, view_width)(curr)

    i = #nchannels-1
    curr = nn.SpatialUpSamplingNearest(usz)(curr)
    -- curr = nn.SpatialReplicationPadding(2, 0, 2, 0)(curr) (175x140)
    curr = nn.SpatialReplicationPadding(1, 0, 2, 0)(curr)
    curr = nn.SpatialFullConvolution(nchannels[i+1],nchannels[i],csz,csz,ssz,ssz)(curr)
    curr = nn.Sigmoid()(curr)

    if dropout then
        curr = nn.Dropout(dropout)(curr)
    end

    i = i - 1
    curr = nn.SpatialUpSamplingNearest(usz)(curr)
    -- curr = nn.SpatialReplicationPadding(1, 0, 0, 0)(curr) (175x140)
    curr = nn.SpatialReplicationPadding(1, 0, 1, 0)(curr)
    curr = nn.SpatialFullConvolution(nchannels[i+1],nchannels[i],csz,csz,ssz,ssz)(curr)
    out = curr

    return nn.gModule({A, B}, {out})
end

function CNN2.adv_classifier(cqt_features, timepoints, dropout)
    local x = nn.Identity()()

    local curr = x
    for i=1, #nchannels - 1 do
        local conv = nn.SpatialConvolution(nchannels[i],nchannels[i+1],csz,csz,ssz,ssz)(curr)
        local relu = nn.ReLU()(conv)
        local pavg = nn.SpatialAveragePooling(hpsz,wpsz,hpsz,wpsz)(relu)
        curr = pavg
    end

    full_sizes[1] = nchannels[#nchannels] * view_height * view_width
    print (full_sizes[1])
    view = nn.View(full_sizes[1])(curr)

    -- Speaker
    curr = nn.Linear(full_sizes[1], 2048)(view)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(2048, 256)(curr)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(256, 33)(curr)
    spk_out = nn.LogSoftMax()(curr)

    -- Word
    curr = nn.Linear(full_sizes[1], 2048)(view)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(2048, 256)(curr)
    curr = nn.ReLU()(curr)
    curr = nn.Linear(256, 31)(curr)
    word_out = nn.LogSoftMax()(curr)

    return nn.gModule({x}, {spk_out, word_out})
end

return CNN2
