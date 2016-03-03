
-- NETWORK

local CNN2 = {}

psz = 2
nchannels = {1,16,64}
full_sizes = {nchannels[#nchannels - 1] * 10, 16384, 4096}

function CNN2.encoder(timepoints)
    local x = nn.Identity()()


    local curr = x
    for i=1, #nchannels - 1 do
        local conv = nn.SpatialConvolution(nchannels[i],nchannels[i+1],4,4)(curr)
        local relu = nn.ReLU()(conv)
        local pavg = nn.SpatialAveragePooling(psz,psz,psz,psz)(relu)
        curr = pavg

    curr = nn.View(full_sizes[1])(curr)

    for i=1, #full_sizes - 2 do
        local full = nn.Linear(full_sizes[i], full_sizes[i+1])(curr)
        local relu = nn.ReLU()(full)
        curr = relu

    curr = nn.Linear(full_sizes[i], full_sizes[i+1])(curr)

    return nn.gModule({x}, {curr})
end

function CNN2.decoder(timepoints)
    local Afull2 = nn.Identity()()
    local Bfull2 = nn.Identity()()

    -- Deep Combiner
    -- local Afull3 = nn.Linear(full_sizes[3], full_sizes[3])(Afull2)
    -- local Arelu = nn.ReLU()(Afull3)

    -- local Bfull3 = nn.Linear(full_sizes[3], full_sizes[3])(Bfull2)
    -- local Brelu = nn.ReLU()(Bfull3)

    -- local full2 = nn.CAddTable()({Arelu, Brelu})
    local full2 = nn.CAddTable()({Afull2, Bfull2})

    return nn.gModule({Afull2, Bfull2}, {full2})

    -- Decode it
    -- local relu4 = nn.Linear(full_sizes[3], full_sizes[2])(full2)
    -- local full1 = nn.ReLU()(relu4)
--
    -- local view = nn.Linear(full_sizes[2], full_sizes[1])(full1)
--
    -- local pavg3 = nn.View(nchannels[4], 40)(view)
--
    -- local relu3 = nn.SpatialUpSamplingNearest(psz)(pavg3)
    -- local conv3 = nn.ReLU()(relu3)
    -- local pavg2 = nn.SpatialConvolution(nchannels[4], nchannels[3],4,4)(conv3)
--
    -- local relu2 = nn.SpatialUpSamplingNearest(psz)(pavg2)
    -- local conv2 = nn.ReLU()(pavg2)
    -- local pavg1 = nn.SpatialConvolution(nchannels[3], nchannels[2],4,4)(conv2)
--
    -- local relu1 = nn.SpatialUpSamplingNearest(psz)(pavg1)
    -- local conv1 = nn.ReLU()(relu1)
    -- local x     = nn.SpatialConvolution(nchannels[2], nchannels[1],4,4)(conv1)
--
    -- return nn.gModule({Afull2, Bfull2}, {x})
end

return CNN2
