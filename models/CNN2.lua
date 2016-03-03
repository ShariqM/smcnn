
-- NETWORK

local CNN2 = {}

psz = 2
nchannels = {1,16,64,128}
full_sizes = {nchannels[4] * 10, 16384, 4096}

function CNN2.encoder(timepoints)
    local x = nn.Identity()()
    local conv1 = nn.SpatialConvolution(nchannels[1],nchannels[2],4,4)(x)
    local relu1 = nn.ReLU()(conv1)
    local pavg1 = nn.SpatialAveragePooling(psz,psz,psz,psz)(relu1)

    local conv2 = nn.SpatialConvolution(nchannels[2],nchannels[3],4,4)(pavg1)
    local relu2 = nn.ReLU()(conv2)
    local pavg2 = nn.SpatialAveragePooling(psz,psz,psz,psz)(relu2)

    local conv3 = nn.SpatialConvolution(nchannels[3],nchannels[4],4,4)(pavg2)
    local relu3 = nn.ReLU()(conv3)
    local pavg3 = nn.SpatialAveragePooling(psz,psz,psz,psz)(relu3)


    local view = nn.View(full_sizes[1])(pavg2)

    local full1 = nn.Linear(full_sizes[1], full_sizes[2])(view)
    local relu4 = nn.ReLU()(full1)

    local full2 = nn.Linear(full_sizes[2], full_sizes[3])(relu4)

    return nn.gModule({x}, {full2})
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
