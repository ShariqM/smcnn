
-- NETWORK

local CNN = {}

tlength = 1024
-- tlength = 124

function get_gModule(input, last, dummy)
    local out
    if dummy == true then
        -- out = nn.Sum(2)(nn.Exp()(batched))
        -- out = nn.Exp()(batched)
        -- out = batched
        out = nn.Sum(2)(last)
        -- out = nn.Exp()(nn.Sum(2)(batched))
        -- out = nn.Sum(2)(nn.Exp()(batched))
    else
        out = nn.LogSoftMax()(last)
    end

    return nn.gModule({input}, {out})
end

function CNN.cnn1(nspeakers, dummy)
    local x = nn.Identity()()
    filt_sizes = {{25,16}}
    div = 1
    local conv1 = nn.SpatialConvolution(1, nspeakers, filt_sizes[1][1], filt_sizes[1][2],
                                        filt_sizes[1][1]/div, filt_sizes[1][2]/div)(x)
    local relu1 = nn.ReLU()(conv1)

    g = -1
    local permute = nn.Transpose({2,3},{3,4})(relu1)
    local view = nn.View(g, nspeakers)(permute)

    return get_gModule(x, view, dummy)
end

function CNN.cnn(nspeakers, dummy)
    nchannels  = {[0]=1,4,16, 64, nspeakers}
    filt_sizes = {{5,8}, {7,2}, {1, 2}, {1, 2}}
    layers = {[0] = nn.Identity()()}
    div = 1

    for i=1, #filt_sizes do
        local conv = nn.SpatialConvolution(nchannels[i-1], nchannels[i],
                        filt_sizes[i][1], filt_sizes[i][2],
                        filt_sizes[i][1]/div, filt_sizes[i][2]/div)(layers[i-1])
        layers[i] = nn.ReLU()(conv)
    end

    g = -1
    local permute = nn.Transpose({2,3},{3,4})(layers[#layers])
    local view = nn.View(g, nspeakers)(permute)

    return get_gModule(layers[0], view, dummy)
end

function CNN.cnn_many(nspeakers, dummy)
    local x = nn.Identity()()
    local conv1 = nn.SpatialConvolution(1,32,5,5)(x)
    local relu1 = nn.ReLU()(conv1)
    local pavg1 = nn.SpatialAveragePooling(2,4,2,4)(relu1)

    local conv2 = nn.SpatialConvolution(32,128,5,5)(pavg1)
    local relu2 = nn.ReLU()(conv2)
    local pavg2 = nn.SpatialAveragePooling(2,4,2,4)(relu2)

    local conv3 = nn.SpatialConvolution(128,512,3,3)(pavg2)
    local relu3 = nn.ReLU()(conv3)

    local conv4 = nn.SpatialConvolution(512,128,3,3)(relu3)
    local relu4 = nn.ReLU()(conv4)

    local conv5 = nn.SpatialConvolution(128,nspeakers,3,3)(relu4)
    local relu5 = nn.ReLU()(conv5)

    local batch_size = 56 * 34
    print ("batch_size is " .. batch_size)
    local view = nn.View(batch_size, nspeakers)(relu5)
    local logsoft = nn.LogSoftMax()(view)

    -- return {nn.gModule({x},{logsoft}), batch_size}
    return get_gModule(x, view, batch_size, dummy)
end

function CNN.cnn_original(nspeakers, dummy)
    local aps = 2 -- Average Pool Size
    -- local filt_sizes = {{3,3}, {3,3}, {3,3}} -- 2 indicates average pooling
    local filt_sizes = {{8,8}, {8,8}, {8,8}} -- 2 indicates average pooling
    local nchannels  = {8, 32, nspeakers}
    nchannels[0] = 1

    local num_layers = #filt_sizes + 1 -- For avg layer
    local avg_layer  = num_layers - 1

    local layers     = {[0] = nn.Identity()()}
    local end_width  = tlength
    local end_height = 175
    local k = 1
    for i = 1, num_layers do
        if i ~= avg_layer then
            w,h = unpack(filt_sizes[k])
            local conv = nn.SpatialConvolution(nchannels[k-1], nchannels[k],
                                               w, h)(layers[i-1])
            layers[i]  = nn.ReLU()(conv)

            end_width  = end_width  - (filt_sizes[k][1] - 1)
            end_height = end_height - (filt_sizes[k][2] - 1)
            k = k + 1
        else
            layers[i] = nn.SpatialAveragePooling(aps, aps, aps, aps)(layers[i-1])
            end_width  = torch.ceil(end_width  / aps)
            end_height = torch.floor(end_height / aps) -- Floor here, ceil there... Don't know why.
        end
    end

    -- Broken
    local batched = nn.Reshape(batch_size, nspeakers)(layers[num_layers])

    return get_gModule(layers[0], batched, batch_size, dummy)
end

return CNN
