
-- NETWORK

local CNN = {}

function CNN.cnn_works(nspeakers)
    local x = nn.Identity()()

    local view = nn.View(1024 * 175)(x)
    local linear = nn.Linear(1024 * 175, nspeakers)(view)
    local logsoft = nn.LogSoftMax()(linear)

    return {nn.gModule({x},{logsoft}), 1}
end

function CNN.cnn_simple(nspeakers)
    local x = nn.Identity()()
    print 'go'

    local batch_size = 17
    local view = nn.View(1024 * 175)(x)
    local linear = nn.Linear(1024 * 175, batch_size * nspeakers)(view)
    local relu = nn.ReLU()(linear)
    local view2 = nn.View(batch_size, nspeakers)(relu)

    local logsoft = nn.LogSoftMax()(view2)

    return {nn.gModule({x},{logsoft}), batch_size}
end

function CNN.cnn_localmin(nspeakers)
    local filt_sizes = {{175,1024}} -- Width height order Ugh
    -- local filt_sizes = {{20,100}} -- Width height order Ugh
    local nchannels  = {nspeakers}
    nchannels[0] = 1

    local layers     = {[0] = nn.Identity()()}
    local end_height = 1024
    local end_width  = 175
    local k = 1
    local i = 1

    w,h = unpack(filt_sizes[k])
    local conv = nn.SpatialConvolution(nchannels[k-1], nchannels[k],
                                               w, h)(layers[i-1])
    local relu  = nn.ReLU()(conv)
    end_height = end_height - (filt_sizes[k][2] - 1)
    end_width  = end_width  - (filt_sizes[k][1] - 1)

    print (end_width, end_height)
    local batch_size = end_width * end_height
    print ('bs', batch_size)
    local view = nn.View(batch_size, nspeakers)(relu)
    -- local linear = nn.Linear(nchannels[k] * end_width * end_height, nspeakers)(view)
    -- local logsoft = nn.LogSoftMax()(linear)
    local logsoft = nn.LogSoftMax()(view)

    return {nn.gModule({layers[0]},{logsoft}), batch_size}
end

function CNN.cnn(nspeakers)
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
    local view = nn.View(batch_size, nspeakers)(relu5)
    local logsoft = nn.LogSoftMax()(view)

    return {nn.gModule({x},{logsoft}), batch_size}
end


function CNN.cnn_avg(nspeakers)
    local aps = 4
    -- local filt_sizes = {{10,20}, {5,10}, {4,8}} -- Width height order Ugh
    local filt_sizes = {{10,20}, {5,10}, {4,8}} -- Width height order Ugh
    local nchannels  = {nspeakers,128,nspeakers}
    -- local nchannels  = {nspeakers,128,512,128,nspeakers}
    nchannels[0] = 1


    local num_layers = 3
    local layers     = {[0] = nn.Identity()()}
    local end_height = 1024
    local end_width  = 175
    local k = 1
    local i = 1

    for i=1,num_layers do
        w,h = unpack(filt_sizes[k])
        local conv = nn.SpatialConvolution(nchannels[k-1], nchannels[k],
                                                   w, h)(layers[i-1])
        local relu = nn.ReLU()(conv)
        end_height = end_height - (filt_sizes[k][2] - 1)
        end_width  = end_width  - (filt_sizes[k][1] - 1)
        k = k + 1

        layers[i] = nn.SpatialAveragePooling(aps, aps, aps, aps)(relu)
        end_height = torch.floor(end_height / aps) -- Floor here, ceil there... Don't know why.
        end_width  = torch.floor(end_width  / aps)
    end

    print ('Calculated end dim', end_width, end_height)
    local batch_size = end_width * end_height
    local view = nn.View(batch_size, nspeakers)(layers[num_layers])
    local logsoft = nn.Log()(nn.SpatialSoftMax()(view))

    return {nn.gModule({layers[0]},{logsoft}), batch_size}
end

function CNN.cnn_original(nspeakers)
    local aps = 2 -- Average Pool Size
    local filt_sizes = {{3,3}, {3,3}, {3,3}} -- 2 indicates average pooling
    local nchannels  = {8, 32, nspeakers}
    nchannels[0] = 1

    local num_layers = #filt_sizes + 1 -- For avg layer
    local avg_layer  = num_layers - 1

    local layers     = {[0] = nn.Identity()()}
    local end_width  = 1024
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

    -- print ('Predicted size', end_width, end_height)
    batch_size = end_width * end_height
    local batched = nn.Reshape(batch_size, nspeakers)(layers[num_layers])
    local logsoft = nn.Log()(nn.SpatialSoftMax()(batched))

    return {nn.gModule({layers[0]}, {logsoft}), batch_size}
end

return CNN
