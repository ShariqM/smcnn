
-- NETWORK

local CNN = {}

function CNN.cnn(nspeakers)
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
            end_height = torch.floor(end_height / aps) -- wtf... floor for this one
        end
    end

    print ('Predicted size', end_width, end_height)
    batch_size = end_width * end_height
    local batched = nn.Reshape(batch_size, nspeakers)(layers[num_layers])
    local logsoft = nn.Log()(nn.SpatialSoftMax()(batched))

    return {nn.gModule({layers[0]}, {logsoft}), batch_size}
end

return CNN
