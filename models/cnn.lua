
-- NETWORK

local CNN = {}

function CNN.cnn(nspeakers)
    local filt_sizes = {{3,3}, {3,3}, {3,3}}
    local apool_size = 2
    local channels   = {{1,8,32,nspeakers}}

    local num_layers = #filt_sizes + 1 -- + 1 for average pooling
    local avg_layer  = num_layers - 1 -- Right before the last convolution

    local layers     = {[0] = nn.Identity()}
    local end_width  = 1024
    local end_height = 175
    for i = 1, num_layers do
        if i ~= avg_layer then
            local conv = nn.SpatialConvolution(nchannels[i-1], nchannels[i], filt_sizes[i])(layers[i-1])
            layers[i]  = nn.ReLU(conv)
            end_width  = end_width  - (filt_sizes[i][1] - 1)
            end_height = end_height - (filt_sizes[i][2] - 1)
        else
            layers[i] = nn.SpatialAveragePooling(apool_size, apool_size)(layers[i-1])
            end_width  = torch.ceil(end_width  / apool_size)
            end_height = torch.ceil(end_height / apool_size)
        end
    end

    batch_size = end_width * end_height
    local batched = nn.Reshape(batch_size, nspeakers)(layers[num_layers])
    local logsoft = nn.LogSoftMax()(batched)

    return {nn.gModule(layers[0], logsoft), batch_size}
end

return CNN
