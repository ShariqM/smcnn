function new_encoder(input, nchannels, filt_sizes, poolsize)
    enc1 = nn.TemporalConvolution(nchannels[1], nchannels[2], filt_sizes[1])(input)
    enc2 = nn.ReLU()(enc1)
    enc3 = nn.TemporalConvolution(nchannels[2], nchannels[3], filt_sizes[2])(enc2)
    enc4 = nn.ReLU()(enc3)

    pool = nn.TemporalAvgPooling(poolsize, nchannels[3])(enc4)

    return enc4, pool
end

function new_decoder(input, nchannels, filt_sizes)
    dec1 = nn.TemporalConvolution(nchannels[3], nchannels[2], filt_sizes[2])(input)
    dec2 = nn.ReLU()(dec1)
    dec3 = nn.TemporalConvolution(nchannels[2], nchannels[1], filt_sizes[1])(dec2)
    dec4 = nn.ReLU()(dec3)
    return dec4
end

function tie_weights(x1, x2)
    px1 = x1
    px2 = x2
    while true do -- Walk the children copying the parameters as you go
        params_px1 = px1.data.module:parameters()
        params_px2 = px2.data.module:parameters()

        if params_px1 then
            for i=1, #params_px1 do
                params_px2[i]:set(params_px1[i])
            end
        end

        if #px1.children == 0 then
            break
        end
        px1 = px1.children[1]
        px2 = px2.children[1]
    end
end

function get_comp_lost(filt_sizes, mult) -- mult - Pass 1 for encoding, 2 for encode&decoe
    sum = 0
    for i, f in pairs(filt_sizes) do
        sum = sum + mult * (f - 1) -- 2 * for backwards pass
    end
    return sum
end

function get_out_length_2(size, filt_sizes, poolsize)
    sum = get_comp_lost(filt_sizes, 1)
    return 1 + torch.floor((size - poolsize - sum)/(poolsize/2))
end
function get_out_length(x, filt_sizes, poolsize)
    return get_out_length_2(x:size()[1], filt_sizes, poolsize)
end

function get_narrow_x(x1, filt_sizes)
    sum = get_comp_lost(filt_sizes, 2)
    start = torch.floor(sum / 2)
    return x1:narrow(1, start, x1:size()[1] - sum)
end

function gradUpdate(net, x, y, hinge, mse, learningRate)
    output_x1, pred = unpack(net:forward(x))

    merr = mse:forward(output_x1, y[1])
    herr = hinge:forward(pred, y[2])

    gradMSE   = mse:backward(output_x1, y[1])
    gradHinge = hinge:backward(pred, y[2])

    net:zeroGradParameters()
    net:backward(x, {gradMSE, gradHinge})
    net:updateParameters(learningRate)
end

function toInt(b)
    if b then
        return 1
    else
        return -1
    end
end

function addKey(T, key, elem)
    if T[key] == nil then
        T[key] = elem
    end
    return T
end

function tableLength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end
