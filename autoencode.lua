
net = nn.Sequential()

input = nn.Identity()()
enc1 = nn.TemporalConvolution(inchannels,nchannels[2], filt_size[2])(input)
enc2 = nn.ReLU()(enc1)
enc3 = nn.TemporalConvolution(nchannels[2], nchannels[3],filt_size[3])(enc2)
enc4 = nn.ReLU()(enc3)

pool = nn.TemporalAvgPooling(poolsize)(enc4)

dec1 = nn.TemporalConvolution(nchannels[3], nchannels[2], filt_size[3])(enc4)
dec2 = nn.ReLU()(dec1)
dec3 = nn.TemporalConvolution(nchannels[2], inchannels, filt_size[2])(dec2)
out = nn.ReLU()(dec3)

net = nn.gModule( {input}, { out, pool } )
