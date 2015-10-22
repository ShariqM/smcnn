-- NETWORK
num_conv_layers = 2
filt_sizes = {5,5}
pool_sizes = {2,2}
channels = {6,16}
channels[0] = 1 -- Input Channels (from data)

net = nn.Sequential()
for i = 1, num_conv_layers do
    net:add(nn.TemporalConvolution(channels[i-1], channels[i], filt_sizes[i])) -- MM ?
    net:add(nn.ReLU())
    net:add(nn.TemporalMaxPooling(pool_sizes[i]))
end

-- TODO
neurons = channels[num_conv_layers]
net:add(nn.View(neurons))
net:add(nn.Linear(16*5*5, 120))
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 10))
net:add(nn.LogSoftMax())

print('Lenet5\n' .. net:__tostring());
