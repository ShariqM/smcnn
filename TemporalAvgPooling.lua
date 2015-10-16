local TemporalAvgPooling, parent = torch.class('nn.TemporalAvgPooling', 'nn.Module')

function TemporalAvgPooling:__init(poolsize, inchannels, stride)
    parent.__init(self)
    self.stride = poolsize/2 or stride
    self.net = nn.Sequential()
    self.net:add(nn.TemporalConvolution(inchannels, inchannels, poolsize, self.stride))

    self.net.modules[1].weight:fill(1/poolsize)
    self.net.modules[1].bias:fill(0)
end

function TemporalAvgPooling:updateOutput(input)
    self.output = self.net:updateOutput(input)
    out = self.output
    return self.output
end

function TemporalAvgPooling:updateGradInput(input, gradOutput)
    self.gradInput = self.net:updateGradInput(input, gradOutput)
    return self.gradInput
end
