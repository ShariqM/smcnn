require 'torch'
require 'math'
require 'lfs'

local GridSpeechBatchLoader = {}
GridSpeechBatchLoader.__index = GridSpeechBatchLoader

function GridSpeechBatchLoader.create(cqt_features, timepoints, batch_size)
    local self = {}
    setmetatable(self, GridSpeechBatchLoader)

    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.timepoints = timepoints

    self.nspeakers = 2 -- For now
    self.trainset = matio.load('grid/words/data2.mat')['X']

    print('data load done.')
    collectgarbage()
    return self
end


function GridSpeechBatchLoader:get_vecs(word)
    x = self.trainset['S1']
    sAwX = torch.Tensor(1, self.cqt_features, self.timepoints)
    sBwX = torch.Tensor(1, self.cqt_features, self.timepoints)

    -- Asz = self.trainset['S1'][word]:size()[1]
    -- Bsz = self.trainset['S2'][word]:size()[1]
    -- sAwX = self.trainset['S1'][word][torch.random(1,Asz)]
    -- sBwX = self.trainset['S2'][word][torch.random(1,Bsz)]
    -- sAwX = self.trainset['S1'][word][{{1,3},{},{}}]
    -- sBwX = self.trainset['S2'][word][{{1,3},{},{}}]
    sAwX[{1,{},{}}] = self.trainset['S1'][word][1]
    sBwX[{1,{},{}}] = self.trainset['S2'][word][1]

    return sAwX, sBwX
end

function GridSpeechBatchLoader:next_batch(train)

    word = 'four'
    oword = 'white'

    sAwX, sBwX = self:get_vecs(word)
    sAwY, sBwY = self:get_vecs(oword)

    return {sAwX, sBwX, sAwY, sBwY}
end

return GridSpeechBatchLoader
