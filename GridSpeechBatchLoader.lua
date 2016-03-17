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


function GridSpeechBatchLoader:next_batch(train)
    sAwX = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    sAwY = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    sBwX = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    sBwY = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)

    words = {'four', 'white', 'nine', 'zero', 'with', 'seven', 'at', 'set', 'soon'}
    -- words = {'four', 'white', 'zero'}
    -- words = {'four', 'white'}
    for i=1, self.batch_size do
        x = self.trainset['S1']
        y = self.trainset['S2']

        word = words[torch.random(1, #words)]
        oword = word
        while word == oword do
            oword = words[torch.random(1, #words)]
        end
        -- print (word, oword)
        -- word = 'four'
        -- oword = 'white'

        s1_wsz = self.trainset['S1'][word]:size()[1]
        s1_osz = self.trainset['S1'][oword]:size()[1]
        s2_wsz = self.trainset['S2'][word]:size()[1]
        s2_osz = self.trainset['S2'][oword]:size()[1]

        sAwX[{i,1,{},{}}] = self.trainset['S1'][word][torch.random(1,s1_wsz)]
        sAwY[{i,1,{},{}}] = self.trainset['S1'][oword][torch.random(1,s1_osz)]
        sBwX[{i,1,{},{}}] = self.trainset['S2'][word][torch.random(1,s2_wsz)]
        -- sBwY[{i,1,{},{}}] = self.trainset['S2'][oword][torch.random(1,s2_osz)]
        sBwY[{i,1,{},{}}] = self.trainset['S2'][oword][1]

        -- print (sAwX[{i,1,{},{}}])
        -- print (sAwY[{i,1,{},{}}])
        -- print (sBwX[{i,1,{},{}}])
        -- print (sBwY[{i,1,{},{}}])
    end

    return {sAwX, sBwX, sAwY, sBwY}
end

return GridSpeechBatchLoader
