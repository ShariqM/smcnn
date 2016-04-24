require 'torch'
require 'math'
require 'lfs'
require 'hdf5'

local GridSpeechBatchLoader = {}
GridSpeechBatchLoader.__index = GridSpeechBatchLoader

function GridSpeechBatchLoader.create(cqt_features, timepoints, batch_size)
    local self = {}
    setmetatable(self, GridSpeechBatchLoader)

    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.timepoints = timepoints

    self.nspeakers = 3 -- For now
    -- self.trainset = matio.load('grid/words/data2.mat')['X']

    self.trainset = {}
    for spk=1, self.nspeakers do
        -- self.trainset[spk] = hdf5.open(string.format('grid/stft_data/S%d.h5', spk), 'r')
        self.trainset[spk] = matio.load(string.format('grid/stft_data/S%d.mat', spk))['X']
        -- print (self.trainset[spk]['lay'])
        -- debug.debug()
        -- self.trainset[spk] = matio.load(string.format('grid/cqt_shariq/data/s%d.mat', spk))['X']
    end


    -- self.words = {'four', 'white', 'nine', 'zero', 'with', 'seven', 'at', 'set', 'soon'}
    self.words = {'bin', 'lay', 'place', 'set',
                  'blue', 'green', 'red', 'white',
                  'one', 'two', 'three', 'four', 'five',
                  'six', 'seven', 'eight', 'nine', 'zero',
                  'again', 'now', 'please'}

    self.words = {'four', 'white', 'zero', 'seven', 'soon',}
    self.words = {
                  'blue', 'green', 'red', 'white',
                  'one', 'two', 'three', 'four', 'five',
                  'six', 'seven', 'zero',
                  'now', 'please'}

    -- self.words = {'four', 'white'}
    self.test_words = {'place', 'nine', 'again'} -- s8 eight is bad
    -- self.words = {'four', 'white'}

    print('data load done.')
    collectgarbage()
    return self
end


function GridSpeechBatchLoader:next_batch_help(test)
    sAwX = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    sAwY = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    sBwX = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    sBwY = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)

    for i=1, self.batch_size do
        sA = 1
        if test then
            sB = self.nspeakers - 1
        else
            sB = torch.random(2, self.nspeakers - 1)
            -- sB = torch.random(2, self.nspeakers)
        end

        if test then
            words = self.test_words
            wsz = #self.test_words
        else
            words = self.words
            wsz = #self.words
        end

        word = 'four'
        -- word = words[torch.random(1, wsz)]
        oword = word
        while word == oword do
            oword = words[torch.random(1, wsz)]
        end

        -- local timer = torch.Timer() FIXME too slow
        -- s1w = self.trainset[sA]:read(word):all()
        -- s1o = self.trainset[sB]:read(oword):all()
        -- s2w = self.trainset[sA]:read(word):all()
        -- s2o = self.trainset[sB]:read(oword):all()
        s1w = self.trainset[sA][word]
        s1o = self.trainset[sB][oword]
        s2w = self.trainset[sA][word]
        s2o = self.trainset[sB][oword]

        -- print (string.format("Time X: %.3f", timer:time().real))

        -- sAwX[{i,1,{},{}}] = s1w[1]
        -- sAwY[{i,1,{},{}}] = s1o[1]
        -- sBwX[{i,1,{},{}}] = s2w[1]
        -- sAwX[{i,1,{},{}}] = s1w[torch.random(1, 10)]
        -- sAwY[{i,1,{},{}}] = s1o[torch.random(1, 10)]
        -- sBwX[{i,1,{},{}}] = s2w[torch.random(1, 10)]
        sAwX[{i,1,{},{}}] = s1w[torch.random(1, s1w:size()[1])]
        sAwY[{i,1,{},{}}] = s1o[torch.random(1, s1o:size()[1])]
        sBwX[{i,1,{},{}}] = s2w[torch.random(1, s2w:size()[1])]
        sBwY[{i,1,{},{}}] = s2o[1]

        -- sAwX[{i,1,{},{}}] = self.trainset['S1'][word][torch.random(1,s1_wsz)]
        -- sAwY[{i,1,{},{}}] = self.trainset['S1'][oword][torch.random(1,s1_osz)]
        -- sBwX[{i,1,{},{}}] = self.trainset['S2'][word][torch.random(1,s2_wsz)]
            -- sBwY[{i,1,{},{}}] = self.trainset['S2'][oword][torch.random(1,s2_osz)]
        -- sBwY[{i,1,{},{}}] = self.trainset['S2'][oword][1]
    end


    return {sAwX, sBwX, sAwY, sBwY}
end

function GridSpeechBatchLoader:next_batch()
    return self:next_batch_help(false)
end

function GridSpeechBatchLoader:next_test_batch()
    return self:next_batch_help(true)
end

return GridSpeechBatchLoader
