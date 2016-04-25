require 'torch'
require 'math'
require 'lfs'
require 'hdf5'

local GridSpeechBatchLoader = {}
GridSpeechBatchLoader.__index = GridSpeechBatchLoader

function GridSpeechBatchLoader.create(cqt_features, timepoints, batch_size, compile_test)
    local self = {}
    setmetatable(self, GridSpeechBatchLoader)

    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.timepoints = timepoints

    self.nspeakers = 3 -- For now
    self.nclass_speakers = 33
    self.gen_speaker = self.nclass_speakers
    self.nclass_words = 31
    self.gen_word = self.nclass_words
    -- self.trainset = matio.load('grid/words/data2.mat')['X']

    self.trainset = {}
    for spk=1, self.nspeakers do
        -- self.trainset[spk] = hdf5.open(string.format('grid/stft_data/S%d.h5', spk), 'r')
        -- self.trainset[spk] = matio.load(string.format('grid/stft_data/S%d.mat', spk))['X']
        -- print (self.trainset[spk]['lay'])
        -- debug.debug()
        if not compile_test then
            self.trainset[spk] = matio.load(string.format('grid/cqt_shariq/data/s%d.mat', spk))['X']
        else
            self.trainset[spk] = {}
            self.trainset[spk]['four'] = torch.zeros(2, cqt_features, timepoints)
            self.trainset[spk]['white'] = torch.zeros(2, cqt_features, timepoints)
        end
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

    self.words = {'four', 'white'}
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
    spk_labels  = torch.zeros(self.batch_size)
    word_labels = torch.zeros(self.batch_size)

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
            oword_idx = torch.random(1, wsz)
            oword = words[oword_idx]
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

        spk_labels[i] = sB
        word_labels[i] = oword_idx

        -- sAwX[{i,1,{},{}}] = self.trainset['S1'][word][torch.random(1,s1_wsz)]
        -- sAwY[{i,1,{},{}}] = self.trainset['S1'][oword][torch.random(1,s1_osz)]
        -- sBwX[{i,1,{},{}}] = self.trainset['S2'][word][torch.random(1,s2_wsz)]
            -- sBwY[{i,1,{},{}}] = self.trainset['S2'][oword][torch.random(1,s2_osz)]
        -- sBwY[{i,1,{},{}}] = self.trainset['S2'][oword][1]
    end


    return {sAwX, sBwX, sAwY, sBwY, spk_labels, word_labels}
end

function GridSpeechBatchLoader:next_batch()
    return self:next_batch_help(false)
end

function GridSpeechBatchLoader:next_test_batch()
    return self:next_batch_help(true)
end

function GridSpeechBatchLoader:next_class_batch()
    x = torch.Tensor(self.batch_size, 1, self.cqt_features, self.timepoints)
    spk_labels  = torch.zeros(self.batch_size)
    word_labels = torch.zeros(self.batch_size)

    for i=1, self.batch_size do
        spk = torch.random(1, self.nspeakers - 1)
        word_idx = torch.random(1, #self.words)
        word = self.words[word_idx]

        -- local timer = torch.Timer() FIXME too slow
        word_examples = self.trainset[spk][word]

        x[{i,1,{},{}}] = word_examples[torch.random(1, word_examples:size()[1])]
        spk_labels[i] = spk
        word_labels[i] = word_idx
    end

    return {x, spk_labels, word_labels}
end

function GridSpeechBatchLoader:next_adv_class_batch(cuda)
    local abatch_size = 2 * self.batch_size
    local x = torch.Tensor(abatch_size, 1, self.cqt_features, self.timepoints)
    local spk_labels  = torch.zeros(abatch_size)
    local word_labels = torch.zeros(abatch_size)

    -- True Distribution
    local true_x, true_spk_labels, true_word_labels = unpack(self:next_class_batch())

    -- Generative Distribution
    local sAwX, sBwX, sAwY, q, q, q = unpack(self:next_batch())
    if cuda then
        x = x:float():cuda()
        spk_labels = spk_labels:float():cuda()
        word_labels = word_labels:float():cuda()
        sAwX = sAwX:float():cuda()
        sBwX = sBwX:float():cuda()
        sAwY = sAwY:float():cuda()
    end

    local rsAwX = encoder:forward(sAwX)
    local rsBwX = encoder:forward(sBwX)
    local rsAwY = encoder:forward(sAwY)
    local diff  = diffnet:forward({rsAwX, rsBwX})
    local gen_x = decoder:forward({diff, rsAwY}) -- pred_sBwY
    local gen_spk_labels  = torch.zeros(self.batch_size):fill(self.gen_speaker)
    local gen_word_labels = torch.zeros(self.batch_size):fill(self.gen_word)

    if cuda then
        gen_spk_labels = gen_spk_labels:float():cuda()
        gen_word_labels = gen_word_labels:float():cuda()
    end

    -- Combine
    x[{{1, self.batch_size},{},{},{}}]             = true_x
    x[{{self.batch_size+1, abatch_size},{},{},{}}] = gen_x

    spk_labels[{{1,self.batch_size}}]           = true_spk_labels
    spk_labels[{{self.batch_size+1,abatch_size}}] = gen_spk_labels

    word_labels[{{1,self.batch_size}}]            = true_word_labels
    word_labels[{{self.batch_size+1,abatch_size}}] = gen_word_labels

    return {x, spk_labels, word_labels}
end

return GridSpeechBatchLoader
