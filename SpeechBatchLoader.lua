require 'torch'
require 'math'
require 'lfs'
require 'hdf5'

local SpeechBatchLoader = {}
SpeechBatchLoader.__index = SpeechBatchLoader

function SpeechBatchLoader.create(cqt_features, timepoints, batch_size)
    local self = {}
    setmetatable(self, SpeechBatchLoader)

    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.timepoints = timepoints
    self.weights_setup = false

    self.nspeakers = 4 -- Rest are not hdf5? or Corrupted?
    total_size =  512000
    self.segments = total_size / timepoints
    self.train_last = 0.96 * self.segments
    self.test_last = self.segments

    -- for i=1,self.nspeakers do
        -- print (i)
        -- rFile = hdf5.open(string.format('/home/joan/scatt2_fs16_NFFT2048/s%d.mat', i), 'r')
        -- data = rFile:read('/data/X1'):all()
--
        -- for j=481,self.segments do
            -- local wFile = hdf5.open(string.format('grid/s%d/s%d_%d.h5', i, i, j), 'w')
            -- wFile:write('/data/X1', data[{{(j-1)*timepoints+1,j*timepoints},{}}])
            -- wFile:close()
        -- end
        -- rFile:close()
    -- end
    -- debug.debug()


    -- data  = matio.load(string.format('timit/TRAIN/process/data_%d.mat', self.num_examples))['X']
    -- data = matio.load(string.format('timit/TRAIN/process/DR1_data_%d.mat', self.num_examples))['X']
    -- phn  = matio.load(string.format('timit/TRAIN/process/DR1_phn_%d.mat', self.num_examples))['X']
    -- spk = matio.load(string.format('timit/TRAIN/process/DR1_spk_%d.mat', self.num_examples))['X'][1]

    -- trainset = matio.load('timit/TRAIN/process/DR1_trainset.mat')['X']
    -- testset  = matio.load('timit/TRAIN/process/DR1_testset.mat')['X']
    -- tr_spk = matio.load('timit/TRAIN/process/DR1_tr_spk.mat')['X'][1]
    -- te_spk = matio.load('timit/TRAIN/process/DR1_te_spk.mat')['X'][1]
    -- tr_spk_to_idx = matio.load('timit/TRAIN/process/DR1_tr_spk_to_idx.mat')['X']
    -- te_spk_to_idx = matio.load('timit/TRAIN/process/DR1_te_spk_to_idx.mat')['X']

    -- torch.save('timit/t7/DR1_trainset.t7', trainset)
    -- torch.save('timit/t7/DR1_testset.t7',  testset)
    -- torch.save('timit/t7/DR1_tr_spk.t7',   tr_spk)
    -- torch.save('timit/t7/DR1_te_spk.t7',   te_spk)
    -- torch.save('timit/t7/DR1_tr_spk_to_idx.t7',   tr_spk_to_idx)
    -- torch.save('timit/t7/DR1_te_spk_to_idx.t7',   te_spk_to_idx)

    -- trainset = torch.load('timit/t7/DR1_trainset.t7')
    -- testset  = torch.load('timit/t7/DR1_testset.t7')
    -- self.trainset = trainset / trainset:mean()
    -- self.testset  = testset  / trainset:mean() -- trainset mean I guess?

    -- self.tr_spk_label  = torch.load('timit/t7/DR1_tr_spk.t7')
    -- self.te_spk_label  = torch.load('timit/t7/DR1_te_spk.t7')
    -- self.tr_spk_to_idx = torch.load('timit/t7/DR1_tr_spk_to_idx.t7')
    -- self.te_spk_to_idx = torch.load('timit/t7/DR1_te_spk_to_idx.t7')

    -- self.tr_examples = self.tr_spk_label:size()[1]
    -- self.te_examples = self.te_spk_label:size()[1]

    -- self.nphonemes = 61
    -- self.nspeakers = self.tr_spk_label:max()
    -- self.nspeakers = 4 -- HACK Let's start with 32

    self.batch_loading = false

    print('data load done.')
    collectgarbage()
    return self
end

function SpeechBatchLoader:init_seq(seq_length)
    batches = math.floor(self.timepoints/seq_length)
    tlength = batches * seq_length -- Cut off the rest
    self.batches = batches
    self.tlength = tlength
    self.seq_length = seq_length
    self.current_batch = 0
    self.evaluated_batches = 1
end

function SpeechBatchLoader:next_seq_batch()
    error("Broken")
    is_new_batch = false
    if self.current_batch % self.batches == 0 then
        is_new_batch = true
        self.current_batch = 0
        self.x_batches = torch.Tensor(self.batches, self.batch_size, self.seq_length, self.cqt_features)
        for i=1,self.batch_size do
            local idx = torch.random(self.num_examples)
            split = data[{idx,{1,self.tlength},{}}]:split(self.seq_length,1)
            for k=1,20 do
                self.x_batches[{k, i, {}, {}}] = split[k]
            end
        end
    end
    self.evaluated_batches = self.evaluated_batches + 1
    self.current_batch = (self.current_batch+1)
    -- self.current_batch = 1

    return self.x_batches[{self.current_batch,{},{},{}}], is_new_batch
end

function SpeechBatchLoader:next_seq_batch_c()
    error("Broken")
    is_new_batch = false
    if self.current_batch % self.batches == 0 then
        is_new_batch = true
        self.current_batch = 0
        self.x_batches = torch.Tensor(self.batches, self.batch_size, self.seq_length, self.cqt_features)
        self.phn_batches = torch.Tensor(self.batches, self.batch_size, self.seq_length)
        self.spk_batches = torch.Tensor(self.batch_size)

        for i=1,self.batch_size do
            local idx = torch.random(self.num_examples)
            split_x = self.data[{idx,{1,self.tlength},{}}]:split(self.seq_length,1)
            split_y = self.phn_class[{idx,{1,self.tlength}}]:split(self.seq_length,1)
            for k=1,self.batches do
                self.x_batches[{k, i, {}, {}}] = split_x[k]
                self.phn_batches[{k, i, {}}] = split_y[k]
            end
            self.spk_batches[i] = self.spk_class[idx]
        end
    end

    self.evaluated_batches = self.evaluated_batches + 1
    self.current_batch = (self.current_batch+1)
    -- self.current_batch = 1

    return {self.x_batches[{self.current_batch,{},{},{}}],
           self.phn_batches[{self.current_batch,{},{}}],
           self.spk_batches, is_new_batch}
end

function SpeechBatchLoader:get_energy(cnn, cuda, idx)
    test_batch = torch.Tensor(1, 1, self.timepoints, self.cqt_features)
    if idx <= self.tr_examples then
        test_batch[{{},{},{},{}}] = self.trainset[idx]
    else
        test_batch[{{},{},{},{}}] = self.testset[idx - self.tr_examples]
    end

    return dummy_cnn:forward(test_batch)
end

function SpeechBatchLoader:setup_weights(dummy_cnn, cuda)
    energy = self:get_energy(cnn, cuda, 1) -- Arbitrary index
    self.tr_weights = torch.Tensor(self.tr_examples, energy:size()[1])
    self.te_weights = torch.Tensor(self.te_examples, energy:size()[1])

    for i=1, self.tr_examples + self.te_examples do
        energy = self:get_energy(cnn, cuda, i)
        weight = energy / energy:max()
        threshold = weight:mean() - weight:std()/2

        weight:apply(function(i)
            if i < threshold then
                return 0
            else
                return 1
            end
        end)

        if i <= self.tr_examples then
            self.tr_weights[i] = weight
        else
            self.te_weights[i - self.tr_examples] = weight
        end
    end
    self.weights_setup = true
end

function SpeechBatchLoader:next_idx(train)
    if self.batch_loading == false then
        self.speaker_order = torch.randperm(self.nspeakers)
        self.batch_loading = true
    end

    spk = self.speaker_order[1]
    if self.speaker_order:size()[1] == 1 then
        self.speaker_order = torch.randperm(self.nspeakers)
    else
        self.speaker_order = self.speaker_order:narrow(1, 2, self.speaker_order:size()[1] - 1)
    end

    if train then
        -- Maybe I should track the sentences too but whatever for now.
        indexes = self.tr_spk_to_idx[spk]
    else
        indexes = self.te_spk_to_idx[spk]
    end

    metaidx = torch.random(1, indexes:size()[1])
    return indexes[metaidx]
end

function SpeechBatchLoader:next_batch(train)
    if self.weights_setup == false then
        error('Setup weights before using next_spk())')
    end

    data_batch = torch.Tensor(self.batch_size, 1, self.timepoints, self.cqt_features)
    spk_batch  = torch.Tensor(self.batch_size)
    w_size     = self.weights:size()[2]
    weight_batch = torch.Tensor(self.batch_size * w_size)

    if train then
        data, spk_label, weights = self.trainset, self.tr_spk_label, self.tr_weights
    else
        data, spk_label, weights = self.testset, self.te_spk_label, self.te_weights
    end

    for i=1, self.batch_size do
        idx = self:next_spk_and_idx(train)
        data_batch[{i,{},{},{}}] = data[idx]
        spk_batch[i] = spk_label[idx]
        -- print (string.format("IDX=%d, SPK=%d", idx, spk_label[idx]))
        weight_batch[{{(i-1)*w_size + 1, i*w_size}}] = weights[idx]
    end
    -- print ('')
    self.batch_loading = false

    return {data_batch, spk_batch, weight_batch}
end

function SpeechBatchLoader:get_grid_energy(cnn, cuda, spk, idx)
    test_batch = torch.Tensor(1, 1, self.timepoints, self.cqt_features)

    rFile = hdf5.open(string.format('grid/s%d/s%d_%d.h5', spk, spk, idx), 'r')
    data  = rFile:read('/data/X1'):all()
    test_batch[{{},{},{},{}}] = data
    rFile:close()
    return dummy_cnn:forward(test_batch)
end

function SpeechBatchLoader:setup_grid_weights(dummy_cnn, cuda)
    energy = self:get_grid_energy(cnn, cuda, 1, 1) -- Arbitrary index
    self.weights = torch.Tensor(self.nspeakers, self.segments, energy:size()[1])
    self.weights:fill(1)
    self.weights_setup = true
    return

    -- for spk=1, self.nspeakers do
        -- print (spk)
        -- for idx=1, self.segments do
            -- energy = self:get_grid_energy(cnn, cuda, spk, idx)
            -- weight = energy / energy:max()
            -- threshold = weight:mean() - weight:std()/2
--
            -- weight:apply(function(i)
                -- if i < threshold then
                    -- return 0
                -- else
                    -- return 1
                -- end
            -- end)
            -- self.weights[spk][idx] = weight
        -- end
    -- end
    -- self.weights_setup = true
end

function SpeechBatchLoader:next_spk_and_idx(train)
    if self.batch_loading == false then
        self.speaker_order = torch.randperm(self.nspeakers)
        self.batch_loading = true
    end

    spk = self.speaker_order[1]
    if self.speaker_order:size()[1] == 1 then
        self.speaker_order = torch.randperm(self.nspeakers)
    else
        self.speaker_order = self.speaker_order:narrow(1, 2, self.speaker_order:size()[1] - 1)
    end

    if train then
        return {spk, torch.random(1, self.train_last)}
    else
        return {spk, torch.random(self.train_last+1, self.test_last)}
    end
end

function SpeechBatchLoader:next_grid_batch(train)
    if self.weights_setup == false then
        error('Setup weights before using next_spk())')
    end

    data_batch = torch.Tensor(self.batch_size, 1, self.timepoints, self.cqt_features)
    spk_batch  = torch.Tensor(self.batch_size)
    w_size     = self.weights:size()[3]
    weight_batch = torch.Tensor(self.batch_size * w_size)

    for i=1, self.batch_size do
        spk, idx = unpack(self:next_spk_and_idx(train))

        rFile = hdf5.open(string.format('grid/s%d/s%d_%d.h5', spk, spk, idx), 'r')
        data  = rFile:read('/data/X1'):all()

        data_batch[{i,{},{},{}}] = data
        spk_batch[i] = spk
        weight_batch[{{(i-1)*w_size + 1, i*w_size}}] = self.weights[{spk,idx,{}}]

        rFile:close()
    end
    self.batch_loading = false

    return {data_batch, spk_batch, weight_batch}
end

return SpeechBatchLoader
