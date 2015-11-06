require 'torch'
require 'math'
require 'lfs'

local TimitBatchLoader = {}
TimitBatchLoader.__index = TimitBatchLoader

function TimitBatchLoader.create(cqt_features, timepoints, batch_size)
    local self = {}
    setmetatable(self, TimitBatchLoader)

    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.timepoints = timepoints
    self.weights_setup = false

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

    trainset = torch.load('timit/t7/DR1_trainset.t7')
    testset  = torch.load('timit/t7/DR1_testset.t7')
    self.trainset = trainset / trainset:mean()
    self.testset  = testset  / trainset:mean() -- trainset mean I guess?

    self.tr_spk_label  = torch.load('timit/t7/DR1_tr_spk.t7')
    self.te_spk_label  = torch.load('timit/t7/DR1_te_spk.t7')
    self.tr_spk_to_idx = torch.load('timit/t7/DR1_tr_spk_to_idx.t7')
    self.te_spk_to_idx = torch.load('timit/t7/DR1_te_spk_to_idx.t7')

    self.tr_examples = self.tr_spk_label:size()[1]
    self.te_examples = self.te_spk_label:size()[1]

    -- self.nphonemes = 61
    self.nspeakers = self.tr_spk_label:max()

    self.batch_loading = false

    print('data load done.')
    collectgarbage()
    return self
end

function TimitBatchLoader:init_seq(seq_length)
    batches = math.floor(self.timepoints/seq_length)
    tlength = batches * seq_length -- Cut off the rest
    self.batches = batches
    self.tlength = tlength
    self.seq_length = seq_length
    self.current_batch = 0
    self.evaluated_batches = 1
end

function TimitBatchLoader:next_seq_batch()
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

function TimitBatchLoader:next_seq_batch_c()
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

function TimitBatchLoader:get_energy(cnn, cuda, idx)
    test_batch = torch.Tensor(1, 1, self.timepoints, self.cqt_features)
    if idx <= self.tr_examples then
        test_batch[{{},{},{},{}}] = self.trainset[idx]
    else
        test_batch[{{},{},{},{}}] = self.testset[idx - self.tr_examples]
    end

    return dummy_cnn:forward(test_batch)
end

function TimitBatchLoader:setup_weights(dummy_cnn, cuda)
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
            self.tr_weights[i - self.tr_examples ] = weight
        end
    end
    self.weights_setup = true
end

function TimitBatchLoader:next_idx(train)
    if self.batch_loading == false then
        self.speaker_order = torch.randperm(self.nspeakers)
        self.SAs_picked = {}
    end

    spk = self.speaker_order[1]
    self.speaker_order:narrow(1, 2, self.speaker_order:size()[1] - 1)
    if self.speaker_order:size() == 0 then
        self.speaker_order = torch.randperm(self.nspeakers)
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

function TimitBatchLoader:next_batch(train)
    if self.weights_setup == false then
        error('Setup weights before using next_spk())')
    end

    data_batch = torch.Tensor(self.batch_size, 1, self.timepoints, self.cqt_features)
    spk_batch  = torch.Tensor(self.batch_size)
    w_size     = self.tr_weights:size()[2]
    weight_batch = torch.Tensor(self.batch_size * w_size)

    if train then
        data, spk_label, weights = self.trainset, self.tr_spk_label, self.tr_weights
    else
        data, spk_label, weights = self.testset, self.te_spk_label, self.te_weights
    end

    for i=1, self.batch_size do
        idx = self:next_idx(train)

        data_batch[{i,{},{},{}}] = data[idx]
        spk_batch[i] = spk_label[idx]
        weight_batch[{{(i-1)*w_size + 1, i*w_size}}] = weights[idx]
    end
    self.batch_loading = true

    return {data_batch, spk_batch, weight_batch}
end

return TimitBatchLoader
