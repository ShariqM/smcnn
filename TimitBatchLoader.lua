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
    self.weights = -1

    -- data  = matio.load(string.format('timit/TRAIN/process/data_%d.mat', self.num_examples))['X']
    -- data = matio.load(string.format('timit/TRAIN/process/DR1_data_%d.mat', self.num_examples))['X']
    -- phn  = matio.load(string.format('timit/TRAIN/process/DR1_phn_%d.mat', self.num_examples))['X']
    -- spk = matio.load(string.format('timit/TRAIN/process/DR1_spk_%d.mat', self.num_examples))['X'][1]

    trainset = matio.load('timit/TRAIN/process/DR1_trainset.mat')['X']
    testset  = matio.load('timit/TRAIN/process/DR1_testset.mat')['X']
    tr_spk = matio.load('timit/TRAIN/process/DR1_tr_spk.mat')['X'][1]
    te_spk = matio.load('timit/TRAIN/process/DR1_te_spk.mat')['X'][1]

    torch.save('timit/t7/DR1_trainset.t7', trainset)
    torch.save('timit/t7/DR1_testset.t7',  testset)
    torch.save('timit/t7/DR1_tr_spk.t7',   tr_spk)
    torch.save('timit/t7/DR1_te_spk.t7',   te_spk)

    trainset = torch.load('timit/t7/DR1_trainset.t7')
    testset  = torch.load('timit/t7/DR1_testset.t7')
    tr_spk   = torch.load('timit/t7/DR1_tr_spk.t7')
    te_spk   = torch.load('timit/t7/DR1_te_spk.t7')

    self.tr_examples = tr_spk:size()[1]
    self.te_examples = te_spk:size()[1]

    self.trainset = trainset / trainset:mean()
    self.testset  = testset  / trainset:mean() -- trainset mean I guess?
    self.tr_spk_label = tr_spk
    self.te_spk_label = te_spk

    self.nphonemes = 61
    self.nspeakers = tr_spk:max()

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

function TimitBatchLoader:next_batch()
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

function TimitBatchLoader:next_batch_c()
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
    self.te_weights = torch.Tensor(self.te_exmaples, energy:size()[1])

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
end

function TimitBatchLoader
    if self.batch_loading == false then
        self.speakers_picked = {}
        self.SAs_picked = {}
    end

    spk =
    while



function TimitBatchLoader:next_spk(tr)
    if self.weights == -1 then
        error('Setup weights before using next_spk())')
    end

    data_batch = torch.Tensor(self.batch_size, 1, self.timepoints, self.cqt_features)
    spk_batch  = torch.Tensor(self.batch_size)
    w_size     = self.tr_weights:size()[2]
    weights    = torch.Tensor(self.batch_size * w_size)

    if tr then
        data, spk_label, weights = self.trainset, self.tr_spk_label, self.tr_weights
    else
        data, spk_label, weights = self.testset, self.te_spk_label, self.te_weights
    end

    for i=1, self.batch_size do
        spk = next_speaker()
        idx = next_data_idx(tr, spk)

        data_batch[{i,{},{},{}}] = data[idx]
        spk_batch[i] = spk_label[idx]
        weights[{{(i-1)*w_size + 1, i*w_size}}] = weights[idx]
    end
    self.batch_loading = true

    return {data_batch, spk_batch, weights}
end

return TimitBatchLoader
