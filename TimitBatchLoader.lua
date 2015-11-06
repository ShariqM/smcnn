require 'torch'
require 'math'
require 'lfs'

local TimitBatchLoader = {}
TimitBatchLoader.__index = TimitBatchLoader

function TimitBatchLoader.create(cqt_features, total_tlength, batch_size)
    local self = {}
    setmetatable(self, TimitBatchLoader)

    self.num_examples = 380 -- 2000 is broken (matio's fault I think)
    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.total_tlength = total_tlength
    self.weights = -1

    -- data  = matio.load(string.format('timit/TRAIN/process/data_%d.mat', self.num_examples))['X']
    -- data = matio.load(string.format('timit/TRAIN/process/DR1_data_%d.mat', self.num_examples))['X']
    -- phn  = matio.load(string.format('timit/TRAIN/process/DR1_phn_%d.mat', self.num_examples))['X']
    -- spk = matio.load(string.format('timit/TRAIN/process/DR1_spk_%d.mat', self.num_examples))['X'][1]

    -- torch.save('timit/DR1_cqt.t7', data)
    -- torch.save('timit/DR1_phn.t7', phn)
    -- torch.save('timit/DR1_spk.t7', spk)

    data = torch.load('timit/DR1_cqt.t7')
    phn  = torch.load('timit/DR1_phn.t7')
    spk  = torch.load('timit/DR1_spk.t7')
    print (spk:size())

    data = data / data:mean() -- Training does not work without this.

    self.nphonemes = 61
    self.nspeakers = 38
    self.data      = data
    self.phn_class = phn
    self.spk_class = spk

    print('data load done.')
    collectgarbage()
    return self
end

function TimitBatchLoader:init_seq(seq_length)
    batches = math.floor(self.total_tlength/seq_length)
    tlength = batches * seq_length -- Cut off the rest
    self.batches = batches
    self.tlength = tlength
    self.seq_length = seq_length
    self.current_batch = 0
    self.evaluated_batches = 1
end

function TimitBatchLoader:next_batch()
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
    test_batch = torch.Tensor(1, 1, self.total_tlength, self.cqt_features)
    test_batch[{{},{},{},{}}] = self.data[idx]
    -- if cuda then test_batch = test_batch:float():cuda() end
    return dummy_cnn:forward(test_batch)
end

function TimitBatchLoader:setup_weights(dummy_cnn, cuda)
    energy = self:get_energy(cnn, cuda, 1) -- Arbitrary index
    self.weights = torch.Tensor(self.num_examples, energy:size()[1])
    -- if cuda then self.weights = self.weights:float():cuda() end
    for i=1, self.num_examples do
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

        self.weights[i] = weight
    end
end

function TimitBatchLoader:next_spk()
    if self.weights == -1 then
        error('Setup weights before using next_spk())')
    end

    data_batch = torch.Tensor(self.batch_size, 1, self.total_tlength, self.cqt_features)
    spk_labels = torch.Tensor(self.batch_size)
    w_size     = self.weights:size()[2]
    weights    = torch.Tensor(self.batch_size * w_size)

    -- g = 1
    for i=1, self.batch_size do
        j = i + 2
        for idx=1, self.num_examples do
            if self.spk_class[idx] == j then
                data_batch[{i,{},{},{}}] = self.data[idx]
                spk_labels[i] = self.spk_class[idx]
                weights[{{(i-1)*w_size + 1, i*w_size}}] = self.weights[idx]
                -- g = g + 1
                break
            end
        end
    end
    return {data_batch, spk_labels, weights}
end

return TimitBatchLoader
