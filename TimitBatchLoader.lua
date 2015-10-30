require 'torch'
require 'math'
require 'lfs'

local TimitBatchLoader = {}
TimitBatchLoader.__index = TimitBatchLoader

function TimitBatchLoader.create(cqt_features, batch_size, seq_length)
    local self = {}
    setmetatable(self, TimitBatchLoader)

    batches = math.floor(1024/seq_length)
    tlength = batches * seq_length -- Cut off the rest

    self.num_examples = 380 -- 2000 is broken (matio's fault I think)
    self.batches = batches
    self.tlength = tlength
    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.seq_length = seq_length

    -- data  = matio.load(string.format('timit/TRAIN/process/data_%d.mat', self.num_examples))['X']
    data = matio.load(string.format('timit/TRAIN/process/DR1_data_%d.mat', self.num_examples))['X']
    phn  = matio.load(string.format('timit/TRAIN/process/DR1_phn_%d.mat', self.num_examples))['X']
    spk  = matio.load(string.format('timit/TRAIN/process/DR1_spk_%d.mat', self.num_examples))['X']

    print (data:size())
    data = data / data:mean() -- Training does not work without this.


    self.nphonemes = 61
    self.nspeakers = 38
    self.data      = data
    self.phn_class = phn
    self.spk_class = spk
    self.current_batch = 0
    self.evaluated_batches = 1

    print('data load done.')
    collectgarbage()
    return self
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
        self.phn_batches = torch.Tensor(self.batches, self.batch_size, self.seq_length, self.nphonemes)
        self.spk_batches = torch.Tensor(self.batch_size, self.nspeakers)

        for i=1,self.batch_size do
            local idx = torch.random(self.num_examples)
            split_x = self.data[{idx,{1,self.tlength},{}}]:split(self.seq_length,1)
            split_y = self.phn_class[{idx,{1,self.tlength},{}}]:split(self.seq_length,1)
            for k=1,20 do
                self.x_batches[{k, i, {}, {}}] = split_x[k]
                self.phn_batches[{k, i, {}, {}}] = split_y[k]
            end
            self.spk_batches[{i,{}}] = self.spk_class[idx]
        end
    end

    self.evaluated_batches = self.evaluated_batches + 1
    self.current_batch = (self.current_batch+1)
    -- self.current_batch = 1

    return {self.x_batches[{self.current_batch,{},{},{}}],
           self.phn_batches[{self.current_batch,{},{},{}}],
           self.spk_batches, is_new_batch}
end

function TimitBatchLoader:next_spk()
    local idx = torch.random(self.num_examples)
    return {self.data[idx], self.spk_class[idx]}
end

return TimitBatchLoader
