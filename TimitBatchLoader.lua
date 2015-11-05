require 'torch'
require 'math'
require 'lfs'

local TimitBatchLoader = {}
TimitBatchLoader.__index = TimitBatchLoader

function TimitBatchLoader.create(cqt_features, batch_size)
    local self = {}
    setmetatable(self, TimitBatchLoader)

    self.num_examples = 380 -- 2000 is broken (matio's fault I think)
    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.total_tlength = 1024

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

    -- print (data:size())
    data = data / data:mean() -- Training does not work without this.
    -- print (data:mean())
    -- print (data:var())

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

function TimitBatchLoader:next_spk()
    data_batch = torch.Tensor(self.batch_size, 1, self.total_tlength, self.cqt_features)
    spk_labels = torch.Tensor(self.batch_size)
    for i=1, self.batch_size do
        local idx = torch.random(self.num_examples)
        idx = i
        data_batch[{i,{},{},{}}] = self.data[idx]
        spk_labels[i] = self.spk_class[idx]
    end
    return {data_batch, spk_labels}
    -- sz = 124
    -- tmp = self.data[idx][{{1,sz},{}}]
    -- return {torch.reshape(tmp,1,sz,175), self.spk_class[idx]}
    -- return {torch.reshape(self.data[idx],1,1024,175), 10}
end

return TimitBatchLoader
