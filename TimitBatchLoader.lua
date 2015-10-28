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

    self.num_examples = 200
    self.batches = batches
    self.tlength = tlength
    self.cqt_features = cqt_features
    self.batch_size = batch_size
    self.seq_length = seq_length

    -- data = torch.Tensor(self.num_examples, self.tlength, self.cqt_features)
    -- count = 1
    -- print('loading data files...')
    -- for fname in lfs.dir('timit/TRAIN') do
        -- if fname:find('DR') and fname:find('.mat') then
            -- print (fname)
            -- spk_data = matio.load('timit/TRAIN/%s' % fname)['data']
            -- for i=0,9 do
                -- data[count] = spk_data['X'][{{}, {1+i*1024,i*1024+tlength}}]
                -- count = count + 1
            -- end
        -- end
        -- if count == 100:
    -- end
    data = matio.load('timit/TRAIN/200.mat')['X']
    data = data / data:mean() -- Training does not work without this.
    print (data:size())

    self.data = data
    self.current_batch = 0
    self.evaluated_batches = 1

    print('data load done.')
    collectgarbage()
    return self
end

function TimitBatchLoader:load_new_data()
    self.x_batches = torch.Tensor(self.batches, self.batch_size, self.cqt_features)
    for i=1,self.batch_size do
        local idx = torch.random(self.num_examples)
        self.x_batches[{{}, i, {}}] = data[{idx,{},{}}]:split(self.seq_length,1)
    end
end

function TimitBatchLoader:next_batch()
    if self.current_batch == 0 or self.current_batch == 20 then
        self.current_batch = 0
        self.x_batches = torch.Tensor(self.batches, self.batch_size, self.seq_length, self.cqt_features)
        for i=1,self.batch_size do
            local idx = torch.random(self.num_examples)
            -- idx = i
            split = data[{idx,{},{1,self.tlength}}]:split(self.seq_length,2)
            for k=1,20 do
                self.x_batches[{k, i, {}, {}}] = split[k]
            end
            -- self.x_batches[{i, {}, {}}] = torch.Tensor(data[{idx,{1,self.tlength},{}}]:split(self.seq_length,2))
        end
    end
    self.evaluated_batches = self.evaluated_batches + 1
    self.current_batch = (self.current_batch+1)
    -- self.current_batch = 1

    -- print (self.x_batches[{self.current_batch,{},{},{}}]:mean()) -- 0.00039700941962218
    -- return self.x_batches[{self.current_batch,{},{},{}}] / (self.x_batches[{self.current_batch,{},{},{}}]:mean())
    return self.x_batches[{self.current_batch,{},{},{}}]
end

return TimitBatchLoader
