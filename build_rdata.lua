local matio = require 'matio'
matio.use_lua_strings = true

local speaker = 'FCJF0'
local phn = 'ix' -- 1 Phoneme for now

trainset = {}
trainset[phn] = {}

all_data = matio.load('timit/TRAIN/PHN_SPK/phn_spk.mat' % speaker)
for name, data in pairs(all_data) do
    if string.find(name, speaker) and string.find(name, phn) then
        if data:size()[2] == 26 then
        -- if data:size()[2] > 20 then
            local idx = tonumber(string.split(name, "_")[4])
            trainset[phn][idx] = data
            -- print (idx, data:size())
        end
    end
end
