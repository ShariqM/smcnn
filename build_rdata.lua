require 'helpers'

local matio = require 'matio'
matio.use_lua_strings = true

local speaker = 'FCJF0'
local phn = 'ix' -- 1 Phoneme for now

trainset = {}
ts = {}
ts['all'] = {}
ts['hs']={{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}}

all_data = matio.load('timit/TRAIN/PHN_SPK/phn_spk.mat' % speaker)
local count = 1
for name, data in pairs(all_data) do
    -- if string.find(name, speaker) and string.find(name, phn) then
    -- if true then
    if string.find(name, speaker) then
        -- if data:size()[2] == 26 then

        tdata = data:transpose(1,2)
        if tdata:size()[1] > 18 then
            d, speaker, phn, idx = unpack(string.split(name, "_"))
            idx = tonumber(idx) + 1
            print (phn)
            len = get_out_length(tdata, filt_sizes, poolsize)
            ts.all[#ts.all+1] = {tdata, phn, speaker, len}
            ts.hs[len][#ts.hs[len]+1] = count

            -- print ('idx', idx, 'size', data:size()[2], 'hinge size:', tmp)
            count = count + 1
        end
    end
end
print ("Total: ", count)
