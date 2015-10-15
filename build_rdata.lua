speaker = 'FCJF0'
phn = 'ix' -- 1 Phoneme for now

trainset = {}
trainset[phn] = {}

data = matio.load('timit/TRAIN/PHN_SPK/phn_spk.mat' % speaker)
for name, data in pairs(data) do
    if string.find(name, speaker) and string.find(name, phn) then
        idx = string.split(name, "_")[4]
        trainset[phn][idx] = data
    end
end
