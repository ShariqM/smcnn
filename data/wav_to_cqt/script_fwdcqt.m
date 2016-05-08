%script to run fwd cqt

clear all;
addpath ./cqt_functions

Npad = 2^16;
T = 2048;
scparam.N = Npad;
scparam.T = T;
scparam.Q = 32;
%scparam.dse = 1;
filts = create_scattfilters( scparam );

fs = 16000
for j=1:32
    files = dir(sprintf('../wavs/s%d/*.wav', j));
    nfiles = length(files);
    j
    for i=1:nfiles
        i
        fname = files(i,1).name;
        [x1,Fs] = audioread(sprintf('../wavs/s%d/%s', j, fname));
        x1 = resample(x1,fs,Fs);
        x1 = x1(:)'; L1 = length(x1);

        in = zeros(Npad, 1);
        in(1:L1,1) = x1;
        X = fwdcqt(in, scparam, filts{1});

        mname = strcat(fname(1:length(fname)-3), 'mat')
        save(sprintf('../cqt_data/s%d/%s', j, mname), 'X')
    end
end
