%script to run inv cqts

clear all;
addpath ./cqt_functions;

Npad = 2^15;

T = 1024;
scparam.N = Npad; % 32 * 64
scparam.T = T;
scparam.Q = 32;
scparam.maxiters = 10;
%scparam.dse = 1;
filts = create_scattfilters( scparam );
ver = 1

for i=1:7;
    fname = sprintf('../../reconstructions/v%d/train_pred_%d_pad', ver, i)
    in = load(sprintf('%s.mat', fname));
    actual = invcqt(in.X, scparam, filts{1});
    audiowrite(sprintf('%s.wav', fname), actual, 16000)
end
