require 'nn'
-- imagine we have one network we are interested in, it is called "p1_mlp"
p1_mlp = nn.Sequential(); p1_mlp:add(nn.Linear(5, 2))

-- But we want to push examples towards or away from each other so we make another copy
-- of it called p2_mlp; this *shares* the same weights via the set command, but has its
-- own set of temporary gradient storage that's why we create it again (so that the gradients
-- of the pair don't wipe each other)
p2_mlp = nn.Sequential(); p2_mlp:add(nn.Linear(5, 2))
p2_mlp:get(1).weight:set(p1_mlp:get(1).weight)
p2_mlp:get(1).bias:set(p1_mlp:get(1).bias)

-- we make a parallel table that takes a pair of examples as input.
-- They both go through the same (cloned) mlp
prl = nn.ParallelTable()
prl:add(p1_mlp)
prl:add(p2_mlp)

-- now we define our top level network that takes this parallel table
-- and computes the pairwise distance betweem the pair of outputs
mlp = nn.Sequential()
mlp:add(prl)
mlp:add(nn.PairwiseDistance(1))

-- and a criterion for pushing together or pulling apart pairs
crit = nn.HingeEmbeddingCriterion(1)

-- lets make two example vectors
x = torch.rand(5)
y = torch.rand(5)


-- Use a typical generic gradient update function
function gradUpdate(mlp, x, y, criterion, learningRate)
local pred = mlp:forward(x)
local err = criterion:forward(pred, y)
local gradCriterion = criterion:backward(pred, y)
mlp:zeroGradParameters()
mlp:backward(x, gradCriterion)
mlp:updateParameters(learningRate)
end

-- push the pair x and y together, notice how then the distance between them given
-- by print(mlp:forward({x, y})[1]) gets smaller
for i = 1, 10 do
   gradUpdate(mlp, {x, y}, 1, crit, 0.01)
   print('Together', mlp:forward({x, y})[1])
end

-- pull apart the pair x and y, notice how then the distance between them given
-- by print(mlp:forward({x, y})[1]) gets larger

for i = 1, 10 do
   gradUpdate(mlp, {x, y}, -1, crit, 0.01)
   print('Apart', mlp:forward({x, y})[1])
end
print (x:dist(y, 1))
