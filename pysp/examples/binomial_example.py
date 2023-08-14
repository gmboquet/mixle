import os
os.environ['NUMBA_DISABLE_JIT'] =  '1'
from pysp.stats import *
from pysp.utils.estimation import optimize
import numpy as np


if __name__ == '__main__':

    dist1 = BinomialDistribution(0.2, 10)
    dist2 = BinomialDistribution(0.8, 10)
    dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])
    data = dist.sampler(1).sample(1000)

    est = MixtureEstimator([BinomialEstimator()]*2)

    model = optimize(data, est, max_its=100, rng=np.random.RandomState(1))

    print(str(model))