import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
import numpy as np
from pysp.stats import *
from pysp.utils.estimation import optimize


if __name__ == '__main__':

    rng = np.random.RandomState(1)
    d1 = GaussianDistribution(1.0, 1.0)
    d2 = GaussianDistribution(2.0, 1.0)
    d3 = GaussianDistribution(3.0, 1.0)

    d0 = CategoricalDistribution({'a': 0.5, 'b': 0.2, 'c': 0.2, 'd': 0.1})

    dist = ConditionalDistribution({'a': d1, 'b': d2}, default_dist=d3, given_dist=d0)

    data = dist.sampler(1).sample(200)

    for i in range(10):
        print(data[i])

    est0 = GaussianEstimator()
    est1 = CategoricalEstimator()
    est  = ConditionalDistributionEstimator({'a': est0, 'b': est0}, default_estimator=est0, given_estimator=est1)

    init = initialize(data, est, rng=np.random.RandomState(2), p=0.10)
    model = estimate(data, est, prev_estimate=init)
    print(str(model))

    model = optimize(data, est)
    print(str(model))