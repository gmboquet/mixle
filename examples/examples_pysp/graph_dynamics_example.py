"""Model evolving integer sets with lookback HMM graph-dynamics emissions."""

#import os

import numpy as np

from pysp.stats import *
from pysp.stats.latent.look_back_hmm import LookbackHiddenMarkovDistribution, LookbackHiddenMarkovEstimator
from pysp.utils.estimation import optimize

if __name__ == '__main__':

    # P(set_1)
    d0 = IntegerBernoulliSetDistribution(np.log([0.3, 0.3, 0.3]))
    # P(set_2 | set_1, Z=0)
    dist1 = IntegerBernoulliEditDistribution(np.log([[0.01, 0.01, 0.50], [0.05, 0.05, 0.90]]).T, init_dist=d0)
    # P(set_2 | set_1, Z=1)
    dist2 = IntegerBernoulliEditDistribution(np.log([[0.01, 0.50, 0.01], [0.05, 0.90, 0.05]]).T, init_dist=d0)
    # P(set_2 | set_1, Z=2)
    dist3 = IntegerBernoulliEditDistribution(np.log([[0.50, 0.01, 0.01], [0.90, 0.05, 0.05]]).T, init_dist=d0)

    init_dists = [SequenceDistribution(d0, CategoricalDistribution({1:1.0}))]*3
    states     = [dist1, dist2, dist3]
    len_dist   = CategoricalDistribution({7:0.5, 8:0.25, 9:0.25})
    transition = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    w          = [0.4, 0.3, 0.3]

    dist = LookbackHiddenMarkovDistribution(states, w=w, transitions=transition, lag=1, init_dist=init_dists, len_dist=len_dist)

    data = dist.sampler(seed=1).sample(200)

    print(data[0])
    print(data[1])
    print(data[2])

    print(dist.seq_log_density(dist.seq_encode(data[:10])))
    print([dist.log_density(data[i]) for i in range(10)])

    est0 = SequenceEstimator(IntegerBernoulliSetEstimator(3), len_estimator=CategoricalEstimator())
    est1 = IntegerBernoulliEditEstimator(3)
    est  = LookbackHiddenMarkovEstimator([est1]*3, lag=1, init_estimators=[est0]*3, len_estimator=CategoricalEstimator())

    model = optimize(data, est, max_its=1000, delta=None, rng=np.random.RandomState(1))

    print(str(model))
