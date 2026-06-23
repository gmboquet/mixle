"""Fit a Spearman ranking model (a distribution over permutations) with the optimize helper."""

import numpy as np

from pysp.stats import SpearmanRankingDistribution, SpearmanRankingEstimator
from pysp.inference import optimize

if __name__ == '__main__':
    # A modal ranking of 4 items; sample permutations concentrated around it.
    dist = SpearmanRankingDistribution([2, 3, 0, 1])
    data = dist.sampler(1).sample(1000)

    # Estimate the modal ranking (and dispersion) back from the sample.
    fit = optimize(data=data, estimator=SpearmanRankingEstimator(4), init_p=0.10, rng=np.random.RandomState(1))

    print('true: %s' % dist)
    print('fit : %s' % fit)
