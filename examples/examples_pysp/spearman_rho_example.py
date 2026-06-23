"""Fit a Spearman ranking model with the standard optimize helper."""

import numpy as np

from pysp.stats import SpearmanRankingDistribution, SpearmanRankingEstimator
from pysp.inference import optimize

if __name__ == '__main__':
    dist = SpearmanRankingDistribution([2, 3, 0, 1])

    data = dist.sampler(1).sample(1000)

    est = SpearmanRankingEstimator(4)
    fit = optimize(data=data, estimator=est, init_p=0.10, rng=np.random.RandomState(1))

    print(str(fit))
