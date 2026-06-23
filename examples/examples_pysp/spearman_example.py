"""Estimate a Spearman ranking model from encoded permutation observations."""

import numpy as np

from pysp.stats import SpearmanRankingDistribution, SpearmanRankingEstimator

if __name__ == '__main__':

    dist = SpearmanRankingDistribution([2,3,0,1])

    data = dist.sampler(1).sample(100)

    est = SpearmanRankingEstimator(4)
    acc = est.accumulator_factory().make()

    enc_data = dist.dist_to_encoder().seq_encode(data)
    acc.seq_update(enc_data, np.ones(len(data)), None)

    est_model = est.estimate(None, acc.value())

    print(str(est_model))
