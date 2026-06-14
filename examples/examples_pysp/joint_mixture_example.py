"""Fit a joint mixture model that couples two heterogeneous observation views."""

import numpy as np
from pysp.stats import *
from pysp.utils.estimation import best_of
from pysp.utils.evaluation import empirical_kl_divergence, partition_data

if __name__ == '__main__':

    rng = np.random.RandomState(1)

    # Create the data
    d11 = CompositeDistribution([CategoricalDistribution({'a': 1.0, 'b': 0.0, 'c': 0.0}), GaussianDistribution(mu=-6.0, sigma2=1.0)])
    d12 = CompositeDistribution([CategoricalDistribution({'a': 0.0, 'b': 1.0, 'c': 0.0}), GaussianDistribution(mu= 0.0, sigma2=1.0)])
    d13 = CompositeDistribution([CategoricalDistribution({'a': 0.0, 'b': 0.0, 'c': 1.0}), GaussianDistribution(mu= 6.0, sigma2=1.0)])

    d21 = SequenceDistribution(CompositeDistribution([GaussianDistribution(mu=-6.0, sigma2=1.0), GammaDistribution(1.0, 3.0)]), PoissonDistribution(3.0), len_normalized=True)
    d22 = SequenceDistribution(CompositeDistribution([GaussianDistribution(mu= 0.0, sigma2=1.0), GammaDistribution(3.0, 3.0)]), PoissonDistribution(3.0), len_normalized=True)
    d23 = SequenceDistribution(CompositeDistribution([GaussianDistribution(mu= 6.0, sigma2=1.0), GammaDistribution(1.0, 3.0)]), PoissonDistribution(3.0), len_normalized=True)

    taus12 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    taus21 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    w1     = [0.6, 0.3, 0.1]
    w2     = [0.7, 0.2, 0.1]

    dist     = JointMixtureDistribution([d11, d12, d13], [d21, d22, d23], w1, w2, taus12, taus21)


    sampler  = dist.sampler(seed=1)
    data     = sampler.sample(10000)

    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

    # Make the estimator
    est1 = CompositeEstimator([CategoricalEstimator(pseudo_count=1.0), GaussianEstimator()])
    est2 = SequenceEstimator(CompositeEstimator([GaussianEstimator(), GammaEstimator()]), PoissonEstimator())
    est = JointMixtureEstimator([est1]*3, [est2]*3, pseudo_count=(0.001, 0.001, 0.001))


    # Estimate parameters
    _, mm = best_of(train_data, valid_data, est, 5, 100, 0.01, 1.0e-8, rng)

    enc_vdata = seq_encode(valid_data, model=mm)
    kl, _, _  = empirical_kl_divergence(mm, dist, enc_vdata)

    print('KL[Estimate||True | data] = %f'%(kl))

    print(str(mm))
