"""Example of HMM sampling and estimation using 'best_of' to find optimal model fit with different initial conditions.
Note that Numba is set for use here. """
USE_NUMBA = True
import numpy as np
import os

if not USE_NUMBA:
    os.environ['NUMBA_DISABLE_JIT'] = '1'


from pysp.stats import *
from pysp.utils.estimation import best_of, partition_data

if __name__ == '__main__':
    rng = np.random.RandomState(2)

    # <code for generating data>
    v = 0.1
    d11 = CompositeDistribution(
        (GaussianDistribution(-20.0, 1.0), CategoricalDistribution({'a': 0.5 - v, 'b': 0.5 - v, 'c': v, 'd': v})))
    d12 = CompositeDistribution(
        (GaussianDistribution(0.0, 1.0), CategoricalDistribution({'a': v, 'b': v, 'c': 0.5 - v, 'd': 0.5 - v})))
    d13 = CompositeDistribution(
        (GaussianDistribution(20.0, 1.0), CategoricalDistribution({'a': v, 'b': 0.5 - v, 'c': 0.5 - v, 'd': v})))

    v = 0.1
    d21 = CompositeDistribution(
        (GaussianDistribution(-5.0, 1.0), CategoricalDistribution({'a': 0.5 - v, 'b': 0.5 - v, 'c': v, 'd': v})))
    d22 = CompositeDistribution(
        (GaussianDistribution(0.0, 1.0), CategoricalDistribution({'a': v, 'b': v, 'c': 0.5 - v, 'd': 0.5 - v})))
    d23 = CompositeDistribution(
        (GaussianDistribution(5.0, 1.0), CategoricalDistribution({'a': v, 'b': 0.5 - v, 'c': 0.5 - v, 'd': v})))

    dist1 = HiddenMarkovModelDistribution([d11, d12, d13], [0.4, 0.4, 0.2],
                                          [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], None,
                                          len_dist=PoissonDistribution(4.0))
    dist2 = HiddenMarkovModelDistribution([d21, d22, d23], [0.4, 0.4, 0.2],
                                          [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], None,
                                          len_dist=PoissonDistribution(8.0))
    dist = MixtureDistribution([dist1, dist2], [0.5, 0.5])

    train_data = dist.sampler(seed=rng.randint(2 ** 32 - 1)).sample(2)

    sampler = dist.sampler(seed=rng.randint(2 ** 32 - 1))
    data = sampler.sample(1000)

    # </code for generating data>

    # Create an initial estimator
    est1 = CompositeEstimator((GaussianEstimator(), CategoricalEstimator(pseudo_count=1.0)))
    est2 = HiddenMarkovEstimator([est1] * 3, pseudo_count=(1.0, 1.0), keys=('init_key', 'trans_key', None),
                                 len_estimator=PoissonEstimator(), use_numba=USE_NUMBA)
    iest = MixtureEstimator([est2] * 2, pseudo_count=1.0)

    # Create the estimator
    est1 = CompositeEstimator((GaussianEstimator(), CategoricalEstimator()))
    est2 = HiddenMarkovEstimator([est1] * 3, keys=('init_key', 'trans_key', None), len_estimator=PoissonEstimator(),
                                 use_numba=USE_NUMBA)
    est = MixtureEstimator([est2] * 2)

    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

    ll, mm = best_of(train_data, valid_data, est, 20, 100, 0.01, 1.0e-8, rng, init_estimator=iest)

    print(str(mm))
