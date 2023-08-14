"""Simulate data and estimate from a Mixture distribution."""
import os

os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np

from pysp.stats import *
from pysp.utils.estimation import optimize, partition_data

if __name__ == '__main__':
    rng = np.random.RandomState(1)

    # Create the example distribution (we'll sample data from this)

    d10 = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0)], [0.5, 0.5])
    d11 = OptionalDistribution(CategoricalDistribution({'a': 0.5, 'b': 0.4, 'c': 0.1}), p=0.1)
    d12 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.2, 'b': 0.8}, 'b': {'a': 0.8, 'b': 0.2}},
                                  len_dist=PoissonDistribution(8.0))
    d13 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
    d14 = MultivariateGaussianDistribution([-1.0, -1.0], [[2.0, 1.0], [1.0, 2.0]])

    d1 = CompositeDistribution([d10, d11, d12, d13, d14])

    d20 = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(6.0, 1.0)], [0.5, 0.5])
    d21 = OptionalDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.1, 'c': 0.8}), p=0.2)
    d22 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.8, 'b': 0.2}, 'b': {'a': 0.2, 'b': 0.8}},
                                  len_dist=PoissonDistribution(8.0))
    d23 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
    d24 = MultivariateGaussianDistribution([1.0, 1.0], [[2.0, 1.0], [1.0, 2.0]])

    d2 = CompositeDistribution([d20, d21, d22, d23, d24])

    dist = MixtureDistribution([d1, d2], [0.5, 0.5])

    # Sample data from the distribution

    sampler = dist.sampler(seed=rng.randint(2 ** 31))
    data = sampler.sample(size=2000)

    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

    # Specify the model estimator

    # Use regularization when initializing
    e0 = MixtureEstimator([GaussianEstimator()] * 2, pseudo_count=1.0)
    e1 = OptionalEstimator(CategoricalEstimator(pseudo_count=1.0), est_prob=False, pseudo_count=1.0)
    e2 = MarkovChainEstimator(pseudo_count=1.0, len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator(keys='asdf')
    e4 = MultivariateGaussianEstimator()
    iest = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2, pseudo_count=1.0, fixed_weights=[0.2, 0.8])

    # And one without regularization
    e0 = MixtureEstimator([GaussianEstimator()] * 2)
    e1 = OptionalEstimator(CategoricalEstimator(), est_prob=False)
    e2 = MarkovChainEstimator(len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator(keys='asdf')
    e4 = MultivariateGaussianEstimator()
    est = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2, fixed_weights=[0.2, 0.8])

    # Estimate parameters
    mm = optimize(train_data, est, max_its=100, print_iter=20, init_estimator=iest, rng=np.random.RandomState(1))

    print(str(list(mm.w)))
    print(str(mm.components[0]))
    print(str(mm.components[1]))
