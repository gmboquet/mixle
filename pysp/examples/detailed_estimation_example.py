"""Detailed example of estimation and model validation with a test set."""
import numpy as np
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
from pysp.stats import *
from pysp.utils.estimation import empirical_kl_divergence, partition_data

if __name__ == '__main__':

    rng = np.random.RandomState(1)

    # Create some data
    d10 = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0)], [0.5, 0.5])
    d11 = OptionalDistribution(CategoricalDistribution({'a': 0.5, 'b': 0.4, 'c': 0.1}), p=0.1)
    d12 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.2, 'b': 0.8}, 'b': {'a': 0.8, 'b': 0.2}},
                                  len_dist=PoissonDistribution(8.0))
    d13 = BernoulliSetDistribution({'a': 0.1, 'b': 0.3})
    d14 = MultivariateGaussianDistribution([-1.0, -1.0], [[2.0, 1.0], [1.0, 2.0]])

    d1 = CompositeDistribution([d10, d11, d12, d13, d14])
    sampler1 = d1.sampler()
    d20 = MixtureDistribution([GaussianDistribution(0.0, 1.0), GaussianDistribution(6.0, 1.0)], [0.5, 0.5])
    d21 = OptionalDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.1, 'c': 0.8}), p=0.2)
    d22 = MarkovChainDistribution({'a': 0.5, 'b': 0.5}, {'a': {'a': 0.8, 'b': 0.2}, 'b': {'a': 0.2, 'b': 0.8}},
                                  len_dist=PoissonDistribution(8.0))
    d23 = BernoulliSetDistribution({'a': 0.9, 'b': 0.8})
    d24 = MultivariateGaussianDistribution([1.0, 1.0], [[2.0, 1.0], [1.0, 2.0]])

    d2 = CompositeDistribution([d20, d21, d22, d23, d24])

    dist = MixtureDistribution([d1, d2], [0.5, 0.5])

    sampler = dist.sampler(seed=rng.randint(2 ** 31))
    data = sampler.sample(size=2000)

    train_data, valid_data = partition_data(data, [0.9, 0.1], rng)

    # Make the component estimator
    e0 = MixtureEstimator([GaussianEstimator()] * 2, pseudo_count=1.0)
    e1 = OptionalEstimator(CategoricalEstimator(pseudo_count=1.0), est_prob=False, pseudo_count=1.0)
    e2 = MarkovChainEstimator(pseudo_count=1.0, len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator()
    e4 = MultivariateGaussianEstimator()
    iest = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2, pseudo_count=1.0)

    e0 = MixtureEstimator([GaussianEstimator()] * 2)
    e1 = OptionalEstimator(CategoricalEstimator(), est_prob=False)
    e2 = MarkovChainEstimator(len_estimator=PoissonEstimator())
    e3 = BernoulliSetEstimator()
    e4 = MultivariateGaussianEstimator()
    est = MixtureEstimator([CompositeEstimator((e0, e1, e2, e3, e4))] * 2)

    # Estimate parameters
    # Note: Checkout pysp.utils.estimation.best_of/optimize for methods that handle this computation

    mm = initialize(train_data, iest, rng, 0.01)

    encoder = mm.dist_to_encoder()
    enc_data = seq_encode(train_data, encoder)
    enc_vdata = seq_encode(valid_data, encoder)
    _, old_ll = seq_log_density_sum(enc_vdata, mm)
    _, old_tll = seq_log_density_sum(enc_data, mm)

    dll = np.inf
    dtll = np.inf
    its_cnt = 0

    best_model = mm
    best_ll = old_ll

    while dtll > 1.0e-8 or its_cnt < 5:

        if its_cnt > 300:
            break

        its_cnt += 1

        mm_next = seq_estimate(enc_data, iest if its_cnt < 1 else est, mm)
        _, ll = seq_log_density_sum(enc_vdata, mm_next)
        _, tll = seq_log_density_sum(enc_data, mm_next)
        kl, _, _ = empirical_kl_divergence(mm_next, dist, enc_data)

        dtll = tll - old_tll
        dll = ll - old_ll

        if ll > best_ll:
            best_model = mm_next
            best_ll = ll

        mm = mm_next

        print('Iteration %d. LL=%e, VLL=%e, dLL=%e, KL[Est||True|data]=%f' % (its_cnt, tll, ll, dtll, kl))
        old_ll = ll
        old_tll = tll

    print(str(list(mm.w)))
    print(str(mm.components[0]))
    print(str(mm.components[1]))
