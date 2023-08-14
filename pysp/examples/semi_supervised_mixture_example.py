"""Example of semi-supervised mixture example. Note that None is included for obs with no known labels."""
import os

os.environ['NUMBA_DISABLE_JIT'] = '1'
from pysp.stats import *
from pysp.stats.ss_mixture import *
from pysp.utils.estimation import best_of


if __name__ == '__main__':
    seq_samp = 10
    c1 = CompositeDistribution((SequenceDistribution(CategoricalDistribution({'a': 0.8, 'b': 0.1, 'c': 0.1}),
                                                     len_dist=CategoricalDistribution({seq_samp: 1.0})),
                                GaussianDistribution(0.0, 1.0)))
    c2 = CompositeDistribution((SequenceDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.8, 'c': 0.1}),
                                                     len_dist=CategoricalDistribution({seq_samp: 1.0})),
                                GaussianDistribution(1.0, 1.0)))
    c3 = CompositeDistribution((SequenceDistribution(CategoricalDistribution({'a': 0.1, 'b': 0.1, 'c': 0.8}),
                                                     len_dist=CategoricalDistribution({seq_samp: 1.0})),
                                GaussianDistribution(2.0, 1.0)))

    dist = SemiSupervisedMixtureDistribution([c1, c2, c3], [0.6, 0.3, 0.1])

    data = dist.sampler(seed=1).sample(1000)
    rng = np.random.RandomState(1)

    data = [((['a'] * seq_samp, 0.0), [(0, 1.0)]), ((['b'] * seq_samp, 1.0), [(1, 1.0)]),
            ((['c'] * seq_samp, 2.0), [(2, 1.0)])] + [(u, None) for u in data]
    suff_stat = {'a': 1.0 / 3.0, 'b': 1.0 / 3.0, 'c': 1.0 / 3.0}

    est = SemiSupervisedMixtureEstimator([CompositeEstimator(
        (SequenceEstimator(CategoricalEstimator(pseudo_count=1.0e-3, suff_stat=suff_stat)), GaussianEstimator()))] * 3,
                                         pseudo_count=1.0e-6)

    _, model = best_of(data, data, est, 10, 1000, 0.05, 1.0e-8, rng, print_iter=1000)

    print(str(model.w))
    print('\n'.join(map(str, model.components)))
