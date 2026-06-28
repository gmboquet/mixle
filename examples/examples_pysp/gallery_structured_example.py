"""Gallery: structured and latent-variable models, each built / sampled / re-estimated with EM.

Covers a first-order Markov chain, a heterogeneous mixture (components from different families), the
Indian Buffet Process feature-allocation model, a Chow-Liu tree and an ICL tree over discrete
features, a quantized HMM (theta^k parameterization), a tree-structured HMM, a segmental HMM, and
named-field Record / DictRecord models. Self-contained random data only. Other model groups have
their own focused scripts (hidden_markov, lda, lookback_hmm, association, set_edit, ...) or galleries
(gallery_graphs for the random-graph families); the grammar (HeterogeneousPCFG) and sparse-association
(SparseMarkovAssociation) models are exercised in pysp/tests.
"""
import numpy as np

from pysp.stats import *
from pysp.stats import IntegerChowLiuTreeEstimator
from pysp.inference import optimize

RNG = lambda: np.random.RandomState(1)

if __name__ == '__main__':
    print('# MarkovChain: first-order chain over symbols with a Poisson length model')
    d = MarkovChainDistribution({'a': 0.5, 'b': 0.5},
                                {'a': {'a': 0.9, 'b': 0.1}, 'b': {'a': 0.2, 'b': 0.8}},
                                len_dist=PoissonDistribution(6.0))
    m = optimize(d.sampler(1).sample(1000), MarkovChainEstimator(len_estimator=PoissonEstimator()),
                 max_its=1, rng=RNG())
    print('  transitions: %s' % m.transition_map)

    print('# HeterogeneousMixture: components drawn from different families (same support x>0)')
    d = HeterogeneousMixtureDistribution([GammaDistribution(2.0, 2.0), LogGaussianDistribution(1.0, 0.25)],
                                         [0.5, 0.5])
    m = optimize(d.sampler(1).sample(2000),
                 HeterogeneousMixtureEstimator([GammaEstimator(), LogGaussianEstimator()]),
                 max_its=40, rng=RNG())
    print('  weights: %s  components: %s' % (np.round(m.w, 2), [str(c) for c in m.components]))

    print('# IndianBuffetProcess: latent binary feature allocation')
    d = IndianBuffetProcessDistribution(num_features=4, feature_probs=[0.8, 0.5, 0.3, 0.1])
    m = optimize(d.sampler(1).sample(2000), IndianBuffetProcessEstimator(num_features=4),
                 max_its=20, rng=RNG())
    print('  feature_probs: %s' % np.round(m.feature_probs, 2))

    print('# ChowLiuTree: max-likelihood tree of pairwise dependencies over discrete features')
    rng = RNG()
    base = rng.randint(0, 2, size=2000)
    feats = [['a' if v else 'b' for v in base],
             ['a' if (v ^ (rng.rand(2000) < 0.1)).astype(int)[i] else 'b' for i, v in enumerate(base)],
             [('x' if rng.rand() < 0.5 else 'y') for _ in range(2000)]]
    tree_data = list(map(list, zip(*feats)))
    m = optimize(tree_data, ChowLiuTreeEstimator([CategoricalEstimator()] * 3), max_its=1, rng=RNG())
    print('  learned parents: %s' % list(m.parents))

    print('# ICLTree: integer-feature dependency tree learned from data')
    rng = RNG()
    base = rng.randint(0, 2, size=2000)
    icl_data = [[int(base[i]), int(base[i] ^ (rng.rand() < 0.1)), int(rng.rand() < 0.5)]
                for i in range(2000)]
    m = optimize(icl_data, IntegerChowLiuTreeEstimator(num_features=3, num_states=2), max_its=1, rng=RNG())
    print('  learned dependencies (feature, parent): %s'
          % [(int(a), None if b is None else int(b)) for a, b in m.dependency_list])

    print('# QuantizedHMM: HMM whose probabilities are powers of a shared theta')
    levels = [0.2, 0.8]
    d = QuantizedHiddenMarkovModelDistribution(
        theta=0.6, levels=levels, transition_exponents=[[0, 1], [1, 0]],
        emission_exponents=[[0, 1], [1, 0]], initial_exponents=[0, 1],
        len_dist=PoissonDistribution(5.0))
    m = optimize(d.sampler(1).sample(400),
                 QuantizedHiddenMarkovEstimator(2, levels=levels, len_estimator=PoissonEstimator()),
                 max_its=10, rng=RNG())
    print('  theta: %.3f' % float(m.theta))

    print('# TreeHiddenMarkov: hidden states over a (small, bounded) tree, Gaussian node emissions')
    d = TreeHiddenMarkovModelDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)],
                                          [0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]],
                                          len_dist=PoissonDistribution(0.8), terminal_level=3)
    m = optimize(d.sampler(1).sample(300),
                 TreeHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()],
                                           len_estimator=PoissonEstimator()),
                 max_its=8, rng=RNG())
    test = d.sampler(2).sample(50)
    print('  held-out mean log-density: %.3f' % np.mean([m.log_density(x) for x in test]))

    print('# SegmentalHiddenMarkov: HMM with explicit-duration (segmental) states')
    d = SegmentalHiddenMarkovModelDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)],
                                               [0.5, 0.5], [[0.7, 0.3], [0.3, 0.7]],
                                               len_dist=PoissonDistribution(6.0))
    m = optimize(d.sampler(1).sample(400),
                 SegmentalHiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()],
                                                len_estimator=PoissonEstimator()),
                 max_its=10, rng=RNG())
    test = d.sampler(2).sample(50)
    print('  held-out mean log-density: %.3f' % np.mean([m.log_density(x) for x in test]))

    print('# Record / DictRecord: named-field records (each field its own distribution)')
    d = RecordDistribution(['height', 'count'], [GaussianDistribution(0.0, 1.0), PoissonDistribution(4.0)])
    m = optimize(d.sampler(1).sample(1000),
                 RecordEstimator(['height', 'count'], [GaussianEstimator(), PoissonEstimator()]),
                 max_its=1, rng=RNG())
    print('  record fit: %s' % m)
    d = DictRecordDistribution(['height', 'count'], [GaussianDistribution(0.0, 1.0), PoissonDistribution(4.0)])
    m = optimize(d.sampler(1).sample(1000),
                 RecordEstimator(['height', 'count'], [GaussianEstimator(), PoissonEstimator()]),
                 max_its=1, rng=RNG())
    print('  dict-record fit: %s' % m)
