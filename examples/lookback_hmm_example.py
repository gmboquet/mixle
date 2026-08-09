"""LookbackHiddenMarkovModel: an HMM whose emission depends on the PREVIOUS observation, not just the state.

A plain HMM emits each observation independently given the hidden state. A lookback HMM (``lag=1``)
instead emits ``x[t]`` from a conditional model of ``x[t] | x[t-1]``, selected by the hidden state --
so each of the 3 hidden states here owns an ``IntegerMarkovChainDistribution`` over the observation
alphabet, and ``init_dist`` supplies the distribution for the first ``lag`` positions that have no
predecessor.

The planted states are three cyclic shifts of the same doubly-stochastic sticky transition
matrix, and the first symbol is UNIFORM -- so the per-position marginal is exactly uniform under
every state at every position (uniform is stationary for all three shifts), and the states are
distinguishable ONLY through the observation-to-observation dependence. (The uniform start is
load-bearing: with a non-uniform first symbol the second position's marginals differ across
states by a total variation of ~0.07, and a lag-0 model could partially tell them apart --
STAT-RR17-18.)

Takeaway: autocorrelation inside a sequence can live in the emission model rather than being forced
into extra hidden states, and it is fit by the same ``optimize`` call. The script also shows the
scoring contract: ``seq_log_density`` on an encoded batch agrees with per-observation ``log_density``.
Runtime is ~20-25 s (1000 EM iterations with ``delta=None``, i.e. no early stop).
"""

import numpy as np

from mixle.inference import optimize
from mixle.stats import *
from mixle.stats import IntegerMarkovChainDistribution, IntegerMarkovChainEstimator
from mixle.stats.latent.lookback_hidden_markov_model import (
    LookbackHiddenMarkovModelDistribution,
    LookbackHiddenMarkovModelEstimator,
)

if __name__ == "__main__":
    # P(set_1): UNIFORM, so every state's per-position marginal is exactly uniform (see the
    # module docstring -- a non-uniform start leaks lag-0 signal, STAT-RR17-18)
    d0 = IntegerCategoricalDistribution(0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    # P(set_2 | set_1, Z=0)
    dist1 = IntegerMarkovChainDistribution(3, [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    # P(set_2 | set_1, Z=1)
    dist2 = IntegerMarkovChainDistribution(3, [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    # P(set_2 | set_1, Z=2)
    dist3 = IntegerMarkovChainDistribution(3, [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

    init_dists = [SequenceDistribution(d0, CategoricalDistribution({1: 1.0}))] * 3
    states = [dist1, dist2, dist3]
    len_dist = CategoricalDistribution({7: 0.5, 8: 0.25, 9: 0.25})
    transition = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    w = [0.4, 0.3, 0.3]

    dist = LookbackHiddenMarkovModelDistribution(
        states, w=w, transitions=transition, lag=1, init_dist=init_dists, len_dist=len_dist
    )

    data = dist.sampler(seed=1).sample(200)

    print(data[0])
    print(data[1])
    print(data[2])

    print(dist.seq_log_density(dist.seq_encode(data[:10])))
    print([dist.log_density(data[i]) for i in range(10)])

    est0 = SequenceEstimator(IntegerCategoricalEstimator(), len_estimator=CategoricalEstimator())
    est1 = IntegerMarkovChainEstimator(3)
    est = LookbackHiddenMarkovModelEstimator(
        [est1] * 3, lag=1, init_estimators=[est0] * 3, len_estimator=CategoricalEstimator()
    )

    model = optimize(data, est, max_its=1000, delta=None, rng=np.random.RandomState(1))

    print(str(model))
