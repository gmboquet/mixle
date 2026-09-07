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

The emissions here are Markov chains over a discrete alphabet, so nothing in the observation space
can be clustered to break the initial symmetry between the three states: EM starts from a random
soft assignment and its optimum depends on the draw. Rather than assert a recovery this fit does not
always achieve, the script RESTARTS (``best_of``) and then PRINTS what it got -- the held-out mean
log-density of the fit beside the generating model's on the same sequences -- so a run that landed on
a poor optimum is visible in the output instead of being hidden behind a printed model (P10-F07).
Runtime is ~20-25 s (1000 EM iterations with ``delta=None``, i.e. no early stop).
"""

import numpy as np

from mixle.inference import best_of
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

    # best_of scores each restart on a held-out split, so a degenerate optimum loses to a good one
    # rather than being whatever the single draw happened to reach.
    valid = dist.sampler(seed=2).sample(200)
    # delta=None: the budget is deliberate and shared by every restart, so they stay comparable.
    _score, model = best_of(data, valid, est, 4, 400, 1.0, None, np.random.RandomState(1))

    print(str(model))

    held_out = dist.sampler(seed=3).sample(500)
    fit_ll = float(np.mean(model.seq_log_density(model.seq_encode(held_out))))
    true_ll = float(np.mean(dist.seq_log_density(dist.seq_encode(held_out))))
    print(
        "held-out mean log-density: fit %.3f vs generating model %.3f (gap %.3f nats/sequence)"
        % (fit_ll, true_ll, true_ll - fit_ll)
    )
