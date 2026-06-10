"""Fit an HMM to generated text-like sequences, comparing fitting with and
without Numba acceleration.

Data is sampled from a known HMM (10 hidden states emitting integer "words"
from a 500-word vocabulary), so the example is fully self-contained.
"""
import time

import numpy as np

from pysp.stats import *
from pysp.utils.estimation import optimize


def make_data(num_states=10, num_words=500, num_seqs=100, seq_len=100, seed=2):
    rng = np.random.RandomState(seed)

    topics = [IntegerCategoricalDistribution(0, list(rng.dirichlet(np.ones(num_words) * 0.1)))
              for _ in range(num_states)]
    transitions = rng.dirichlet(np.ones(num_states) * 0.5, size=num_states)
    w = rng.dirichlet(np.ones(num_states))
    len_dist = CategoricalDistribution({seq_len: 1.0})

    truth = HiddenMarkovModelDistribution(topics=topics, w=list(w), transitions=transitions,
                                          len_dist=len_dist)
    return truth.sampler(seed=seed).sample(size=num_seqs), num_words


if __name__ == '__main__':
    chunks, num_words = make_data()

    est = IntegerCategoricalEstimator(min_val=0, max_val=num_words - 1, pseudo_count=1.0)
    est = HiddenMarkovEstimator([est] * 10, use_numba=False)
    imodel = optimize(chunks, est, max_its=1, rng=np.random.RandomState(1), init_p=1.0)

    t00 = time.time()
    model = optimize(chunks, est, max_its=200, prev_estimate=imodel, print_iter=200)
    t01 = time.time()
    print(t01 - t00)

    est = IntegerCategoricalEstimator(min_val=0, max_val=num_words - 1, pseudo_count=1.0)
    est = HiddenMarkovEstimator([est] * 10, use_numba=True)
    imodel = optimize(chunks, est, max_its=1, rng=np.random.RandomState(1), init_p=1.0)

    t10 = time.time()
    model = optimize(chunks, est, max_its=200, prev_estimate=imodel, print_iter=200)
    t11 = time.time()
    print(t11 - t10)

    print('Speedup = %f' % ((t01 - t00) / (t11 - t10)))
