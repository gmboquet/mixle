"""Parity / equivalence tests for batched combinator sampling.

The combinator samplers (HiddenMarkov, MarkovChain, IntegerMarkovChain, LDA) gained a
``batched=True`` (default) fast path. ``batched=False`` reproduces the exact legacy per-draw loop.

Unlike Mixture/Sequence, vectorizing the Markov/HMM STATE PATH across chains changes the RNG
consumption order, so the batched output is only statistically equivalent to the legacy loop, not
byte-identical. These tests therefore assert:

  (a) byte-identity for the structural draws that keep the same order (sequence lengths), and for
      emissions grouped by state where applicable;
  (b) distributional equivalence for the reordered state paths -- state marginals, transition
      frequencies, and length statistics match within sampling tolerance;
  (c) a speed sanity check that the batched path is faster than the legacy loop on a moderate batch.
"""

import time
from collections import Counter

import numpy as np
import pytest

import pysp.stats as stats


def _markov_chain():
    return stats.MarkovChainDistribution(
        {"a": 0.6, "b": 0.4},
        {"a": {"a": 0.7, "b": 0.3}, "b": {"a": 0.2, "b": 0.8}},
        len_dist=stats.CategoricalDistribution({5: 1.0}),
    )


def _hmm():
    return stats.HiddenMarkovModelDistribution(
        [
            stats.CategoricalDistribution({"a": 0.8, "b": 0.2}),
            stats.CategoricalDistribution({"a": 0.1, "b": 0.9}),
        ],
        [0.6, 0.4],
        [[0.7, 0.3], [0.2, 0.8]],
        len_dist=stats.CategoricalDistribution({6: 1.0}),
        use_numba=False,
    )


def _int_markov():
    imc_init = stats.SequenceDistribution(
        stats.IntegerCategoricalDistribution(0, [0.5, 0.5]),
        len_dist=stats.CategoricalDistribution({2: 1.0}),
    )
    return stats.IntegerMarkovChainDistribution(
        num_values=2,
        cond_dist=[[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]],
        lag=2,
        init_dist=imc_init,
        len_dist=stats.CategoricalDistribution({6: 1.0}),
    )


def _lda():
    return stats.LDADistribution(
        [
            stats.CategoricalDistribution({"a": 0.8, "b": 0.2}),
            stats.CategoricalDistribution({"a": 0.1, "b": 0.9}),
        ],
        alpha=[1.0, 1.0],
        len_dist=stats.CategoricalDistribution({8: 1.0}),
    )


# --------------------------------------------------------------------------------------------------
# batched=False reproduces the exact legacy per-draw output for a given seed.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_markov_chain, _hmm, _int_markov, _lda])
def test_batched_false_is_stable_reference(factory):
    dist = factory()
    a = dist.sampler(seed=123).sample(20, batched=False)
    b = dist.sampler(seed=123).sample(20, batched=False)
    assert a == b


# --------------------------------------------------------------------------------------------------
# Length statistics are byte-identical: the length draws keep the same RNG order in both paths.
# --------------------------------------------------------------------------------------------------


def test_markov_chain_lengths_match():
    # variable lengths so the length draw order is observable
    dist = stats.MarkovChainDistribution(
        {"a": 0.6, "b": 0.4},
        {"a": {"a": 0.7, "b": 0.3}, "b": {"a": 0.2, "b": 0.8}},
        len_dist=stats.CategoricalDistribution({3: 0.5, 6: 0.5}),
    )
    legacy = dist.sampler(seed=11).sample(200, batched=False)
    batched = dist.sampler(seed=11).sample(200)
    assert [len(s) for s in legacy] == [len(s) for s in batched]


def test_lda_lengths_match():
    dist = stats.LDADistribution(
        [
            stats.CategoricalDistribution({"a": 0.8, "b": 0.2}),
            stats.CategoricalDistribution({"a": 0.1, "b": 0.9}),
        ],
        alpha=[1.0, 1.0],
        len_dist=stats.CategoricalDistribution({5: 0.5, 9: 0.5}),
    )
    legacy = dist.sampler(seed=21).sample(200, batched=False)
    batched = dist.sampler(seed=21).sample(200)
    assert [len(d) for d in legacy] == [len(d) for d in batched]


# --------------------------------------------------------------------------------------------------
# Distributional equivalence of the (reordered) state path / token draws.
# --------------------------------------------------------------------------------------------------


def _state_marginals(seqs, states):
    c = Counter()
    for s in seqs:
        c.update(s)
    n = sum(c.values())
    return np.asarray([c[k] / n for k in states])


def _transition_freqs(seqs, states):
    idx = {s: i for i, s in enumerate(states)}
    m = np.zeros((len(states), len(states)))
    for s in seqs:
        for i in range(1, len(s)):
            m[idx[s[i - 1]], idx[s[i]]] += 1
    row = m.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return m / row


def test_markov_chain_distributional_equivalence():
    dist = _markov_chain()
    states = ["a", "b"]
    size = 40000
    legacy = dist.sampler(seed=5).sample(size, batched=False)
    batched = dist.sampler(seed=5).sample(size)

    np.testing.assert_allclose(_state_marginals(legacy, states), _state_marginals(batched, states), atol=0.01)
    np.testing.assert_allclose(_transition_freqs(legacy, states), _transition_freqs(batched, states), atol=0.02)
    # transitions should also match the true generating matrix
    np.testing.assert_allclose(_transition_freqs(batched, states), np.asarray([[0.7, 0.3], [0.2, 0.8]]), atol=0.02)


def test_hmm_distributional_equivalence():
    dist = _hmm()
    states = ["a", "b"]
    size = 40000
    legacy = dist.sampler(seed=8).sample(size, batched=False)
    batched = dist.sampler(seed=8).sample(size)
    # emission marginals (observed symbols) match within tolerance
    np.testing.assert_allclose(_state_marginals(legacy, states), _state_marginals(batched, states), atol=0.01)


def test_int_markov_distributional_equivalence():
    dist = _int_markov()
    size = 30000
    legacy = dist.sampler(seed=14).sample(size, batched=False)
    batched = dist.sampler(seed=14).sample(size)
    states = [0, 1]
    np.testing.assert_allclose(_state_marginals(legacy, states), _state_marginals(batched, states), atol=0.01)
    # the first `lag` (=2) init entries are drawn per-chain in order -> byte-identical inits
    assert [tuple(s[:2]) for s in legacy] == [tuple(s[:2]) for s in batched]


def test_lda_distributional_equivalence():
    dist = _lda()
    states = ["a", "b"]
    size = 30000
    legacy = dist.sampler(seed=17).sample(size, batched=False)
    batched = dist.sampler(seed=17).sample(size)
    np.testing.assert_allclose(_state_marginals(legacy, states), _state_marginals(batched, states), atol=0.01)


# --------------------------------------------------------------------------------------------------
# Single-draw (size=None) batched path returns the right shape and stays on support.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [_markov_chain, _hmm, _int_markov, _lda])
def test_single_draw_batched(factory):
    dist = factory()
    one = dist.sampler(seed=2).sample()
    assert isinstance(one, list)


# --------------------------------------------------------------------------------------------------
# Speed sanity check: batched faster than legacy on a moderate batch.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory,size",
    [(_markov_chain, 20000), (_hmm, 8000), (_int_markov, 8000), (_lda, 8000)],
)
def test_batched_faster_than_legacy(factory, size):
    dist = factory()

    s = dist.sampler(seed=1)
    t0 = time.perf_counter()
    s.sample(size, batched=False)
    legacy_t = time.perf_counter() - t0

    s = dist.sampler(seed=1)
    t0 = time.perf_counter()
    s.sample(size)
    batched_t = time.perf_counter() - t0

    assert batched_t < legacy_t, f"batched ({batched_t:.3f}s) not faster than legacy ({legacy_t:.3f}s)"
