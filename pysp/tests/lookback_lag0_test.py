"""Tests that lag=0 lookback hidden Markov models behave as consistent ordinary HMMs.

Covers both pysp.stats.lookback_hmm (the original module) and pysp.stats.look_back_hmm (the typed
rewrite). With lag=0 there is no initial segment: the number of hidden positions is len(x), the first
state is drawn from w and emits the window x[0:1], and a sampled length n yields exactly n emissions.
Checks log_density against a hand-computed ordinary-HMM forward pass and against
pysp.stats.hidden_markov.HiddenMarkovModelDistribution, that sampled sequence lengths match the
length distribution's draws exactly, and that EM (scalar and vectorized paths) runs and agrees.
"""

import unittest

import numpy as np
from numpy.random import RandomState

import pysp.stats.look_back_hmm as new_mod
import pysp.stats.lookback_hmm as old_mod
from pysp.arithmetic import maxrandint
from pysp.stats import seq_encode, seq_estimate, seq_initialize, seq_log_density_sum
from pysp.stats.categorical import CategoricalDistribution, CategoricalEstimator
from pysp.stats.hidden_markov import HiddenMarkovModelDistribution
from pysp.stats.int_range import IntegerCategoricalDistribution, IntegerCategoricalEstimator
from pysp.stats.null_dist import NullDistribution, NullEstimator
from pysp.stats.sequence import SequenceDistribution, SequenceEstimator

MODULES = [old_mod, new_mod]

W = [0.6, 0.4]
TRANSITIONS = [[0.8, 0.2], [0.3, 0.7]]
EMISSION_PROBS = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]
LEN_PROBS = {2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25}


class _MarginalWindowDistribution:
    """Topic over length-1 windows: scores window[-1] with a base distribution, history-free sampling."""

    def __init__(self, base):
        self.base = base

    def log_density(self, x):
        return self.base.log_density(x[-1])

    def sampler(self, seed=None):
        return _MarginalWindowSampler(self.base.sampler(seed=seed))


class _MarginalWindowSampler:
    def __init__(self, base_sampler):
        self.base_sampler = base_sampler

    def sample(self, size=None):
        return self.base_sampler.sample(size=size)

    def sample_given(self, x):
        return self.base_sampler.sample()


def make_lag0_dist(mod, len_dist):
    """Lag-0 lookback HMM with categorical emissions wrapped as length-1-window topics."""
    topics = [
        SequenceDistribution(IntegerCategoricalDistribution(0, p), len_dist=CategoricalDistribution({1: 1.0}))
        for p in EMISSION_PROBS
    ]
    init_dist = [NullDistribution()] * 2 if mod is old_mod else None

    return mod.LookbackHiddenMarkovDistribution(
        topics, w=W, transitions=TRANSITIONS, lag=0, init_dist=init_dist, len_dist=len_dist
    )


def make_lag0_estimator(mod, with_len=True):
    """Estimator matching make_lag0_dist(); init estimators are Null since lag=0 has no initial segment."""
    topic_est = SequenceEstimator(
        IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1),
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
    )
    if with_len:
        len_est = CategoricalEstimator(pseudo_count=0.1)
    else:
        len_est = None if mod is old_mod else NullEstimator()

    return mod.LookbackHiddenMarkovEstimator(
        [topic_est] * 2, lag=0, init_estimators=[NullEstimator()] * 2, len_estimator=len_est, pseudo_count=(1.0, 1.0)
    )


def forward_log_density(seq, w, transitions, emission_probs, len_probs):
    """Hand-computed ordinary-HMM forward log-likelihood with a categorical length term."""
    w = np.asarray(w, dtype=np.float64)
    a_mat = np.asarray(transitions, dtype=np.float64)
    e_mat = np.asarray(emission_probs, dtype=np.float64)

    if len(seq) == 0:
        return np.log(len_probs.get(0, 0.0))

    alpha = w * e_mat[:, seq[0]]
    ll = 0.0
    for v in seq[1:]:
        s = alpha.sum()
        ll += np.log(s)
        alpha = (a_mat.T @ (alpha / s)) * e_mat[:, v]
    ll += np.log(alpha.sum())

    return ll + np.log(len_probs[len(seq)])


class Lag0LogDensityTestCase(unittest.TestCase):
    def setUp(self):
        self.len_dist = CategoricalDistribution(dict(LEN_PROBS))
        self.sequences = [[0, 1], [2, 2, 0], [1, 0, 2, 1], [0, 0, 0, 0, 0], [2, 1, 0, 1, 2]]

    def test_log_density_matches_hand_forward(self):
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = make_lag0_dist(mod, self.len_dist)
                for seq in self.sequences:
                    expected = forward_log_density(seq, W, TRANSITIONS, EMISSION_PROBS, LEN_PROBS)
                    self.assertAlmostEqual(dist.log_density(seq), expected, places=10)

    def test_log_density_matches_ordinary_hmm(self):
        hmm = HiddenMarkovModelDistribution(
            [IntegerCategoricalDistribution(0, p) for p in EMISSION_PROBS],
            w=W,
            transitions=TRANSITIONS,
            len_dist=self.len_dist,
        )
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = make_lag0_dist(mod, self.len_dist)
                for seq in self.sequences:
                    self.assertAlmostEqual(dist.log_density(seq), hmm.log_density(seq), places=10)

    def test_seq_log_density_matches_log_density(self):
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = make_lag0_dist(mod, self.len_dist)
                enc = dist.seq_encode(self.sequences)
                ld_seq = dist.seq_log_density(enc)
                ld_item = np.asarray([dist.log_density(u) for u in self.sequences])
                self.assertTrue(np.all(np.isfinite(ld_item)))
                self.assertTrue(np.allclose(ld_seq, ld_item))

    def test_viterbi_sequence_has_one_state_per_observation(self):
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = make_lag0_dist(mod, self.len_dist)
                for seq in self.sequences:
                    states = dist.viterbi_sequence(seq)
                    self.assertEqual(len(states), len(seq))
                    self.assertTrue(all(0 <= s < 2 for s in states))


class Lag0SamplerTestCase(unittest.TestCase):
    def make_sampler_dist(self, mod, len_dist):
        topics = [_MarginalWindowDistribution(IntegerCategoricalDistribution(0, p)) for p in EMISSION_PROBS]
        init_dist = [NullDistribution()] * 2
        return mod.LookbackHiddenMarkovDistribution(
            topics, w=W, transitions=TRANSITIONS, lag=0, init_dist=init_dist, len_dist=len_dist
        )

    def test_deterministic_lengths_emit_exactly_n_observations(self):
        for mod in MODULES:
            for n in (0, 1, 5):
                with self.subTest(module=mod.__name__, n=n):
                    dist = self.make_sampler_dist(mod, CategoricalDistribution({n: 1.0}))
                    data = dist.sampler(seed=42).sample(25)
                    self.assertTrue(all(len(seq) == n for seq in data))

    def test_sampled_lengths_match_length_draws_exactly(self):
        len_dist = CategoricalDistribution({1: 0.3, 2: 0.4, 5: 0.3})
        n_samp = 200
        seed = 123

        # replicate the sampler's internal seed chain: 2 init samplers, 2 obs samplers, then len_sampler
        rng = RandomState(seed)
        for _ in range(4):
            rng.randint(0, maxrandint)
        len_sampler = len_dist.sampler(seed=rng.randint(0, maxrandint))
        expected_lens = [len_sampler.sample() for _ in range(n_samp)]

        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = self.make_sampler_dist(mod, len_dist)
                data = dist.sampler(seed=seed).sample(n_samp)
                self.assertEqual([len(seq) for seq in data], expected_lens)

    def test_sampled_values_are_in_emission_support(self):
        len_dist = CategoricalDistribution({3: 0.5, 4: 0.5})
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = self.make_sampler_dist(mod, len_dist)
                data = dist.sampler(seed=7).sample(50)
                for seq in data:
                    self.assertTrue(all(v in (0, 1, 2) for v in seq))


class Lag0EstimationTestCase(unittest.TestCase):
    @staticmethod
    def make_data(n_seq, rng):
        return [[int(v) for v in rng.randint(0, 3, size=rng.randint(2, 6))] for _ in range(n_seq)]

    def test_em_iteration_returns_finite_parameters(self):
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = make_lag0_dist(mod, CategoricalDistribution(dict(LEN_PROBS)))
                data = self.make_data(40, RandomState(11))
                est = make_lag0_estimator(mod)

                enc_data = seq_encode(data, model=dist)

                init_model = seq_initialize(enc_data, est, RandomState(1), p=1.0)
                self.assertIsInstance(init_model, mod.LookbackHiddenMarkovDistribution)

                model1 = seq_estimate(enc_data, est, init_model)
                model2 = seq_estimate(enc_data, est, model1)

                for model in (init_model, model1, model2):
                    self.assertTrue(np.all(np.isfinite(model.w)))
                    self.assertTrue(np.all(np.isfinite(model.transitions)))
                    self.assertAlmostEqual(model.w.sum(), 1.0, places=8)
                    self.assertTrue(np.allclose(model.transitions.sum(axis=1), 1.0))

                _, ll0 = seq_log_density_sum(enc_data, init_model)
                _, ll1 = seq_log_density_sum(enc_data, model1)
                _, ll2 = seq_log_density_sum(enc_data, model2)

                self.assertTrue(np.isfinite(ll0))
                self.assertTrue(np.isfinite(ll1))
                self.assertTrue(np.isfinite(ll2))
                # EM steps (lightly regularized) should not noticeably decrease the likelihood
                self.assertGreaterEqual(ll1, ll0 - 0.5)
                self.assertGreaterEqual(ll2, ll1 - 0.5)

    def test_update_matches_seq_update(self):
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                dist = make_lag0_dist(mod, CategoricalDistribution(dict(LEN_PROBS)))
                data = self.make_data(8, RandomState(5))
                est = make_lag0_estimator(mod, with_len=False)

                acc1 = est.accumulator_factory().make()
                for u in data:
                    acc1.update(u, 1.0, dist)

                acc2 = est.accumulator_factory().make()
                acc2.seq_update(dist.seq_encode(data), np.ones(len(data)), dist)

                v1, v2 = acc1.value(), acc2.value()
                self.assertTrue(np.allclose(v1[2], v2[2]))  # init_counts
                self.assertTrue(np.allclose(v1[3], v2[3]))  # state_counts
                self.assertTrue(np.allclose(v1[4], v2[4]))  # trans_counts

                # the total state mass is one unit per observed position (ordinary HMM, no init block)
                tot_pos = float(sum(len(u) for u in data))
                self.assertAlmostEqual(float(np.sum(v1[3])), tot_pos, places=8)
                self.assertAlmostEqual(float(np.sum(v1[2])), float(len(data)), places=8)

    def test_initialize_counts_one_position_per_observation(self):
        for mod in MODULES:
            with self.subTest(module=mod.__name__):
                data = self.make_data(10, RandomState(3))
                est = make_lag0_estimator(mod, with_len=False)

                acc_s = est.accumulator_factory().make()
                for u in data:
                    acc_s.initialize(u, 1.0, RandomState(2))

                acc_v = est.accumulator_factory().make()
                enc = acc_v.acc_to_encoder().seq_encode(data)
                acc_v.seq_initialize(enc, np.ones(len(data)), RandomState(2))

                tot_pos = float(sum(len(u) for u in data))
                for acc in (acc_s, acc_v):
                    v = acc.value()
                    self.assertAlmostEqual(float(np.sum(v[3])), tot_pos, places=8)  # state_counts
                    self.assertAlmostEqual(float(np.sum(v[2])), float(len(data)), places=8)  # init_counts
                    self.assertAlmostEqual(float(np.sum(v[4])), tot_pos - len(data), places=8)  # trans_counts


if __name__ == "__main__":
    unittest.main()
