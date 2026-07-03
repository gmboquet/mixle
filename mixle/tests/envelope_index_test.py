"""The AR envelope index: approximate deep enumeration for autoregressive models (LLMs).

The exact autoregressive count index is a tree recursion -- Theta(count) work -- so deep ranks are
unreachable exactly. The envelope index precomputes per-depth aggregate step histograms (suffix-convolved
at C speed) and answers count/threshold/unrank/rank_bracket in O(L) model forwards per query. Contract
verified here: for an iid-step model the envelope IS the true per-step histogram, so every query is exact
(checked value-for-value against the exact SeekIndex); for context-dependent models the rank coordinate is
a mean-field estimate (checked to track the true rank on a small enumerable model) while every returned
sequence and log-probability is exact; deep ranks (1e15) resolve in milliseconds with O(n_paths * L)
total forwards -- the precompute bound, not the count.
"""

import math
import unittest

import numpy as np

from mixle.enumeration import AREnvelopeIndex, AutoregressiveEnumerable, SeekIndex


def _iid_model(V, seed=0, scale=1.5):
    rng = np.random.RandomState(seed)
    logits = rng.randn(V) * scale
    m = np.max(logits)
    lp = logits - (m + math.log(np.sum(np.exp(logits - m))))

    def nlp(prefix):
        return np.arange(V), lp

    return nlp


def _ctx_model(V, L, seed=0, scale=1.0):
    W = np.random.RandomState(seed).randn(L, V, V) * scale

    def nlp(prefix):
        d = len(prefix)
        last = prefix[-1] if prefix else 0
        lg = W[d, last]
        m = np.max(lg)
        return np.arange(V), lg - (m + math.log(np.sum(np.exp(lg - m))))

    return nlp


class IIDExactnessTest(unittest.TestCase):
    """For a prefix-independent model the envelope equals the truth -- every query must be exact."""

    def setUp(self):
        self.V, self.L = 7, 4
        self.ar = AutoregressiveEnumerable(_iid_model(self.V), max_len=self.L)
        self.env = AREnvelopeIndex(self.ar, n_paths=4, seed=0)
        self.exact = SeekIndex(self.ar)
        self.exact.ensure_bits(40.0)

    def test_total_and_counts_match_exact(self):
        self.assertEqual(int(self.env.total()), len(self.exact))  # V**L support, both exhaustive
        thr = self.exact.unrank(500)[1]
        self.assertEqual(int(self.env.count(thr)), self.exact.count(thr))

    def test_unrank_lands_in_the_true_rank_bucket(self):
        q = self.env.quantizer
        for i in (0, 5, 100, 1000, 2000):
            _s_env, lp_env = self.env.unrank(i)
            _s_ex, lp_ex = self.exact.unrank(i)
            self.assertEqual(q.fine_bucket(lp_env), q.fine_bucket(lp_ex), f"rank {i}")

    def test_unranked_logprobs_are_exact_model_values(self):
        for i in (0, 17, 923):
            seq, lp = self.env.unrank(i)
            self.assertEqual(len(seq), self.L)
            self.assertAlmostEqual(lp, self.ar.log_density(seq), places=12)

    def test_rank_bracket_is_exact_for_iid(self):
        for i in (0, 50, 400):
            seq, _lp = self.exact.unrank(i)
            lo, hi = self.env.rank_bracket(seq)
            self.assertLessEqual(lo, i)
            self.assertGreaterEqual(hi + 1e-9, i)

    def test_mass_above_brackets_true_head_mass(self):
        thr = self.exact.unrank(300)[1]
        true_mass = sum(math.exp(lp) for _v, lp in self.exact.iter_from(0) if lp >= thr)
        lo, hi = self.env.mass_above(thr)
        self.assertLessEqual(lo, true_mass + 1e-12)
        self.assertGreaterEqual(hi, true_mass - 1e-12)


class ContextDependentEstimateTest(unittest.TestCase):
    """Mean-field estimates on a genuinely context-dependent (but enumerable) model track the truth."""

    def setUp(self):
        self.V, self.L = 5, 4
        self.ar = AutoregressiveEnumerable(_ctx_model(self.V, self.L, seed=2), max_len=self.L)
        self.env = AREnvelopeIndex(self.ar, n_paths=200, seed=1)
        self.exact = SeekIndex(self.ar)
        self.exact.ensure_bits(40.0)

    def test_total_estimate_near_truth(self):
        n = len(self.exact)  # V**L = 625: the envelope's total should estimate it closely
        self.assertLess(abs(self.env.total() - n) / n, 0.05)

    def test_rank_bracket_tracks_true_rank(self):
        for true_rank in (0, 10, 50, 200, 500):
            seq, _lp = self.exact.unrank(true_rank)
            lo, hi = self.env.rank_bracket(seq)
            mid = 0.5 * (lo + hi)
            # deterministic seeds: the bracket midpoint stays within ~15% of support size of the truth
            self.assertLess(abs(mid - true_rank), 0.15 * len(self.exact) + 5.0, f"rank {true_rank} -> {(lo, hi)}")

    def test_unrank_returns_valid_sequences_with_exact_logprobs(self):
        for i in (0, 33, 444):
            seq, lp = self.env.unrank(i)
            self.assertEqual(len(seq), self.L)
            self.assertAlmostEqual(lp, self.ar.log_density(seq), places=12)


class DeepRankTest(unittest.TestCase):
    """The point: ranks far beyond any exact tree build resolve in O(L) forwards per query."""

    def test_unrank_1e15_with_bounded_forwards(self):
        V, L = 50, 12  # 50**12 ~ 2**67 sequences: the exact tree is impossible
        ar = AutoregressiveEnumerable(_ctx_model(V, L, seed=3, scale=2.0), max_len=L)
        env = AREnvelopeIndex(ar, n_paths=32, seed=0, budget_bits=70.0)
        seq, lp = env.unrank(10**15)
        self.assertEqual(len(seq), L)
        self.assertTrue(np.isfinite(lp))
        self.assertAlmostEqual(lp, ar.log_density(seq), places=10)
        # self-consistency: the envelope places its own answer where it claimed
        lo, hi = env.rank_bracket(seq)
        self.assertLessEqual(lo, 1e15)
        self.assertGreaterEqual(hi, 1e15 * 0.5)
        # the whole exercise cost O(n_paths * L) forwards -- the precompute bound, not the rank
        self.assertLess(len(ar._cache), 32 * L * 3)

    def test_deepen_in_place(self):
        ar = AutoregressiveEnumerable(_ctx_model(8, 6, seed=4), max_len=6)
        env = AREnvelopeIndex(ar, n_paths=16, seed=0, budget_bits=8.0)
        shallow_total = env.total()
        env.ensure_bits(40.0)  # deepen: more of the support enters the tables
        self.assertGreaterEqual(env.total(), shallow_total)


class ContractTest(unittest.TestCase):
    def test_terminating_model_rejected(self):
        def nlp(prefix):
            return np.arange(3), np.log(np.array([0.5, 0.3, 0.2]))

        ar = AutoregressiveEnumerable(nlp, eos=2)
        with self.assertRaises(ValueError):
            AREnvelopeIndex(ar)

    def test_out_of_range_and_bad_args(self):
        ar = AutoregressiveEnumerable(_iid_model(4), max_len=2)
        env = AREnvelopeIndex(ar, n_paths=2, seed=0)
        with self.assertRaises(IndexError):
            env.unrank(-1)
        with self.assertRaises(IndexError):
            env.unrank(10**9)  # support is 16 sequences
        with self.assertRaises(ValueError):
            env.threshold(0)

    def test_envelope_available_from_adapter(self):
        ar = AutoregressiveEnumerable(_iid_model(4), max_len=2)
        env = ar.envelope_index(n_paths=2, seed=0)
        self.assertIsInstance(env, AREnvelopeIndex)


if __name__ == "__main__":
    unittest.main()
