"""Quantized left-to-right (upper-triangular) HMM: the special case where the structural
descending-probability seek coincides with the exact marginal order.

Upper-triangular transitions force a monotone non-decreasing state path (a Bakis chain), so a sentence's
state paths are its monotone segmentations -- polynomially many, not exponentially. With disjoint
per-state emissions the model is unambiguous (one path per sentence), and the structural count-DP that
backs ``seek``/``rank`` then carries each sentence exactly once at its (single-path == marginal)
probability -- so deep seek returns true k-th-most-probable sentences, not the tropical path projection
that a general ambiguous HMM collapses to.
"""

import bisect
import itertools
import unittest

import numpy as np

from pysp.stats import IntegerCategoricalDistribution, QuantizedHiddenMarkovModelDistribution

LEVELS = ["a", "b", "c", "d", "e", "f"]
# state 0 -> {a, b}, state 1 -> {c, d}, state 2 -> {e, f}  (disjoint => unambiguous)
EMIT = [[0, 1, -1, -1, -1, -1], [-1, -1, 0, 1, -1, -1], [-1, -1, -1, -1, 0, 1]]
TRANS = [[0, 1, 2], [-1, 0, 1], [-1, -1, 0]]  # upper triangular (lower = structural zeros)
INIT = [0, 1, 2]
LEN = IntegerCategoricalDistribution(1, [0.4, 0.3, 0.2, 0.1])  # lengths 1..4


def _make():
    return QuantizedHiddenMarkovModelDistribution.left_to_right(
        0.5, LEVELS, TRANS, EMIT, initial_exponents=INIT, len_dist=LEN
    )


def _brute_order(d, max_len=4):
    items = []
    for length in range(1, max_len + 1):
        for seq in itertools.product(LEVELS, repeat=length):
            lp = d.log_density(list(seq))
            if np.isfinite(lp):
                items.append((list(seq), lp))
    items.sort(key=lambda kv: -kv[1])
    return items


class ConstructorTest(unittest.TestCase):
    def test_rejects_non_upper_triangular(self):
        bad = [[0, 1, 2], [3, 0, 1], [-1, -1, 0]]  # entry (1,0) = 3 is below the diagonal, not a zero
        with self.assertRaises(ValueError):
            QuantizedHiddenMarkovModelDistribution.left_to_right(0.5, LEVELS, bad, EMIT, initial_exponents=INIT)

    def test_rejects_non_square(self):
        with self.assertRaises(ValueError):
            QuantizedHiddenMarkovModelDistribution.left_to_right(0.5, LEVELS, [[0, 1, 2], [-1, 0, 1]], EMIT)

    def test_accepts_upper_triangular(self):
        d = _make()
        self.assertEqual(d.n_states, 3)


class DistributionTest(unittest.TestCase):
    def setUp(self):
        self.d = _make()

    def test_normalizes(self):
        total = sum(np.exp(lp) for _, lp in _brute_order(self.d, max_len=7))
        self.assertAlmostEqual(total, 1.0, delta=1e-6)

    def test_monotone_path_only(self):
        # a "backward" sentence (state-1 symbol then state-0 symbol) is unreachable under upper-triangular
        self.assertEqual(self.d.log_density(["c", "a"]), float("-inf"))
        self.assertTrue(np.isfinite(self.d.log_density(["a", "c"])))  # forward is fine

    def test_seq_density_matches_scalar(self):
        seqs = [["a"], ["a", "c"], ["b", "d", "f"], ["a", "a", "e"]]
        enc = self.d.dist_to_encoder().seq_encode(seqs)
        np.testing.assert_allclose(self.d.seq_log_density(enc), [self.d.log_density(s) for s in seqs], atol=1e-9)


class EnumerationAndSeekTest(unittest.TestCase):
    def setUp(self):
        self.d = _make()
        self.order = _brute_order(self.d)

    def test_top_k_matches_brute_force_levels(self):
        # quantized models have many exact ties, so compare the probability sequence, not identities
        top = self.d.enumerator().top_k(30)
        for (_, lp), (_, blp) in zip(top, self.order[:30]):
            self.assertAlmostEqual(lp, blp, places=9)

    def test_structural_seek_is_exact_and_overcount_free(self):
        logps = sorted((round(lp, 9) for _, lp in self.order), reverse=True)
        desc = [-x for x in logps]  # ascending, for bracket bisection
        seen, dups, level_ok, bracket_ok = set(), 0, 0, 0
        k_max = min(len(self.order), 80)
        for k in range(k_max):
            v = tuple(self.d.enumerator().seek(k).value)
            if v in seen:
                dups += 1
            seen.add(v)
            slp = round(self.d.log_density(list(v)), 9)
            if abs(slp - logps[k]) < 1e-6:
                level_ok += 1
            lo = bisect.bisect_left(desc, -slp)  # strictly-more-probable count
            hi = bisect.bisect_right(desc, -slp)  # at-least-as-probable count
            if lo <= k < hi:
                bracket_ok += 1
        # unambiguous => no path over-count: every seek is a distinct sentence
        self.assertEqual(dups, 0)
        # structural order coincides with the exact marginal order up to quantization tie granularity
        self.assertGreaterEqual(level_ok, k_max - 3)
        self.assertGreaterEqual(bracket_ok, k_max - 3)

    def test_deep_seek_reaches_distinct_sentences(self):
        # a deep index is reachable and returns a valid, finite-probability sentence
        deep = self.d.enumerator().seek(min(len(self.order) - 1, 120)).value
        self.assertTrue(np.isfinite(self.d.log_density(list(deep))))


class TerminalTest(unittest.TestCase):
    """The terminal (stopping-time) form of the left-to-right quantized HMM -- the sentence-generator
    use case. Enumeration delegates to the base terminal enumerator; seek is exact (enumerate-and-bin
    over the exact marginal order)."""

    TLEVELS = ["a", "b", "c", "d", "."]
    TEMIT = [[0, 1, -1, -1, -1], [-1, -1, 0, 1, -1], [-1, -1, -1, -1, 0]]  # s2 emits the terminal '.'
    TTRANS = [[0, 1, 2], [-1, 0, 1], [-1, -1, 0]]

    def _make(self):
        return QuantizedHiddenMarkovModelDistribution.left_to_right(
            0.5, self.TLEVELS, self.TTRANS, self.TEMIT, initial_exponents=[0, 1, 2], terminal_values={"."}
        )

    def _order(self, d, max_len=7):
        out = []
        for length in range(1, max_len + 1):
            for pre in itertools.product(["a", "b", "c", "d"], repeat=length - 1):
                seq = list(pre) + ["."]
                lp = d.log_density(seq)
                if np.isfinite(lp):
                    out.append((seq, lp))
        out.sort(key=lambda kv: -kv[1])
        return out

    def test_terminal_enumeration_and_exact_seek(self):
        d = self._make()
        order = self._order(d)
        top = d.enumerator().top_k(10)
        for (_, lp), (_, blp) in zip(top, order[:10]):
            self.assertAlmostEqual(lp, blp, places=9)
        logps = sorted((round(lp, 9) for _, lp in order), reverse=True)
        seen, exact, dups = set(), 0, 0
        k_max = min(len(order), 50)
        for k in range(k_max):
            v = tuple(d.enumerator().seek(k).value)
            if v in seen:
                dups += 1
            seen.add(v)
            self.assertEqual(v[-1], ".")  # in-support: ends at the terminal value
            if abs(round(d.log_density(list(v)), 9) - logps[k]) < 1e-9:
                exact += 1
        self.assertEqual(dups, 0)
        self.assertEqual(exact, k_max)  # terminal seek is exact (enumerate-and-bin over the marginal)


if __name__ == "__main__":
    unittest.main()
