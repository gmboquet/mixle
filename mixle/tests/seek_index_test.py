"""The persistent SeekIndex + the float64 approximate count carrier (quantized counting).

SeekIndex is the precompute-once/query-many structure over the count-budget enumeration: repeated
unrank/count/threshold queries reuse one structural DP build (observable via ``builds``), deepening in
place only when a query outruns the built depth. ``count_mode='float'`` swaps exact big-integer counts
for float64 -- identical results below 2**53 (verified value-for-value against the exact carrier on
mixtures, HMMs, and sequences), C-speed numpy convolutions, and documented ~1e-16/op relative error
beyond. The autoregressive adapter routes its convenience surface through a cached SeekIndex and keeps
numpy speed past the int64-safe budget by carrying float64 counts.
"""

import math
import unittest

import numpy as np

from mixle.enumeration import AutoregressiveEnumerable, SeekIndex
from mixle.enumeration.quantization.core import Quantizer, count_budget_index
from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    HiddenMarkovModelDistribution,
    MixtureDistribution,
    PoissonDistribution,
    SequenceDistribution,
)


def _hmm(seed=0, n_sym=8):
    topics = [
        CategoricalDistribution({chr(97 + i): p for i, p in enumerate(np.random.RandomState(s).dirichlet([1] * n_sym))})
        for s in range(3)
    ]
    return HiddenMarkovModelDistribution(
        topics=topics,
        w=[0.5, 0.3, 0.2],
        transitions=[[0.7, 0.2, 0.1], [0.15, 0.7, 0.15], [0.1, 0.3, 0.6]],
        len_dist=PoissonDistribution(5.0),
    )


def _mixture(seed=0):
    comps = [
        CompositeDistribution(
            [
                CategoricalDistribution(
                    {chr(97 + i): p for i, p in enumerate(np.random.RandomState(s * 5 + f).dirichlet([1] * 6))}
                )
                for f in range(3)
            ]
        )
        for s in range(3)
    ]
    return MixtureDistribution(comps, [0.5, 0.3, 0.2])


def _seq(seed=0):
    cat = CategoricalDistribution(
        {chr(97 + i): p for i, p in enumerate(np.random.RandomState(seed).dirichlet([1] * 6))}
    )
    return SequenceDistribution(cat, len_dist=PoissonDistribution(4.0))


class FloatCountModeTest(unittest.TestCase):
    """count_mode='float' is value-for-value identical to exact while counts stay below 2**53."""

    def test_parity_on_mixture_hmm_sequence(self):
        for dist in (_mixture(), _hmm(), _seq()):
            ex = count_budget_index(dist, budget_bits=16.0, count_mode="exact")
            fl = count_budget_index(dist, budget_bits=16.0, count_mode="float")
            self.assertEqual(ex.total_count, fl.total_count, type(dist).__name__)
            probes = [0, 7, ex.total_count // 2, ex.total_count - 1]
            self.assertEqual([ex.get(i) for i in probes], [fl.get(i) for i in probes], type(dist).__name__)

    def test_float_convolve_matches_exact(self):
        from mixle.enumeration.quantization.core import CountHistogram

        a = CountHistogram(2, [1, 0, 3, 5, 2])
        b = CountHistogram(1, [4, 1, 2])
        exact = a.convolve(b)
        approx = a.convolve_float(b)
        self.assertEqual(exact.base, approx.base)
        self.assertEqual([float(c) for c in exact.data], [float(c) for c in approx.data])
        # capped variant agrees too
        self.assertEqual(
            [float(c) for c in a.convolve(b, max_fine_bucket=6).data],
            [float(c) for c in a.convolve_float(b, max_fine_bucket=6).data],
        )

    def test_float_counts_survive_past_2_53(self):
        # a self-convolution power chain whose counts blow far past 2**53: float mode keeps
        # a finite, relatively-accurate total (exact carrier is the ground truth)
        from mixle.enumeration.quantization.core import CountHistogram, Quantizer

        q_exact = Quantizer(count_mode="exact")
        q_float = Quantizer(count_mode="float")
        h = CountHistogram(1, [1000] * 40)
        he, hf = h, h
        for _ in range(6):  # totals reach (40 * 1000)**6 ~ 2**92
            he = q_exact.convolve(he, h, max_fine_bucket=300)
            hf = q_float.convolve(hf, h, max_fine_bucket=300)
        rel = abs(hf.total() - float(he.total())) / float(he.total())
        self.assertLess(rel, 1e-9)

    def test_quantizer_validates_mode(self):
        with self.assertRaises(ValueError):
            Quantizer(count_mode="bogus")


class SeekIndexTest(unittest.TestCase):
    """Build once, query many; deepen in place; results match the one-shot driver."""

    def test_matches_one_shot_driver(self):
        dist = _mixture()
        one_shot = count_budget_index(dist, budget_bits=14.0)
        si = SeekIndex(dist)
        si.ensure_bits(14.0)
        for i in (0, 3, 50, one_shot.total_count - 1):
            self.assertEqual(si.unrank(i), one_shot.get(i))

    def test_repeated_queries_reuse_one_build(self):
        si = SeekIndex(_hmm())
        si.unrank(50_000)  # deep enough for everything below
        builds = si.builds
        for i in range(200, 300):
            si.unrank(i)
        si.threshold(100)
        si.slice(10, 25)
        self.assertEqual(si.builds, builds)  # every query reused the one built DP

    def test_deepens_in_place_and_extends(self):
        si = SeekIndex(_hmm())
        v_small = si.unrank(10)
        shallow_builds = si.builds
        si.unrank(200_000)  # forces at least one deepen
        self.assertGreater(si.builds, shallow_builds)
        self.assertEqual(si.unrank(10), v_small)  # shallow answers unchanged by deepening

    def test_count_is_smear_bracketed_by_brute_truth(self):
        # The structural fine bucket is a SUM of per-term floors, so it under-estimates a value's bits by
        # at most #terms/oversample -- count(thr) therefore includes every true qualifier (lower bound) and
        # at most the values within that smear band below thr (upper bound). Verify both sides.
        dist = _seq()
        si = SeekIndex(dist)
        head = [lp for _v, lp in dist.enumerator().top_k(3000)]
        thr = head[150]
        n = si.count(thr)
        brute = sum(1 for lp in head if lp >= thr)
        smear_nats = 2.5 * math.log(2.0)  # ~(max terms)/oversample bits of accumulated floor rounding
        brute_hi = sum(1 for lp in head if lp >= thr - smear_nats)
        self.assertLess(brute_hi, len(head), "head too small to bound the smear band")
        self.assertGreaterEqual(n, brute)
        self.assertLessEqual(n, brute_hi)

    def test_rank_bracket_spans_the_value(self):
        # The bracket is the value's whole structural bucket, so unranking [lo, hi] must surface the value
        # -- the bracket's contract, valid for every family (tropical semantics included).
        for dist in (_mixture(), _seq()):
            si = SeekIndex(dist)
            for true_rank in (0, 5, 25):
                value, _lp = list(dist.enumerator().top_k(30))[true_rank]
                lo, hi = si.rank_bracket(value)
                self.assertLessEqual(lo, hi)
                if hi - lo <= 500:
                    found = any(si.unrank(i)[0] == value for i in range(lo, hi + 1))
                    self.assertTrue(found, f"{type(dist).__name__} rank {true_rank} not inside its bracket")

    def test_out_of_range_raises(self):
        cat = CategoricalDistribution({"a": 0.6, "b": 0.4})
        si = SeekIndex(cat)
        si.ensure_bits(10.0)  # deep enough to exhaust the 2-value support
        self.assertFalse(si.truncated)
        self.assertEqual(len(si), 2)
        with self.assertRaises(IndexError):
            si.unrank(2)
        with self.assertRaises(IndexError):
            si.unrank(-1)

    def test_float_mode_end_to_end(self):
        si = SeekIndex(_hmm(), count_mode="float")
        sx = SeekIndex(_hmm(), count_mode="exact")
        for i in (0, 10, 1000):
            self.assertEqual(si.unrank(i), sx.unrank(i))


class AutoregressiveSeekTest(unittest.TestCase):
    """The AR adapter's convenience surface runs on one cached SeekIndex; float64 covers deep budgets."""

    @staticmethod
    def _model(V, L, seed=0):
        W = np.random.RandomState(seed).randn(L, V, V)

        def nlp(prefix):
            d = len(prefix)
            last = prefix[-1] if prefix else 0
            lg = W[d, last]
            m = np.max(lg)
            return np.arange(V), lg - (m + math.log(np.sum(np.exp(lg - m))))

        return nlp

    def test_convenience_queries_share_one_index(self):
        ar = AutoregressiveEnumerable(self._model(6, 4), max_len=4)
        seq, lp = ar.unrank(500)
        builds = ar.seek_index().builds
        for i in (0, 100, 499):
            ar.unrank(i)
        ar.count(lp)
        ar.threshold(50)
        ar.mass_above(lp)
        self.assertEqual(ar.seek_index().builds, builds)  # no rebuilds after the covering build

    def test_unrank_matches_fresh_budget_index(self):
        ar = AutoregressiveEnumerable(self._model(5, 3, seed=2), max_len=3)
        fresh = ar.budget_index(budget_bits=8.0)
        for i in (0, 3, 40, min(100, fresh.total_count - 1)):
            self.assertEqual(ar.unrank(i), fresh.get(i))

    def test_float64_budget_past_int64_cliff_matches_exact(self):
        nlp = self._model(4, 3, seed=3)
        q = Quantizer(oversample=8)
        fb = int(math.ceil(70.0 * q.fine_per_bit()))  # budget past _INT64_SAFE_BITS
        idx_f, _ = AutoregressiveEnumerable(nlp, max_len=3, count_mode="float").quantized_count_index(q, fb)
        idx_e, _ = AutoregressiveEnumerable(nlp, max_len=3, count_mode="exact").quantized_count_index(q, fb)
        self.assertEqual(int(idx_f.total()), idx_e.total())  # 4**3 = 64 sequences, exactly counted both ways
        # bucket-level parity
        self.assertEqual(idx_f.hist.base, idx_e.hist.base)
        self.assertEqual([float(c) for c in idx_f.hist.data], [float(c) for c in idx_e.hist.data])

    def test_count_mode_validated(self):
        with self.assertRaises(ValueError):
            AutoregressiveEnumerable(self._model(3, 2), max_len=2, count_mode="bogus")


if __name__ == "__main__":
    unittest.main()
