"""Tests for structural quantized enumeration (pysp.utils.quantization).

Covers the count semiring against brute force; the count-budget index for the additive families
(Composite/Sequence/MarkovChain) against the exact enumerator on finite supports (value set, exact
log densities, descending-probability ordering); the 128-bit scale target; nesting (Composite over
a Sequence child); and the parallel/distributed layer (determinism vs serial, distributed unranking
order).
"""
import math
import unittest

from pysp.stats import *
from pysp.utils.enumeration import freeze
from pysp.utils.quantization import (Quantizer, CountHistogram, leaf_count_index, convolve_indices,
                                     count_budget_index)
from pysp.utils.quantization_parallel import distributed_unrank
from pysp.utils.quantization_semiring import (CountSemiring, enumerate_and_bin,
                                              ordered_stream_from_count_index)


def _collect(index):
    return [index.get(i) for i in range(index.total_count)]


def _norm(items):
    return sorted((freeze(v), round(lp, 9)) for v, lp in items)


class CountSemiringTestCase(unittest.TestCase):

    def test_convolve_matches_brute_force(self):
        a = CountHistogram(2, [1, 3, 0, 5])
        b = CountHistogram(-1, [2, 0, 4])
        c = a.convolve(b)
        truth = {}
        for i, ai in enumerate(a.data):
            for j, bj in enumerate(b.data):
                truth[a.base + i + b.base + j] = truth.get(a.base + i + b.base + j, 0) + ai * bj
        for k, v in truth.items():
            self.assertEqual(c.count_at(k), v)
        self.assertEqual(c.total(), a.total() * b.total())

    def test_convolution_unranker_matches_brute_force(self):
        q = Quantizer(bin_width_bits=1.0, oversample=8)
        d1 = [('a', math.log(0.5)), ('b', math.log(0.3)), ('c', math.log(0.2))]
        d2 = [(0, math.log(0.6)), (1, math.log(0.4))]
        i1, _ = leaf_count_index(iter(d1), q, 10 ** 6)
        i2, _ = leaf_count_index(iter(d2), q, 10 ** 6)
        conv = convolve_indices([i1, i2], q, 10 ** 6)
        got = []
        h = conv.hist
        for i, c in enumerate(h.data):
            for off in range(c):
                got.append(conv.get_in_bucket(h.base + i, off))
        truth = [((v1, v2), lp1 + lp2) for v1, lp1 in d1 for v2, lp2 in d2]
        self.assertEqual(_norm(got), _norm(truth))


class SemiringContractTestCase(unittest.TestCase):
    """The witness-retaining count semiring: plus/times/product and the axis bridges."""

    def setUp(self):
        self.q = Quantizer(bin_width_bits=1.0, oversample=8)
        self.sr = CountSemiring()
        self.d1 = [('a', math.log(0.5)), ('b', math.log(0.3)), ('c', math.log(0.2))]
        self.d2 = [(0, math.log(0.6)), (1, math.log(0.4))]

    def _enum(self, items):
        return self.sr.from_enumerator(iter(items), self.q, 10 ** 6)[0]

    def _all(self, idx):
        out = []
        h = idx.hist
        for i, c in enumerate(h.data):
            for off in range(c):
                out.append(idx.get_in_bucket(h.base + i, off))
        return out

    def test_times_matches_independent_product(self):
        i1, i2 = self._enum(self.d1), self._enum(self.d2)
        prod = self.sr.times(i1, i2, self.q, 10 ** 6)
        truth = [((v1, v2), lp1 + lp2) for v1, lp1 in self.d1 for v2, lp2 in self.d2]
        self.assertEqual(_norm(self._all(prod)), _norm(truth))

    def test_product_equals_convolve_indices(self):
        # The retrofit must be equivalent to the previous hand-written path: identical histogram.
        i1, i2 = self._enum(self.d1), self._enum(self.d2)
        a = self.sr.product([i1, i2], self.q, 10 ** 6)
        b = convolve_indices([i1, i2], self.q, 10 ** 6)
        self.assertEqual(a.hist.base, b.hist.base)
        self.assertEqual(a.hist.data, b.hist.data)

    def test_plus_pools_alternatives(self):
        d3 = [('x', math.log(0.6)), ('y', math.log(0.4))]  # disjoint, same value space as d1
        i1, i3 = self._enum(self.d1), self._enum(d3)
        pooled = self.sr.plus(i1, i3)
        self.assertEqual(pooled.hist.total(), len(self.d1) + len(d3))
        self.assertEqual(_norm(self._all(pooled)), _norm(self.d1 + d3))

    def test_one_is_empty_product(self):
        one = self.sr.one()
        self.assertEqual(one.hist.total(), 1)
        v, lp = one.get_in_bucket(0, 0)
        self.assertEqual(v, ())
        self.assertEqual(lp, 0.0)

    def test_bridges_round_trip(self):
        cat = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
        # Axis B -> A: enumerate-and-bin equals the leaf count index.
        idx_bridge, _ = enumerate_and_bin(cat.enumerator(), self.q, 10 ** 6)
        self.assertEqual(_norm(self._all(idx_bridge)), _norm(list(cat.enumerator())))
        # Axis A -> B: unrank a built budget index back into a stream of the same value set.
        built = cat.count_budget_index(budget_bits=10, oversample=8)
        streamed = list(ordered_stream_from_count_index(built))
        self.assertEqual(_norm(streamed), _norm(list(cat.enumerator())))


class BudgetIndexVsEnumeratorTestCase(unittest.TestCase):
    """On finite supports the count-budget index must reproduce the exact enumerator."""

    def _check_finite(self, dist):
        truth = _norm(list(dist.enumerator()))
        index = dist.count_budget_index(budget_bits=24, oversample=8)
        got = _collect(index)
        self.assertEqual(index.total_count, len(truth))
        self.assertEqual(_norm(got), truth)
        for v, lp in got:
            self.assertAlmostEqual(lp, dist.log_density(v), places=9)
        # Descending probability across coarse bins (allow within-bin reordering up to a bin).
        lps = [lp for _, lp in got]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - 1.0)

    def test_leaf_categorical(self):
        self._check_finite(CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2}))

    def test_composite(self):
        cat3 = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
        intcat = IntegerCategoricalDistribution(2, [0.1, 0.0, 0.6, 0.3])
        self._check_finite(CompositeDistribution((cat3, intcat)))

    def test_sequence_finite_length(self):
        cat = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
        self._check_finite(SequenceDistribution(
            cat, len_dist=IntegerCategoricalDistribution(0, [0.2, 0.5, 0.3])))

    def test_markov_chain_finite_length(self):
        mc = MarkovChainDistribution(
            {'x': 0.6, 'y': 0.4},
            {'x': {'x': 0.8, 'y': 0.2}, 'y': {'x': 0.5, 'y': 0.5}},
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))
        self._check_finite(mc)


class ScaleTestCase(unittest.TestCase):
    """Deep budgets must be reachable structurally (no materialization) with exact log densities."""

    def _check_scale(self, dist, budget_bits=128):
        index = dist.count_budget_index(budget_bits=budget_bits, oversample=4)
        self.assertGreaterEqual(index.total_count, 2 ** budget_bits)
        for i in [0, 1, 137, index.total_count // 2, index.total_count - 1]:
            if i < index.total_count:
                v, lp = index.get(i)
                self.assertAlmostEqual(lp, dist.log_density(v), places=9)

    def test_sequence_128_bits(self):
        cat = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
        self._check_scale(SequenceDistribution(cat, len_dist=GeometricDistribution(0.4)))

    def test_markov_128_bits(self):
        mc = MarkovChainDistribution(
            {'x': 0.6, 'y': 0.4},
            {'x': {'x': 0.8, 'y': 0.2}, 'y': {'x': 0.5, 'y': 0.5}},
            len_dist=GeometricDistribution(0.3))
        self._check_scale(mc)

    def test_nested_composite_over_sequence_128_bits(self):
        cat = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2})
        seq = SequenceDistribution(cat, len_dist=GeometricDistribution(0.4))
        comp = CompositeDistribution((CategoricalDistribution({'x': 0.7, 'y': 0.3}),
                                      seq, GeometricDistribution(0.5)))
        self._check_scale(comp)


class RecursiveLawDepthTestCase(unittest.TestCase):
    """MarkovChain now routes through the count semiring; its reified carrier unranks iteratively."""

    def test_deep_markov_no_recursion_overflow(self):
        mc = MarkovChainDistribution(
            {'x': 0.55, 'y': 0.45}, {'x': {'x': 0.7, 'y': 0.3}, 'y': {'x': 0.4, 'y': 0.6}},
            len_dist=GeometricDistribution(0.2))
        idx = mc.count_budget_index(budget_bits=160, oversample=4)
        self.assertGreaterEqual(idx.total_count, 2 ** 160)
        n = idx.total_count
        deepest_v, deepest_lp = idx.get(n - 1)  # longest sequence -> deepest trellis walk
        self.assertGreater(len(deepest_v), 100)
        self.assertAlmostEqual(deepest_lp, mc.log_density(deepest_v), places=9)


class MarginalLawTestCase(unittest.TestCase):
    """BoundedCount for the MARGINAL families: Mixture (tropical pool) and HMM (enumerate-and-bin)."""

    def test_disjoint_mixture_is_exact(self):
        # Disjoint component supports => no overlap => the pooled count index is exact.
        m = MixtureDistribution(
            [IntegerCategoricalDistribution(0, [0.5, 0.3, 0.2]),
             IntegerCategoricalDistribution(10, [0.4, 0.6])], [0.6, 0.4])
        idx = m.count_budget_index(budget_bits=20, oversample=8)
        got = [idx.get(i) for i in range(idx.total_count)]
        self.assertEqual(_norm(got), _norm(list(m.enumerator())))
        for v, lp in got:
            self.assertAlmostEqual(lp, m.log_density(v), places=9)

    def test_overlapping_mixture_is_conservative_bound(self):
        # Overlapping supports => upper-bound counts (a shared value is counted per component),
        # but the value set covers the true support and every reported log-prob is exact.
        m = MixtureDistribution(
            [IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]),
             IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5])], [0.5, 0.5])
        true = m.enumerator()
        true_set = set(freeze(v) for v, _ in true)
        idx = m.count_budget_index(budget_bits=20, oversample=8)
        got = [idx.get(i) for i in range(idx.total_count)]
        got_set = set(freeze(v) for v, _ in got)
        self.assertTrue(true_set.issubset(got_set))           # covers the true support
        self.assertGreaterEqual(idx.total_count, len(true_set))  # conservative (>= distinct)
        for v, lp in got:
            self.assertAlmostEqual(lp, m.log_density(v), places=9)  # exact mixture log-prob

    def test_hmm_structural_is_conservative_bound(self):
        # Finite-length HMM: the structural trellis counts (path, obs) pairs, so an observation
        # produced by several paths is counted per path (conservative upper bound). The value set
        # covers the true support and every reported log-prob is the exact marginal.
        hmm = HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution({'a': 0.8, 'b': 0.2}),
                    CategoricalDistribution({'b': 0.6, 'c': 0.4})],
            w=[0.7, 0.3], transitions=[[0.9, 0.1], [0.4, 0.6]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))
        true_set = set(freeze(v) for v, _ in hmm.enumerator())
        idx = hmm.count_budget_index(budget_bits=16, oversample=8)
        got = [idx.get(i) for i in range(idx.total_count)]
        got_set = set(freeze(v) for v, _ in got)
        self.assertTrue(true_set.issubset(got_set))
        self.assertGreaterEqual(idx.total_count, len(true_set))
        for v, lp in got:
            self.assertAlmostEqual(lp, hmm.log_density(v), places=9)

    def test_hmm_structural_reaches_large_budget(self):
        # Geometric length -> the trellis reaches a deep budget structurally (no enumeration),
        # with exact marginal log-density on deep unranked observations and no recursion overflow.
        hmm = HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution({'a': 0.6, 'b': 0.3, 'c': 0.1}),
                    CategoricalDistribution({'a': 0.2, 'b': 0.3, 'c': 0.5})],
            w=[0.5, 0.5], transitions=[[0.7, 0.3], [0.4, 0.6]],
            len_dist=GeometricDistribution(0.3))
        idx = hmm.count_budget_index(budget_bits=64, oversample=4)
        self.assertGreaterEqual(idx.total_count, 2 ** 64)
        for i in [0, 1, idx.total_count // 2, idx.total_count - 1]:
            v, lp = idx.get(i)
            self.assertAlmostEqual(lp, hmm.log_density(v), places=9)


class MarginalDedupTestCase(unittest.TestCase):
    """Dedup of the over-counting MARGINAL stream: stateless canonical (seekable) + windowed LRU."""

    def _overlapping_mixture(self):
        return MixtureDistribution(
            [IntegerCategoricalDistribution(0, [0.7, 0.2, 0.1]),
             IntegerCategoricalDistribution(0, [0.2, 0.3, 0.5])], [0.5, 0.5])

    def _finite_hmm(self):
        return HiddenMarkovModelDistribution(
            topics=[CategoricalDistribution({'a': 0.8, 'b': 0.2}),
                    CategoricalDistribution({'b': 0.6, 'c': 0.4})],
            w=[0.7, 0.3], transitions=[[0.9, 0.1], [0.4, 0.6]],
            len_dist=IntegerCategoricalDistribution(0, [0.1, 0.3, 0.4, 0.2]))

    def test_canonical_mixture_dedups_to_distinct_support(self):
        m = self._overlapping_mixture()
        got = list(m.count_budget_distinct(budget_bits=20, oversample=8, dedup='canonical'))
        keys = [freeze(v) for v, _ in got]
        self.assertEqual(len(keys), len(set(keys)))                  # no duplicates (no bin ties here)
        self.assertEqual(set(keys), set(freeze(v) for v, _ in m.enumerator()))
        for v, lp in got:
            self.assertAlmostEqual(lp, m.log_density(v), places=9)

    def test_canonical_is_random_accessible(self):
        # Partition the STRUCTURAL rank range; the stateless predicate makes each slice independent,
        # so concatenating slices reproduces the full distinct stream exactly (start anywhere).
        m = self._overlapping_mixture()
        idx = m.count_budget_index(budget_bits=20, oversample=8)
        n = idx.total_count
        full = list(m.count_budget_distinct(budget_bits=20, oversample=8, dedup='canonical'))
        parts = []
        for a, b in [(0, n // 3), (n // 3, 2 * n // 3), (2 * n // 3, n)]:
            parts += list(m.count_budget_distinct(budget_bits=20, oversample=8,
                                                  dedup='canonical', start=a, stop=b))
        self.assertEqual([(freeze(v), round(lp, 9)) for v, lp in parts],
                         [(freeze(v), round(lp, 9)) for v, lp in full])

    def test_canonical_hmm_covers_distinct_support(self):
        hmm = self._finite_hmm()
        true_set = set(freeze(v) for v, _ in hmm.enumerator())
        structural_total = hmm.count_budget_index(budget_bits=24, oversample=8).total_count
        got = list(hmm.count_budget_distinct(budget_bits=24, oversample=8, dedup='canonical'))
        keys = [freeze(v) for v, _ in got]
        self.assertEqual(set(keys), true_set)                 # covers exactly the distinct support
        self.assertLess(len(keys), structural_total)          # path-copies substantially collapsed
        # Min-cost-path dedup is exact up to coarse-bin ties (>=2 paths sharing the min bin), which
        # are rare and shrink with finer bins; allow a small residual rather than asserting zero.
        self.assertLessEqual(len(keys) - len(true_set), 2)
        for v, lp in got:
            self.assertAlmostEqual(lp, hmm.log_density(v), places=9)

    def test_window_mode_dedups_sequentially(self):
        m = self._overlapping_mixture()
        got = list(m.count_budget_distinct(budget_bits=20, oversample=8,
                                           dedup='window', max_entries=1000))
        keys = [freeze(v) for v, _ in got]
        self.assertEqual(set(keys), set(freeze(v) for v, _ in m.enumerator()))
        with self.assertRaises(ValueError):  # window mode cannot seek
            list(m.count_budget_distinct(budget_bits=20, dedup='window', start=5))

    def test_exact_family_dedup_is_noop(self):
        comp = CompositeDistribution((CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2}),
                                      IntegerCategoricalDistribution(2, [0.1, 0.0, 0.6, 0.3])))
        idx = comp.count_budget_index(budget_bits=20, oversample=8)
        distinct = list(comp.count_budget_distinct(budget_bits=20, oversample=8))
        self.assertEqual(len(distinct), idx.total_count)
        self.assertEqual(_norm(distinct), _norm([idx.get(i) for i in range(idx.total_count)]))


class ParallelTestCase(unittest.TestCase):

    def setUp(self):
        cat = CategoricalDistribution({'a': 0.5, 'b': 0.3, 'c': 0.2, 'd': 0.05, 'e': 0.05})
        self.seq = SequenceDistribution(cat, len_dist=GeometricDistribution(0.5))

    def test_parallel_quantization_is_deterministic(self):
        serial = self.seq.count_budget_index(budget_bits=96, oversample=8)
        parallel = self.seq.count_budget_index(budget_bits=96, oversample=8, num_workers=4)
        self.assertEqual(serial.counts, parallel.counts)
        self.assertEqual(serial.total_count, parallel.total_count)
        for r in [0, 1, 2, 5, 100, 1000, serial.total_count // 2, serial.total_count - 1]:
            if r < serial.total_count:
                sv, slp = serial.get(r)
                pv, plp = parallel.get(r)
                self.assertEqual(freeze(sv), freeze(pv))
                self.assertAlmostEqual(slp, plp, places=12)

    def test_distributed_unranking_matches_serial_order(self):
        index = self.seq.count_budget_index(budget_bits=96, oversample=8)
        serial = [index.get(i) for i in range(2000)]
        dist_items = distributed_unrank(self.seq, budget_bits=96, start=0, count=2000,
                                        oversample=8, num_workers=4, backend='local')
        self.assertEqual(len(dist_items), 2000)
        self.assertEqual([(freeze(v), round(lp, 9)) for v, lp in serial],
                         [(freeze(v), round(lp, 9)) for v, lp in dist_items])


if __name__ == '__main__':
    unittest.main()
