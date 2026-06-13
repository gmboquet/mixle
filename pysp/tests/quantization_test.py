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
