"""Tests for support_size() and truncated_sum_bound() (enumeration upper bounds via truncation).

The descending-probability enumerator plus the support cardinality let us bound a truncated sum: the
top-k items are exact, and every un-enumerated item has probability <= p_k, so the tail is at most
(support_size - k) * p_k. These tests pin the cardinalities and that the bound is provably valid.
"""

import math
import unittest

from mixle.enumeration.density_rank import truncated_sum_bound
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.record import RecordDistribution
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.univariate.discrete.bernoulli import BernoulliDistribution
from mixle.stats.univariate.discrete.binomial import BinomialDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.integer_categorical import IntegerCategoricalDistribution
from mixle.stats.univariate.discrete.point_mass import PointMassDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

TOL = 1e-9


class SupportSizeTestCase(unittest.TestCase):
    def test_finite_leaf_cardinalities(self):
        self.assertEqual(CategoricalDistribution({"a": 0.5, "b": 0.5}).support_size(), 2)
        self.assertEqual(BernoulliDistribution(0.3).support_size(), 2)
        self.assertEqual(BinomialDistribution(p=0.4, n=6).support_size(), 7)
        self.assertEqual(IntegerCategoricalDistribution(0, [0.6, 0.4]).support_size(), 2)
        self.assertEqual(PointMassDistribution("x").support_size(), 1)

    def test_infinite_support_is_none(self):
        self.assertIsNone(PoissonDistribution(2.0).support_size())
        self.assertFalse(PoissonDistribution(2.0).support_is_finite())
        self.assertTrue(CategoricalDistribution({"a": 1.0}).support_is_finite())

    def test_decomposable_composition(self):
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        intc = IntegerCategoricalDistribution(0, [0.6, 0.4])
        self.assertEqual(CompositeDistribution((cat, intc)).support_size(), 6)  # 3 * 2
        self.assertEqual(RecordDistribution({"u": cat, "v": intc}).support_size(), 6)
        # an infinite child poisons the product
        self.assertIsNone(CompositeDistribution((cat, PoissonDistribution(1.0))).support_size())
        # mixture: upper bound = sum over components (union <= sum)
        mix = MixtureDistribution([cat, IntegerCategoricalDistribution(0, [0.5, 0.5])], [0.5, 0.5])
        self.assertEqual(mix.support_size(), 5)  # 3 + 2


class TruncatedSumBoundTestCase(unittest.TestCase):
    def test_tail_bound_is_a_valid_upper_bound(self):
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})
        b = truncated_sum_bound(cat, 2)
        self.assertEqual(b.num_enumerated, 2)
        self.assertFalse(b.exhausted)
        self.assertAlmostEqual(b.enumerated_mass, 0.8, delta=TOL)
        true_tail = 0.2
        self.assertGreaterEqual(b.tail_upper_bound, true_tail - TOL)  # (4-2)*0.3 = 0.6 >= 0.2
        self.assertGreaterEqual(b.total_upper_bound, 1.0 - TOL)

    def test_exhausted_support_is_exact(self):
        cat = CategoricalDistribution({"a": 0.6, "b": 0.4})
        b = truncated_sum_bound(cat, 10)
        self.assertTrue(b.exhausted)
        self.assertEqual(b.num_enumerated, 2)
        self.assertAlmostEqual(b.enumerated_mass, 1.0, delta=TOL)
        self.assertEqual(b.tail_upper_bound, 0.0)
        self.assertAlmostEqual(b.total_upper_bound, 1.0, delta=TOL)

    def test_infinite_support_has_no_tail_bound(self):
        b = truncated_sum_bound(PoissonDistribution(2.0), 3)
        self.assertIsNone(b.support_size)
        self.assertIsNone(b.tail_upper_bound)
        self.assertIsNone(b.total_upper_bound)
        self.assertGreater(b.enumerated_mass, 0.0)  # lower bound still available

    def test_zero_k_bounds_without_enumerating_anything(self):
        # MXR-080-0215: k=0 must mean "enumerate nothing", not "pull one item and stop" -- the old code
        # always drew at least one value from the stream before checking n >= k.
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})
        b = truncated_sum_bound(cat, 0)
        self.assertEqual(b.num_enumerated, 0)
        self.assertEqual(b.enumerated_mass, 0.0)
        self.assertEqual(b.last_log_prob, 0.0)  # exp(0.0) == 1.0: the loosest sound per-item cap
        self.assertFalse(b.exhausted)
        self.assertEqual(b.support_size, 4)
        self.assertAlmostEqual(b.tail_upper_bound, 4.0, delta=TOL)  # (4 - 0) * exp(0.0)
        self.assertAlmostEqual(b.total_upper_bound, 4.0, delta=TOL)

    def test_zero_k_on_empty_support_is_exhausted(self):
        # a genuinely empty (zero-item) support at k=0 is vacuously exhausted -- exhausted should track
        # the true state of the support, not just "we didn't pull anything because k said not to".
        from mixle.stats.univariate.discrete.categorical import CategoricalDistribution as _Cat

        empty = _Cat({})
        b = truncated_sum_bound(empty, 0)
        self.assertEqual(b.num_enumerated, 0)
        self.assertTrue(b.exhausted)
        self.assertEqual(b.support_size, 0)
        self.assertEqual(b.tail_upper_bound, 0.0)

    def test_negative_k_raises(self):
        cat = CategoricalDistribution({"a": 1.0})
        with self.assertRaises(ValueError):
            truncated_sum_bound(cat, -1)

    def test_composite_tail_bound_vs_brute_force(self):
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        intc = IntegerCategoricalDistribution(0, [0.6, 0.4])
        comp = CompositeDistribution((cat, intc))
        all_p = sorted((math.exp(comp.log_density((a, i))) for a in "abc" for i in (0, 1)), reverse=True)
        for k in (1, 3, 5):
            b = truncated_sum_bound(comp, k)
            true_tail = sum(all_p[b.num_enumerated :])
            self.assertGreaterEqual(b.tail_upper_bound, true_tail - TOL, "k=%d" % k)
            self.assertAlmostEqual(b.enumerated_mass, sum(all_p[: b.num_enumerated]), delta=1e-9)


if __name__ == "__main__":
    unittest.main()
