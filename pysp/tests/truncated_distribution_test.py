"""Tests for TruncatedDistribution (restrict a base distribution to an allowed support)."""

import math
import unittest

import numpy as np

from pysp.stats.base.categorical import CategoricalDistribution
from pysp.stats.base.poisson import PoissonDistribution
from pysp.stats.combinator.truncated import TruncatedDistribution

TOL = 1e-12


class TruncatedDistributionTestCase(unittest.TestCase):
    def setUp(self):
        self.cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})

    def test_allowed_form_renormalizes(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        z = 0.5 + 0.3 + 0.15
        self.assertAlmostEqual(math.exp(t.log_density("a")), 0.5 / z, delta=TOL)
        self.assertEqual(t.log_density("d"), -np.inf)
        self.assertAlmostEqual(sum(math.exp(t.log_density(v)) for v in "abc"), 1.0, delta=TOL)
        self.assertEqual(t.support_size(), 3)

    def test_forbidden_form_on_infinite_base(self):
        t = TruncatedDistribution(PoissonDistribution(2.0), forbidden=[0])
        z = 1.0 - math.exp(-2.0)
        self.assertAlmostEqual(math.exp(t.log_density(1)), math.exp(-2.0) * 2.0 / z, delta=1e-12)
        self.assertEqual(t.log_density(0), -np.inf)
        self.assertIsNone(t.support_size())  # infinite base minus a finite set is still infinite

    def test_enumerator_is_descending_and_normalized(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        items = list(t.enumerator())
        self.assertEqual([v for v, _ in items], ["a", "b", "c"])
        self.assertAlmostEqual(sum(math.exp(lp) for _, lp in items), 1.0, delta=TOL)
        lps = [lp for _, lp in items]
        self.assertTrue(all(lps[i] >= lps[i + 1] for i in range(len(lps) - 1)))
        for v, lp in items:
            self.assertAlmostEqual(lp, t.log_density(v), delta=TOL)

    def test_seq_log_density_matches(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        enc = t.dist_to_encoder().seq_encode(["a", "b", "c", "d"])
        sld = t.seq_log_density(enc)
        np.testing.assert_allclose(sld[:3], [t.log_density(v) for v in "abc"], atol=TOL)
        self.assertEqual(sld[3], -np.inf)

    def test_sampler_respects_truncation(self):
        t = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        s = t.sampler(0).sample(4000)
        self.assertNotIn("d", set(s))
        z = 0.95
        self.assertAlmostEqual(sum(1 for v in s if v == "a") / 4000, 0.5 / z, delta=0.03)

    def test_fixed_truncation_estimation_round_trip(self):
        truth = TruncatedDistribution(self.cat, allowed=["a", "b", "c"])
        data = truth.sampler(1).sample(6000)
        est = truth.estimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(truth.dist_to_encoder().seq_encode(data), np.ones(len(data)), truth)
        fitted = est.estimate(len(data), acc.value())
        self.assertIsInstance(fitted, TruncatedDistribution)
        z = 0.95
        self.assertAlmostEqual(math.exp(fitted.log_density("a")), 0.5 / z, delta=0.04)
        self.assertEqual(fitted.log_density("d"), -np.inf)

    def test_validation(self):
        with self.assertRaises(ValueError):
            TruncatedDistribution(self.cat)  # neither allowed nor forbidden
        with self.assertRaises(ValueError):
            TruncatedDistribution(self.cat, allowed=["a"], forbidden=["b"])  # both


if __name__ == "__main__":
    unittest.main()
