"""Contracts for Monte-Carlo density representatives."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mixle.stats.compute.pdist import DistributionEnumerator
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution


class _UnfreezableRepresentative:
    __hash__ = None

    def __init__(self, value):
        self.value = value


class _ListEnumerator(DistributionEnumerator):
    def __init__(self, values, support_size):
        self._values = iter(values)
        super().__init__(SimpleNamespace(support_size=lambda: support_size))

    def __next__(self):
        return next(self._values)


class DensityRepresentativeContractTest(unittest.TestCase):
    def setUp(self):
        self.dist = GaussianDistribution(0.0, 1.0)

    def test_quantile_rejects_invalid_index_and_sample_budgets(self):
        for q in (np.nan, np.inf, -0.1, 1.1):
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    self.dist.density_quantile(q, n_samples=2)
        with self.assertRaises(TypeError):
            self.dist.density_quantile(True, n_samples=2)
        for budget in (0, -1):
            with self.subTest(n_samples=budget):
                with self.assertRaises(ValueError):
                    self.dist.density_quantile(0.5, n_samples=budget)
        for budget in (True, 1.5):
            with self.subTest(n_samples=budget):
                with self.assertRaises(TypeError):
                    self.dist.density_quantile(0.5, n_samples=budget)

    def test_enumeration_rejects_invalid_or_incoherent_budgets(self):
        for points in (0, -1):
            with self.subTest(num_points=points):
                with self.assertRaises(ValueError):
                    self.dist.density_enumeration(points, n_samples=2)
        with self.assertRaises(TypeError):
            self.dist.density_enumeration(1.5, n_samples=2)
        with self.assertRaises(ValueError):
            self.dist.density_enumeration(3, n_samples=2)
        with self.assertRaises(ValueError):
            self.dist.density_enumeration(1, n_samples=0)

    def test_nonfinite_score_policy_is_explicit(self):
        samples = SimpleNamespace(sample=lambda _: [0.0, 1.0])
        with (
            mock.patch.object(self.dist, "sampler", return_value=samples),
            mock.patch.object(
                self.dist,
                "log_density",
                side_effect=lambda value: np.nan if value == 0.0 else -1.0,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                self.dist.density_quantile(0.5, n_samples=2)
            self.assertEqual(
                self.dist.density_quantile(0.5, n_samples=2, invalid_score="omit"),
                1.0,
            )
            self.assertEqual(
                self.dist.density_enumeration(1, n_samples=2, invalid_score="omit"),
                [(1.0, -1.0)],
            )
            with self.assertRaises(ValueError):
                self.dist.density_enumeration(1, n_samples=2, invalid_score="ignore")

    def test_unfreezable_values_require_a_stable_key(self):
        values = [_UnfreezableRepresentative(1), _UnfreezableRepresentative(1)]
        samples = SimpleNamespace(sample=lambda _: values)
        with (
            mock.patch.object(self.dist, "sampler", return_value=samples),
            mock.patch.object(self.dist, "log_density", return_value=-1.0),
        ):
            with self.assertRaisesRegex(TypeError, "stable deduplication key"):
                self.dist.density_enumeration(1, n_samples=2)
            result = self.dist.density_enumeration(
                1,
                n_samples=2,
                dedup_key=lambda value: value.value,
            )
            self.assertEqual(len(result), 1)
            self.assertIs(result[0][0], values[0])

    def test_sampler_must_honor_the_requested_budget(self):
        samples = SimpleNamespace(sample=lambda _: [0.0])
        with mock.patch.object(self.dist, "sampler", return_value=samples):
            with self.assertRaisesRegex(ValueError, "returned 1 values"):
                self.dist.density_quantile(0.5, n_samples=2)
            with self.assertRaisesRegex(ValueError, "returned 1 values"):
                self.dist.density_enumeration(1, n_samples=2)

    def test_top_p_validates_controls_before_consuming(self):
        categorical = CategoricalDistribution({"a": 0.6, "b": 0.4})
        for target in (np.nan, np.inf, -0.1, 1.1):
            with self.subTest(p=target):
                with self.assertRaises(ValueError):
                    categorical.enumerator().top_p(target)
        with self.assertRaises(TypeError):
            categorical.enumerator().top_p(True)
        for cap in (0, -1):
            with self.subTest(max_items=cap):
                with self.assertRaises(ValueError):
                    categorical.enumerator().top_p(0.0, max_items=cap)
        for cap in (True, 1.5):
            with self.subTest(max_items=cap):
                with self.assertRaises(TypeError):
                    categorical.enumerator().top_p(0.0, max_items=cap)
        with self.assertRaisesRegex(ValueError, "requires max_items"):
            PoissonDistribution(2.0).enumerator().top_p(0.9)

    def test_top_p_reports_cap_and_target_status(self):
        categorical = CategoricalDistribution({"a": 0.6, "b": 0.4})
        capped = categorical.enumerator().top_p(0.9, max_items=1)
        self.assertFalse(capped.reached_target)
        self.assertTrue(capped.capped)
        self.assertFalse(capped.exhausted)
        self.assertAlmostEqual(capped.cumulative_probability, 0.6)

        reached = categorical.enumerator().top_p(0.9)
        self.assertTrue(reached.reached_target)
        self.assertFalse(reached.capped)
        self.assertTrue(reached.exhausted)
        self.assertAlmostEqual(reached.cumulative_probability, 1.0)

    def test_top_p_rejects_invalid_enumerated_log_masses(self):
        for log_prob in (np.nan, np.inf, -np.inf, 0.1):
            with self.subTest(log_prob=log_prob):
                enum = _ListEnumerator([("x", log_prob)], support_size=1)
                with self.assertRaises(ValueError):
                    enum.top_p(0.5)
        enum = _ListEnumerator([("x", True)], support_size=1)
        with self.assertRaises(TypeError):
            enum.top_p(0.5)


if __name__ == "__main__":
    unittest.main()
