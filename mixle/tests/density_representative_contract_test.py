"""Contracts for Monte-Carlo density representatives."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class _UnfreezableRepresentative:
    __hash__ = None

    def __init__(self, value):
        self.value = value


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


if __name__ == "__main__":
    unittest.main()
