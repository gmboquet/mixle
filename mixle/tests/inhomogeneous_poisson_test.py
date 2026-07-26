"""Tests for the inhomogeneous Poisson process (piecewise-constant intensity)."""

import math
import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats import InhomogeneousPoissonProcessDistribution, InhomogeneousPoissonProcessEstimator


class InhomogeneousPoissonProcessTest(unittest.TestCase):
    def test_log_density_matches_closed_form(self):
        d = InhomogeneousPoissonProcessDistribution([2.0, 0.5, 4.0], t_max=3.0)  # unit-width bins
        events = [0.2, 0.7, 2.1, 2.5, 2.9]  # counts per bin = [2, 0, 3]
        expected = 2 * math.log(2.0) + 0 * math.log(0.5) + 3 * math.log(4.0) - (2.0 + 0.5 + 4.0)
        self.assertAlmostEqual(d.log_density(events), expected, places=12)

    def test_events_outside_window_are_minus_inf(self):
        d = InhomogeneousPoissonProcessDistribution([1.0, 1.0], t_max=2.0)
        self.assertEqual(d.log_density([0.5, 2.5]), -np.inf)  # 2.5 > t_max
        self.assertEqual(d.log_density([]), -sum(d.rates * d.widths))  # empty realization is valid

    def test_seq_log_density_matches_scalar(self):
        d = InhomogeneousPoissonProcessDistribution([1.5, 0.3, 2.0, 1.0], t_max=4.0)
        realizations = [d.sampler(s).sample() for s in range(5)]
        enc = d.dist_to_encoder().seq_encode(realizations)
        seq = np.asarray(d.seq_log_density(enc))
        scalar = np.array([d.log_density(r) for r in realizations])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_sampler_intensity_matches_rates(self):
        rates = np.array([1.0, 5.0, 0.5, 3.0])
        d = InhomogeneousPoissonProcessDistribution(rates, t_max=4.0)  # unit-width bins
        reals = d.sampler(0).sample(4000)
        edges = d.edges
        per_bin = np.zeros(d.num_bins)
        for r in reals:
            per_bin += np.histogram(r, bins=edges)[0]
        empirical = per_bin / (d.widths * len(reals))  # events per unit time per bin
        np.testing.assert_allclose(empirical, rates, atol=0.2)

    def test_estimator_recovers_rates(self):
        true = InhomogeneousPoissonProcessDistribution([0.5, 4.0, 2.0], t_max=3.0)
        data = true.sampler(1).sample(5000)
        fit = optimize(
            data,
            InhomogeneousPoissonProcessEstimator(num_bins=3, t_max=3.0),
            max_its=1,
            rng=np.random.RandomState(0),
            out=None,
        )
        np.testing.assert_allclose(fit.rates, true.rates, atol=0.2)

    def test_state_is_finite_owned_and_immutable(self):
        with self.assertRaises(ValueError):
            InhomogeneousPoissonProcessDistribution(
                [1.0],
                edges=[0.0, np.inf],
            )
        rates = np.array([1.0, 2.0])
        edges = np.array([0.0, 1.0, 2.0])
        dist = InhomogeneousPoissonProcessDistribution(rates, edges=edges)
        expected = dist.log_density([0.5])
        rates[:] = 9.0
        edges[:] = [0.0, 10.0, 20.0]
        self.assertEqual(dist.log_density([0.5]), expected)
        with self.assertRaises(ValueError):
            dist.rates[0] = 3.0
        with self.assertRaises(ValueError):
            dist.edges[1] = 3.0

    def test_raw_encoded_and_accumulator_event_contracts_agree(self):
        dist = InhomogeneousPoissonProcessDistribution(
            [1.0, 1.0],
            t_max=2.0,
        )
        accumulator = dist.estimator().accumulator_factory().make()
        before = accumulator.value()
        for events in ([-1.0, 0.5], [0.5, 3.0], [0.5, np.nan]):
            with self.subTest(events=events):
                self.assertEqual(dist.log_density(events), -np.inf)
                with self.assertRaises(ValueError):
                    dist.dist_to_encoder().seq_encode([events])
                with self.assertRaises(ValueError):
                    accumulator.update(events, 1.0, None)
                np.testing.assert_array_equal(
                    accumulator.bin_counts,
                    before.bin_counts,
                )

        malformed_counts = (
            [[-1.0, 0.0]],
            [[0.5, 0.0]],
            [[np.nan, 0.0]],
            [[1.0]],
        )
        for counts in malformed_counts:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                dist.seq_log_density(counts)

    def test_statistics_and_weights_fail_closed(self):
        estimator = InhomogeneousPoissonProcessEstimator(
            num_bins=2,
            t_max=2.0,
        )
        accumulator = estimator.accumulator_factory().make()
        encoded = np.array([[1.0, 0.0]])
        for weights in ([], [-1.0], [np.nan]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                accumulator.seq_update(encoded, weights, None)
            np.testing.assert_array_equal(accumulator.bin_counts, [0.0, 0.0])
        malformed = (
            (np.array([1.0]), 1.0),
            (np.array([-1.0, 0.0]), 1.0),
            (np.array([1.0, 0.0]), 0.0),
            (np.array([0.0, 0.0]), np.nan),
        )
        for statistic in malformed:
            with self.subTest(statistic=statistic), self.assertRaises(ValueError):
                estimator.estimate(None, statistic)

    def test_gamma_prior_and_pseudo_count_are_applied_per_bin(self):
        estimator = InhomogeneousPoissonProcessEstimator(
            edges=[0.0, 1.0, 3.0],
            prior_shape=[2.0, 3.0],
            prior_rate=[4.0, 5.0],
            name="process",
            keys="shared",
        )
        model = estimator.estimate(None, (np.array([2.0, 4.0]), 2.0))
        np.testing.assert_allclose(model.rates, [3.0 / 6.0, 6.0 / 9.0])
        self.assertEqual(model.name, "process")
        self.assertEqual(model.keys, "shared")
        round_trip = model.estimator()
        np.testing.assert_array_equal(
            round_trip.prior_shape,
            estimator.prior_shape,
        )
        np.testing.assert_array_equal(
            round_trip.prior_rate,
            estimator.prior_rate,
        )

        smoothed = InhomogeneousPoissonProcessEstimator(
            num_bins=1,
            t_max=2.0,
            pseudo_count=2.0,
        ).estimate(None, (np.array([2.0]), 1.0))
        self.assertAlmostEqual(smoothed.rates[0], 4.0 / 4.0)
        with self.assertRaises(ValueError):
            InhomogeneousPoissonProcessEstimator(
                num_bins=1,
                t_max=1.0,
                pseudo_count=1.0,
                prior_shape=2.0,
            )

    def test_statistics_are_versioned_and_restored_defensively(self):
        estimator = InhomogeneousPoissonProcessEstimator(
            num_bins=2,
            t_max=2.0,
        )
        accumulator = estimator.accumulator_factory().make()
        accumulator.update([0.5], 1.0, None)
        value = accumulator.value()
        self.assertEqual(value.schema_version, 1)
        restored = estimator.accumulator_factory().make().from_value(value)
        value.bin_counts[0] = 99.0
        self.assertEqual(restored.bin_counts[0], 1.0)


if __name__ == "__main__":
    unittest.main()
