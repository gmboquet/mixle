"""Tests for the inhomogeneous Poisson process (piecewise-constant intensity)."""

import math
import unittest

import numpy as np

from pysp.inference.estimation import optimize
from pysp.stats import InhomogeneousPoissonProcessDistribution, InhomogeneousPoissonProcessEstimator


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


if __name__ == "__main__":
    unittest.main()
