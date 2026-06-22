"""Tests for the Plackett-Luce ranking distribution (normalization, sampling, MM estimation)."""

import itertools
import unittest

import numpy as np

from pysp.inference.estimation import fit
from pysp.stats import PlackettLuceDistribution


def _orderings(k):
    return [list(p) for p in itertools.permutations(range(k))]


class PlackettLuceTestCase(unittest.TestCase):
    def test_density_normalizes_over_all_orderings(self):
        dist = PlackettLuceDistribution([1.5, 0.2, -0.8, 0.0])
        orders = _orderings(4)
        enc = dist.dist_to_encoder().seq_encode(orders)
        self.assertAlmostEqual(float(np.sum(np.exp(dist.seq_log_density(enc)))), 1.0, places=10)

    def test_enumerator_matches_sorted_finite_support(self):
        dist = PlackettLuceDistribution([2.0, 0.5, -1.0])
        brute = [(o, dist.log_density(o)) for o in _orderings(3)]
        brute.sort(key=lambda u: -u[1])

        items = list(dist.enumerator())

        self.assertEqual([v for v, _ in items], [v for v, _ in brute])
        np.testing.assert_allclose([lp for _, lp in items], [lp for _, lp in brute], rtol=1.0e-12, atol=1.0e-12)
        self.assertAlmostEqual(float(np.logaddexp.reduce([lp for _, lp in items])), 0.0, places=10)

    def test_seq_matches_scalar(self):
        dist = PlackettLuceDistribution([2.0, 0.5, -1.0])
        orders = _orderings(3)
        enc = dist.dist_to_encoder().seq_encode(orders)
        scalar = np.asarray([dist.log_density(o) for o in orders])
        np.testing.assert_allclose(dist.seq_log_density(enc), scalar, rtol=1.0e-12, atol=1.0e-12)

    def test_density_invariant_to_log_worth_shift(self):
        orders = _orderings(3)
        base = PlackettLuceDistribution([5.0, 4.0, 3.0])
        shifted = PlackettLuceDistribution([5.0 - 2.5, 4.0 - 2.5, 3.0 - 2.5])
        enc = base.dist_to_encoder().seq_encode(orders)
        np.testing.assert_allclose(base.seq_log_density(enc), shifted.seq_log_density(enc), atol=1.0e-12)

    def test_string_round_trip(self):
        dist = PlackettLuceDistribution([1.0, 0.0, -1.0], name="pl", keys="k")
        self.assertEqual(str(eval(str(dist))), str(dist))

    def test_sampler_frequencies_match_density(self):
        dist = PlackettLuceDistribution([2.0, 0.5, -1.0])
        n = 60000
        samples = dist.sampler(seed=0).sample(n)
        orders = _orderings(3)
        index = {tuple(o): i for i, o in enumerate(orders)}
        counts = np.zeros(len(orders))
        for s in samples:
            counts[index[tuple(s)]] += 1
        empirical = counts / n
        expected = np.exp(dist.seq_log_density(dist.dist_to_encoder().seq_encode(orders)))
        np.testing.assert_allclose(empirical, expected, atol=0.01)

    def test_mm_recovers_log_worths(self):
        true = PlackettLuceDistribution([2.0, 1.0, 0.0, -1.0, -2.0])
        data = true.sampler(seed=1).sample(20000)
        fitted = fit(data, true.estimator(), max_its=300, rng=np.random.RandomState(0), print_iter=0)
        # log-worths are identified up to an additive constant; compare centered values.
        centered_true = true.log_w - true.log_w.mean()
        centered_fit = fitted.log_w - fitted.log_w.mean()
        np.testing.assert_allclose(centered_fit, centered_true, atol=0.1)

    def test_encoder_rejects_non_permutations(self):
        with self.assertRaises(ValueError):
            PlackettLuceDistribution([0.0, 0.0, 0.0]).dist_to_encoder().seq_encode([[0, 1, 1]])

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            PlackettLuceDistribution([1.0])


if __name__ == "__main__":
    unittest.main()
