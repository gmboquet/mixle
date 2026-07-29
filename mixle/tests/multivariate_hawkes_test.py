"""Multivariate Hawkes: exact likelihood, bounded sampling, and exact MLE."""

import unittest
import warnings

import numpy as np

from mixle.inference import estimate
from mixle.stats import MultivariateHawkesProcessDistribution
from mixle.stats.processes.multivariate_hawkes import _split


def _brute_log_density(d, ev):
    """O(n^2) reference likelihood (direct kernel sum over all earlier events)."""
    t, m = _split(ev)
    n = t.size
    mu, al, be, w = d.mu, d.alpha, d.beta, d.window
    ll = 0.0
    for i in range(n):
        lam = mu[m[i]] + sum(al[m[i], m[k]] * np.exp(-be * (t[i] - t[k])) for k in range(i))
        ll += np.log(lam)
    comp = w * mu.sum() + sum(
        al[dd, m[k]] * (1 - np.exp(-be * (w - t[k]))) / be for dd in range(d.dim) for k in range(n)
    )
    return ll - comp


class MultivariateHawkesTest(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([0.5, 0.3])
        self.alpha = np.array([[0.4, 0.1], [0.2, 0.5]])
        self.beta = 1.5
        self.d = MultivariateHawkesProcessDistribution(self.mu, self.alpha, self.beta, 20.0)

    def test_subcritical_spectral_radius(self):
        self.assertLess(self.d.spectral_radius, 1.0)

    def test_recursion_matches_brute_force(self):
        for seed in (3, 4, 5):
            ev = self.d.sampler(seed=seed).sample()
            self.assertAlmostEqual(self.d.log_density(ev), _brute_log_density(self.d, ev), places=8)

    def test_seq_matches_scalar(self):
        evs = self.d.sampler(seed=4).sample(3)
        np.testing.assert_allclose(self.d.seq_log_density(evs), [self.d.log_density(e) for e in evs], atol=1e-10)

    def test_sampler_validity(self):
        ev = self.d.sampler(seed=3).sample()
        self.assertTrue(all(0 <= m < 2 for _, m in ev))
        self.assertTrue(all(ev[i][0] <= ev[i + 1][0] for i in range(len(ev) - 1)))
        self.assertTrue(all(0.0 <= t <= 20.0 for t, _ in ev))

    def test_exact_finite_window_fit_recovers_parameters(self):
        data = self.d.sampler(seed=0).sample(200)
        m = estimate(data, self.d.estimator())
        np.testing.assert_allclose(m.mu, self.mu, atol=0.1)
        np.testing.assert_allclose(m.alpha / m.beta, self.alpha / self.beta, atol=0.06)

    def test_super_critical_warns(self):
        d = MultivariateHawkesProcessDistribution([0.5, 0.3], [[1.6, 0.2], [0.2, 1.6]], 1.0, 20.0)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            d.sampler(seed=0)
            self.assertTrue(any("super-critical" in str(x.message) for x in wl))

    def test_bad_alpha_shape_raises(self):
        with self.assertRaises(ValueError):
            MultivariateHawkesProcessDistribution([0.5, 0.3], [[0.4, 0.1, 0.0], [0.2, 0.5, 0.0]], 1.5, 20.0)

    def test_constructor_owns_finite_parameter_snapshot(self):
        mu = self.mu.copy()
        alpha = self.alpha.copy()
        dist = MultivariateHawkesProcessDistribution(
            mu,
            alpha,
            self.beta,
            20.0,
        )
        before = dist.log_density([(1.0, 0)])
        mu[0] = 99.0
        alpha[0, 0] = 99.0
        self.assertEqual(dist.log_density([(1.0, 0)]), before)
        self.assertFalse(dist.mu.flags.writeable)
        self.assertFalse(dist.alpha.flags.writeable)
        for field in ("mu", "alpha", "beta", "window"):
            values = {
                "mu": self.mu.copy(),
                "alpha": self.alpha.copy(),
                "beta": self.beta,
                "window": 20.0,
            }
            if field in {"mu", "alpha"}:
                values[field].flat[0] = np.inf
            else:
                values[field] = np.inf
            with self.subTest(field=repr(field)), self.assertRaises(ValueError):
                MultivariateHawkesProcessDistribution(**values)

    def test_strict_history_and_query_contracts(self):
        tied = [(1.0, 0), (1.0, 1)]
        self.assertEqual(self.d.log_density(tied), -np.inf)
        with self.assertRaises(ValueError):
            self.d.dist_to_encoder().seq_encode([tied])
        with self.assertRaises(ValueError):
            self.d.intensity(2.0, [1.0, 1.5], [0])
        with self.assertRaises(ValueError):
            self.d.intensity(2.0, [1.0], [-1])
        with self.assertRaises(ValueError):
            self.d.intensity(2.0, [1.0], [0.5])
        with self.assertRaises(ValueError):
            self.d.expected_count(3.0, 2.0, [], [])

    def test_event_budget_fails_with_receipt(self):
        sampler = MultivariateHawkesProcessDistribution(
            [100.0],
            [[0.0]],
            1.0,
            1.0,
            max_events=0,
        ).sampler(seed=0)
        with self.assertRaises(RuntimeError):
            sampler.sample()
        self.assertFalse(sampler.last_receipt["complete"])
        self.assertEqual(
            sampler.last_receipt["termination_reason"],
            "event_budget_exhausted",
        )

    def test_evidence_schema_and_absent_fit_fail_closed(self):
        estimator = self.d.estimator()
        accumulator = estimator.accumulator_factory().make()
        events = np.array([[1.0, 0.0], [2.0, 1.0]])
        accumulator.update(events, 1.0, None)
        events[0, 0] = 9.0
        value = accumulator.value()
        self.assertEqual(value.schema_version, 1)
        np.testing.assert_array_equal(
            value.realizations[0],
            [[1.0, 0.0], [2.0, 1.0]],
        )
        self.assertFalse(value.realizations[0].flags.writeable)
        with self.assertRaises(ValueError):
            accumulator.seq_update(
                [[(1.0, 0)], [(2.0, 1)]],
                np.ones(1),
                None,
            )
        with self.assertRaises(ValueError):
            estimator.estimate(None, ((), np.zeros(0), 2, 20.0))

    def test_fractional_mark_rejected(self):
        # marks index the process dimension in {0, ..., D-1}; a fractional mark like 0.9 must not be
        # silently truncated to 0 by the int cast in _split before any validation ever sees it.
        with self.assertRaises(ValueError):
            _split([(1.0, 0.9), (2.0, 1)])
        with self.assertRaises(ValueError):
            self.d.log_density([(1.0, 0.9), (2.0, 1)])
        with self.assertRaises(ValueError):
            self.d.dist_to_encoder().seq_encode([[(1.0, 0.9), (2.0, 1)]])
        # integer-valued floats (e.g. 1.0) are legitimate marks and must still work
        times, marks = _split([(1.0, 1.0), (2.0, 0.0)])
        np.testing.assert_array_equal(marks, [1, 0])


if __name__ == "__main__":
    unittest.main()
