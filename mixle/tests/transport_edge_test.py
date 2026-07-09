"""CARD F2-a: per-edge cross-modal transport premise check, run on a real edge distinct from
TRANSPORT-a's own toy problems (linear-Gaussian, x^2-ambiguous). The edge here is a cubic sensor
response ``y = x^3 + noise`` -- genuinely nonlinear (unlike TRANSPORT-a's linear-Gaussian case) but
monotonic and single-valued (unlike its bimodal x^2 case), the shape of a real saturating-sensor
inverse. No closed-form/reference posterior is used or needed: per the plan, a genuine edge only has
calibration (credible-interval coverage against held-out truth) to check itself against.

Skip note (also recorded in the module docstring): A3-a's quotient-leaf research spike already
recorded a negative result, so no equivariant refinement is attempted even where the premise passes
-- there is nothing proven to graft on.
"""

import unittest

import numpy as np

try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from mixle.reason.transport_edge import (
    COVERAGE_P_FLOOR,
    EdgeTransportVerdict,
    coverage_consistent_with_nominal,
    fit_conditional_transport,
    verify_edge_transport,
)


def _cubic_sensor_data(n, rng, noise_std=0.15):
    x = rng.uniform(-2.0, 2.0, size=(n, 1))
    y = x**3 + rng.normal(scale=noise_std, size=(n, 1))
    return x.astype(np.float64), y.astype(np.float64)


@unittest.skipUnless(_HAS_TORCH, "NeuralConditionalDensity needs torch")
class RealEdgePremiseCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(0)
        x_train, y_train = _cubic_sensor_data(400, rng)
        cls.x_test, cls.y_test = _cubic_sensor_data(60, np.random.RandomState(1))
        data = list(zip(y_train.tolist(), x_train.tolist()))  # (cond=y, target=x): p(x | y)
        cls.sampler = fit_conditional_transport(data, x_dim=1, y_dim=1, seed=0)

    def test_the_premise_is_checked_and_computed_not_assumed(self):
        verdict = verify_edge_transport("sensor_cubic", self.sampler, self.x_test, self.y_test, n_draws=200)
        self.assertIsInstance(verdict, EdgeTransportVerdict)
        self.assertEqual(verdict.edge_name, "sensor_cubic")
        self.assertEqual(len(verdict.coverage_rates), 1)  # x_dim == 1
        self.assertEqual(len(verdict.p_values), 1)

    def test_premise_passes_on_this_edge(self):
        verdict = verify_edge_transport("sensor_cubic", self.sampler, self.x_test, self.y_test, n_draws=200)
        self.assertTrue(verdict.usable, msg=verdict.reason)
        self.assertGreater(verdict.p_values[0], COVERAGE_P_FLOOR)

    def test_coverage_rate_is_close_to_the_nominal_90_percent(self):
        verdict = verify_edge_transport("sensor_cubic", self.sampler, self.x_test, self.y_test, n_draws=200)
        self.assertAlmostEqual(verdict.coverage_rates[0], 0.9, delta=0.15)


class CoverageHelperTest(unittest.TestCase):
    def test_all_covered_gives_high_p_value(self):
        rate, p = coverage_consistent_with_nominal([True] * 90 + [False] * 10)
        self.assertAlmostEqual(rate, 0.9)
        self.assertGreater(p, COVERAGE_P_FLOOR)

    def test_wildly_off_coverage_fails_the_floor(self):
        rate, p = coverage_consistent_with_nominal([True] * 10 + [False] * 90)
        self.assertAlmostEqual(rate, 0.1)
        self.assertLess(p, COVERAGE_P_FLOOR)


class _ConstantSampler:
    """A stand-in for a transport that learned nothing: always samples near 0, regardless of ``y``."""

    def sample_given(self, y):
        return np.zeros(1) + np.random.RandomState(0).normal(scale=0.01, size=1)

    def sample_given_batch(self, x_batch):
        n = np.atleast_2d(x_batch).shape[0]
        return np.zeros((n, 1)) + np.random.RandomState(0).normal(scale=0.01, size=(n, 1))


class KillCriterionUnusableEdgeTest(unittest.TestCase):
    def test_an_uninformative_transport_is_reported_unusable_with_a_named_reason(self):
        rng = np.random.RandomState(2)
        x_test, y_test = _cubic_sensor_data(40, rng)  # true x ranges over [-2, 2]^3-ish, far from 0
        verdict = verify_edge_transport("broken_edge", _ConstantSampler(), x_test, y_test, n_draws=50)
        self.assertFalse(verdict.usable)
        self.assertIn("dim(s)", verdict.reason)


if __name__ == "__main__":
    unittest.main()
