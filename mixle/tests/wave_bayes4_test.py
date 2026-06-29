"""Tests for the mixle.stats driver/wrapper fixes (wave 4).

Covers:
  - local estimate() / seq_estimate() working end-to-end on a small Gaussian mixture,
  - IgnoredEstimator.set_prior accepting a single prior argument (protocol fix) and delegating to
    the wrapped distribution,
  - OptionalDistribution/OptionalEstimator estimation round-trips and seq/scalar parity.

The original wave-4 suite also exercised the legacy driver's getattr/arity dispatch helpers
(``_accumulator_factory`` / ``_estimator_estimate``) and the pandas ``df_update`` /
``df_initialize`` branches. Those were internal compatibility shims for the older camelCase /
two-argument estimator API; the folded mixle.stats driver has a single clean
``accumulator_factory()`` + ``estimate(nobs, suff_stat)`` surface and no DataFrame branch, so those
tests do not apply and are dropped.
"""

import inspect
import unittest

import numpy as np

from mixle.inference import estimate, initialize, seq_estimate
from mixle.stats import (
    GaussianDistribution,
    GaussianEstimator,
    MixtureDistribution,
    MixtureEstimator,
    seq_encode,
    seq_log_density_sum,
)
from mixle.stats.combinator.ignored import (
    IgnoredAccumulator,
    IgnoredDistribution,
    IgnoredEstimator,
)
from mixle.stats.combinator.optional import OptionalDistribution, OptionalEstimator


class LocalEstimateTestCase(unittest.TestCase):
    """Local (iterable) estimation works end-to-end on a mixture."""

    @staticmethod
    def make_problem(n=400, seed=1):
        truth = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        data = truth.sampler(seed=seed).sample(n)
        est = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        return data, est

    def test_estimate_end_to_end(self):
        data, est = self.make_problem()
        mm = initialize(data, est, np.random.RandomState(2), 0.5)

        enc = seq_encode(data, model=mm)
        _, ll0 = seq_log_density_sum(enc, mm)

        for _ in range(8):
            mm = estimate(data, est, prev_estimate=mm)

        _, ll1 = seq_log_density_sum(enc, mm)
        self.assertGreater(ll1, ll0)
        self.assertAlmostEqual(np.sum(mm.w), 1.0, places=8)

    def test_estimate_recovers_components_from_warm_start(self):
        # EM symmetry breaking can take arbitrarily many iterations from a symmetric
        # initialization, so component recovery is tested from a deterministic warm start.
        data, est = self.make_problem()
        mm = MixtureDistribution([GaussianDistribution(-1.0, 4.0), GaussianDistribution(1.0, 4.0)], [0.5, 0.5])

        for _ in range(10):
            mm = estimate(data, est, prev_estimate=mm)

        self.assertAlmostEqual(np.sum(mm.w), 1.0, places=8)
        mus = sorted(c.mu for c in mm.components)
        self.assertAlmostEqual(mus[0], -3.0, delta=0.5)
        self.assertAlmostEqual(mus[1], 3.0, delta=0.5)

    def test_seq_estimate_matches_estimate(self):
        data, est = self.make_problem(n=300, seed=4)
        mm = initialize(data, est, np.random.RandomState(3), 0.5)
        enc = seq_encode(data, model=mm)

        m_seq = seq_estimate(enc, est, mm)
        m_loc = estimate(data, est, prev_estimate=mm)

        self.assertTrue(np.allclose(np.sort(m_seq.w), np.sort(m_loc.w)))
        self.assertTrue(np.allclose(sorted(c.mu for c in m_seq.components), sorted(c.mu for c in m_loc.components)))


class IgnoredTestCase(unittest.TestCase):
    class _RecordingDist:
        def __init__(self):
            self.prior = None

        def set_prior(self, prior):
            self.prior = prior

        def get_prior(self):
            return self.prior

    def test_estimator_set_prior_signature(self):
        params = list(inspect.signature(IgnoredEstimator.set_prior).parameters)
        self.assertEqual(params, ["self", "prior"])

    def test_estimator_set_prior_delegates(self):
        inner = self._RecordingDist()
        est = IgnoredEstimator(dist=inner)
        sentinel = object()
        est.set_prior(sentinel)
        self.assertIs(inner.prior, sentinel)
        self.assertIs(est.get_prior(), sentinel)

    def test_distribution_set_prior_delegates(self):
        inner = self._RecordingDist()
        d = IgnoredDistribution(inner)
        sentinel = object()
        d.set_prior(sentinel)
        self.assertIs(d.get_prior(), sentinel)

    def test_estimate_ignores_data(self):
        g = GaussianDistribution(2.5, 1.5)
        est = IgnoredEstimator(dist=g)

        acc = est.accumulator_factory().make()
        self.assertIsInstance(acc, IgnoredAccumulator)
        for x in [1.0, 2.0, 3.0]:
            acc.update(x, 1.0, None)
        self.assertIsNone(acc.value())
        self.assertIs(acc.combine(None), acc)
        self.assertIs(acc.from_value(None), acc)

        d = est.estimate(None, acc.value())
        self.assertIsInstance(d, IgnoredDistribution)
        self.assertIs(d.dist, g)

    def test_local_estimate_driver_with_ignored(self):
        g = GaussianDistribution(0.5, 2.0)
        d = estimate([10.0, 20.0, 30.0], IgnoredEstimator(dist=g))
        self.assertIsInstance(d, IgnoredDistribution)
        self.assertEqual(d.dist.mu, 0.5)
        self.assertAlmostEqual(d.log_density(0.5), g.log_density(0.5), places=12)

    def test_scoring_and_sampling_delegate(self):
        g = GaussianDistribution(1.0, 2.0)
        d = IgnoredDistribution(g)
        data = [0.0, 1.0, 2.5]
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [g.log_density(x) for x in data]))
        samples = d.sampler(seed=1).sample(size=5)
        self.assertEqual(len(samples), 5)


class OptionalTestCase(unittest.TestCase):
    def test_estimate_recovers_missing_rate_and_component(self):
        truth = OptionalDistribution(GaussianDistribution(2.0, 1.0), p=0.3)
        data = truth.sampler(seed=5).sample(400)
        self.assertTrue(any(x is None for x in data))

        model = estimate(data, OptionalEstimator(GaussianEstimator(), est_prob=True))

        n_missing = sum(1 for x in data if x is None)
        self.assertAlmostEqual(model.p, n_missing / float(len(data)), delta=0.01)
        self.assertAlmostEqual(model.dist.mu, 2.0, delta=0.3)

    def test_distribution_estimator_roundtrip(self):
        # Regression: estimator() used to omit the required inner estimator
        # argument and raise TypeError.
        d = OptionalDistribution(GaussianDistribution(2.0, 1.0), p=0.3)
        est = d.estimator()
        self.assertIsInstance(est, OptionalEstimator)
        self.assertIsNone(est.missing_value)

        data = d.sampler(seed=5).sample(200)
        model = estimate(data, est)
        self.assertIsInstance(model, OptionalDistribution)
        n_missing = sum(1 for x in data if x is None)
        self.assertAlmostEqual(model.p, n_missing / float(len(data)), delta=0.02)

    def test_seq_log_density_matches_scalar_with_missing(self):
        d = OptionalDistribution(GaussianDistribution(0.0, 1.0), p=0.25)
        data = d.sampler(seed=7).sample(60)
        enc = d.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(d.seq_log_density(enc), [d.log_density(x) for x in data]))


if __name__ == "__main__":
    unittest.main()
