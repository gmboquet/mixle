"""Tests for ExponentialTiltedDistribution (reweight a base by exp(theta . T(x)), renormalized)."""

import math
import unittest

import numpy as np

from mixle.inference.estimation import optimize
from mixle.stats.combinator.exponential_tilt import (
    ExponentialTiltedDistribution,
    register_exponential_tilt,
    registered_tilt_families,
)
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution
from mixle.stats.univariate.discrete.poisson import PoissonDistribution

TOL = 1e-9


class TiltNormalizerTestCase(unittest.TestCase):
    def test_gaussian_tilt_is_shifted_gaussian(self):
        # N(mu, s2) tilted by theta (identity stat) == N(mu + theta s2, s2).
        base = GaussianDistribution(1.0, 4.0)
        t = ExponentialTiltedDistribution(base, theta=0.5)
        shifted = GaussianDistribution(1.0 + 0.5 * 4.0, 4.0)
        for x in (-2.0, 0.0, 1.5, 3.0):
            self.assertAlmostEqual(t.log_density(x), shifted.log_density(x), delta=1e-9)
        self.assertIsNotNone(t.closed_form())
        # logZ matches the analytic Gaussian CGF theta*mu + theta^2 s2 / 2.
        self.assertAlmostEqual(t.log_z, 0.5 * 1.0 + 0.5 * 0.25 * 4.0, delta=TOL)

    def test_poisson_tilt_is_rescaled_poisson(self):
        base = PoissonDistribution(3.0)
        theta = 0.4
        t = ExponentialTiltedDistribution(base, theta=theta)
        rescaled = PoissonDistribution(3.0 * math.exp(theta))
        for k in (0, 1, 3, 6):
            self.assertAlmostEqual(t.log_density(k), rescaled.log_density(k), delta=1e-9)
        self.assertAlmostEqual(t.log_z, 3.0 * (math.exp(theta) - 1.0), delta=TOL)

    def test_gamma_and_exponential_tilts(self):
        g = ExponentialTiltedDistribution(GammaDistribution(2.0, 1.5), theta=0.2)
        self.assertAlmostEqual(g.log_z, -2.0 * math.log1p(-0.2 * 1.5), delta=TOL)
        e = ExponentialTiltedDistribution(ExponentialDistribution(2.0), theta=0.3)
        self.assertAlmostEqual(e.log_z, -math.log1p(-0.3 * 2.0), delta=TOL)

    def test_out_of_domain_theta_raises(self):
        with self.assertRaises(ValueError):
            ExponentialTiltedDistribution(ExponentialDistribution(2.0), theta=0.6)  # theta*beta = 1.2 >= 1

    def test_registered_families(self):
        fams = registered_tilt_families()
        for name in ("GaussianDistribution", "PoissonDistribution", "GammaDistribution", "ExponentialDistribution"):
            self.assertIn(name, fams)


class EnumerableTiltTestCase(unittest.TestCase):
    def setUp(self):
        self.cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})

    def test_enumeration_normalizer_and_density(self):
        # T over a categorical via a callable statistic; Z by exact enumeration.
        stat = {"a": 0.0, "b": 1.0, "c": 2.0}
        t = ExponentialTiltedDistribution(self.cat, theta=1.0, statistic=lambda x: stat[x])
        z = 0.5 * math.exp(0.0) + 0.3 * math.exp(1.0) + 0.2 * math.exp(2.0)
        self.assertAlmostEqual(t.log_z, math.log(z), delta=TOL)
        self.assertAlmostEqual(math.exp(t.log_density("b")), 0.3 * math.exp(1.0) / z, delta=TOL)
        self.assertAlmostEqual(sum(math.exp(t.log_density(v)) for v in "abc"), 1.0, delta=TOL)

    def test_enumerator_descending_and_normalized(self):
        stat = {"a": 0.0, "b": 1.0, "c": 2.0}
        t = ExponentialTiltedDistribution(self.cat, theta=1.5, statistic=lambda x: stat[x])
        items = list(t.enumerator())
        lps = [lp for _, lp in items]
        self.assertTrue(all(lps[i] >= lps[i + 1] for i in range(len(lps) - 1)))
        self.assertAlmostEqual(sum(math.exp(lp) for _, lp in items), 1.0, delta=TOL)

    def test_seq_log_density_matches(self):
        stat = {"a": 0.0, "b": 1.0, "c": 2.0}
        t = ExponentialTiltedDistribution(self.cat, theta=0.7, statistic=lambda x: stat[x])
        data = ["a", "b", "c", "b"]
        enc = t.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(t.seq_log_density(enc), [t.log_density(v) for v in data], atol=TOL)


class TemperingTestCase(unittest.TestCase):
    def test_tempering_is_power(self):
        # T(x) = log p(x) -> p_theta ~ p^(1+theta). Check ratio of densities is the power law.
        cat = CategoricalDistribution({"a": 0.6, "b": 0.3, "c": 0.1})
        theta = 1.0  # p^2
        t = ExponentialTiltedDistribution(cat, theta=theta, statistic="log_density")
        z = sum(p ** (1 + theta) for p in (0.6, 0.3, 0.1))
        self.assertAlmostEqual(math.exp(t.log_density("a")), 0.6**2 / z, delta=TOL)
        self.assertAlmostEqual(math.exp(t.log_density("c")), 0.1**2 / z, delta=TOL)


class TiltSamplerTestCase(unittest.TestCase):
    def test_exact_sampler_for_poisson(self):
        base = PoissonDistribution(3.0)
        t = ExponentialTiltedDistribution(base, theta=0.5)
        s = t.sampler(0).sample(20000)
        self.assertAlmostEqual(np.mean(s), 3.0 * math.exp(0.5), delta=0.2)  # tilted mean = lam e^theta

    def test_enumerable_sampler_matches_pmf(self):
        cat = CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2})
        stat = {"a": 0.0, "b": 1.0, "c": 2.0}
        t = ExponentialTiltedDistribution(cat, theta=1.0, statistic=lambda x: stat[x])
        s = t.sampler(1).sample(20000)
        freq_c = sum(1 for v in s if v == "c") / 20000
        self.assertAlmostEqual(freq_c, math.exp(t.log_density("c")), delta=0.03)


class TiltEstimatorTestCase(unittest.TestCase):
    def test_mle_of_theta_recovers_tilt_gaussian(self):
        # Data from a tilted Gaussian; fit theta by the score equation with the base fixed.
        base = GaussianDistribution(0.0, 1.0)
        truth = ExponentialTiltedDistribution(base, theta=1.3)  # == N(1.3, 1)
        data = truth.sampler(0).sample(4000)
        est = ExponentialTiltedDistribution(base, theta=0.0).estimator(fit="theta")
        acc = est.accumulator_factory().make()
        acc.seq_update(
            ExponentialTiltedDistribution(base, theta=0.0).dist_to_encoder().seq_encode(data), np.ones(len(data)), None
        )
        fitted = est.estimate(len(data), acc.value())
        self.assertAlmostEqual(float(np.atleast_1d(fitted.theta)[0]), 1.3, delta=0.15)

    def test_mle_of_theta_recovers_tilt_poisson(self):
        base = PoissonDistribution(2.0)
        truth = ExponentialTiltedDistribution(base, theta=0.5)  # Poisson(2 e^0.5)
        data = truth.sampler(2).sample(5000)
        est = ExponentialTiltedDistribution(base, theta=0.0).estimator(fit="theta")
        acc = est.accumulator_factory().make()
        acc.seq_update(
            ExponentialTiltedDistribution(base, theta=0.0).dist_to_encoder().seq_encode(data), np.ones(len(data)), None
        )
        fitted = est.estimate(len(data), acc.value())
        self.assertAlmostEqual(float(np.atleast_1d(fitted.theta)[0]), 0.5, delta=0.1)

    def test_fit_base_mode_refits_base(self):
        base = GaussianDistribution(0.0, 1.0)
        proto = ExponentialTiltedDistribution(base, theta=0.5)
        data = GaussianDistribution(2.0, 1.0).sampler(0).sample(2000)
        est = proto.estimator(fit="base")
        acc = est.accumulator_factory().make()
        acc.seq_update(proto.dist_to_encoder().seq_encode(data), np.ones(len(data)), proto)
        fitted = est.estimate(len(data), acc.value())
        self.assertIsInstance(fitted, ExponentialTiltedDistribution)
        self.assertAlmostEqual(float(np.atleast_1d(fitted.theta)[0]), 0.5, delta=TOL)  # theta held fixed


class TiltColdStartTestCase(unittest.TestCase):
    """Regression tests for the cold-start path: ``optimize()`` with no ``prev_estimate``, which
    drives ``ExponentialTiltedAccumulator.acc_to_encoder()``/``initialize()``/``seq_initialize()``
    before any estimate exists (as opposed to ``update()``/``seq_update()``, exercised by a warm
    start / second pass)."""

    def test_cold_optimize_does_not_crash(self):
        # acc_to_encoder() used to advertise only the child's (base) encoder while seq_initialize
        # expected the (base_enc, tvals) pair that a prev_estimate=None (cold) optimize() pass
        # encodes data with -- raising "ValueError: too many values to unpack".
        base = GaussianDistribution(0.0, 1.0)
        proto = ExponentialTiltedDistribution(base, theta=0.0)
        estimator = proto.estimator(fit="theta")
        data = [2.0] * 200
        fitted = optimize(data, estimator, out=None)
        self.assertIsInstance(fitted, ExponentialTiltedDistribution)
        self.assertAlmostEqual(float(np.atleast_1d(fitted.theta)[0]), 2.0, delta=1e-4)

    def test_cold_fit_estimates_theta_matching_warm_path(self):
        # initialize() used to only forward observations to the child (base) accumulator, leaving
        # sum_t=0/count=0 -- so fitting from scratch (a single initialize() pass, no prior
        # estimate) on data that's all 2.0 estimated theta=0. update() (a warm-start / second pass
        # given a prior estimate) correctly estimates theta ~= 2.0.
        #
        # This drives the module-level initialize()/update() accumulator calls directly rather
        # than going through optimize()'s EM loop: for this closed-form score-equation fit, each EM
        # iteration re-derives theta from a fresh seq_update() independent of the previous theta,
        # so optimize() happens to self-correct a bad cold-start estimate after a single step --
        # masking this bug end-to-end once Bug 1 (the encoder crash) is fixed.
        from mixle.stats.compute.sequence import initialize as free_initialize

        base = GaussianDistribution(0.0, 1.0)
        proto = ExponentialTiltedDistribution(base, theta=0.0)
        estimator = proto.estimator(fit="theta")
        data = [2.0] * 200

        # p=1.0 keeps every observation (no random subsampling), so the result is deterministic.
        cold = free_initialize(data=data, estimator=estimator, rng=np.random.RandomState(0), p=1.0)

        acc_warm = estimator.accumulator_factory().make()
        for v in data:
            acc_warm.update(v, 1.0, None)
        warm = estimator.estimate(len(data), acc_warm.value())

        cold_theta = float(np.atleast_1d(cold.theta)[0])
        warm_theta = float(np.atleast_1d(warm.theta)[0])
        self.assertAlmostEqual(cold_theta, 2.0, delta=1e-4)
        self.assertAlmostEqual(warm_theta, 2.0, delta=1e-4)
        self.assertAlmostEqual(cold_theta, warm_theta, delta=1e-6)

    def test_cold_initialize_accumulates_tilt_statistic(self):
        # Direct accumulator-level check (bypassing optimize()) that initialize()/seq_initialize()
        # accumulate the tilt's own sum_t/count -- not just the child's sufficient statistics.
        base = GaussianDistribution(0.0, 1.0)
        proto = ExponentialTiltedDistribution(base, theta=0.0)
        estimator = proto.estimator(fit="theta")
        data = [2.0] * 200
        rng = np.random.RandomState(0)

        acc = estimator.accumulator_factory().make()
        for v in data:
            acc.initialize(v, 1.0, rng)
        self.assertEqual(acc.count, 200.0)
        np.testing.assert_allclose(acc.sum_t, [400.0], atol=1e-9)

        acc_seq = estimator.accumulator_factory().make()
        enc = acc_seq.acc_to_encoder().seq_encode(data)
        acc_seq.seq_initialize(enc, np.ones(len(data)), rng)
        self.assertEqual(acc_seq.count, 200.0)
        np.testing.assert_allclose(acc_seq.sum_t, [400.0], atol=1e-9)


class TiltValidationTestCase(unittest.TestCase):
    def test_user_normalizer_used(self):
        base = GaussianDistribution(0.0, 1.0)
        # Supply the analytic CGF explicitly; should match the registered path.
        t = ExponentialTiltedDistribution(base, theta=0.5, log_normalizer=lambda th: 0.5 * th * th)
        self.assertAlmostEqual(t.log_z, 0.5 * 0.25, delta=TOL)

    def test_non_enumerable_without_normalizer_raises(self):
        base = GaussianDistribution(0.0, 1.0)
        with self.assertRaises(ValueError):
            ExponentialTiltedDistribution(base, theta=0.5, statistic=lambda x: x * x)  # no CGF, not enumerable

    def test_custom_registration(self):
        class _Dummy(GaussianDistribution):
            pass

        register_exponential_tilt(
            _Dummy,
            lambda b, th: __import__("mixle.stats.combinator.exponential_tilt", fromlist=["TiltResult"]).TiltResult(
                0.0
            ),
        )
        self.assertIn("_Dummy", registered_tilt_families())


if __name__ == "__main__":
    unittest.main()
