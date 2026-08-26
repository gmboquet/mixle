"""Release 0.8.0 wave-3 family regressions: GEV covering-clamp disclosure/recovery (B12), Gamma
zero-support fail-closed (t4-major), zero/NaN/boundary error naming for Weibull, Rayleigh and Beta
(t4-minors), and the von Mises wrap-around disclosure (t4-minor)."""

import unittest

import numpy as np

import mixle
import mixle.dist as D
from mixle.stats import GeneralizedExtremeValueEstimator
from mixle.stats.directional import von_mises as von_mises_module
from mixle.stats.directional.von_mises import VonMisesDistribution, VonMisesEstimator
from mixle.stats.univariate.continuous.beta import BetaAccumulator
from mixle.stats.univariate.continuous.gamma import GammaAccumulator
from mixle.stats.univariate.continuous.rayleigh import RayleighAccumulator
from mixle.stats.univariate.continuous.weibull import WeibullAccumulator


def _heavy_right_tail_with_deep_min(n: int = 5000) -> np.ndarray:
    """Heavy right tail plus deep negative points: the covering clamp binds with xi > 0."""
    rng = np.random.RandomState(7)
    x = np.exp(rng.normal(7.0, 1.0, size=n)) - 500.0
    return np.append(x, [-8000.0, -6000.0, -7500.0])


def _left_skewed_no_outlier(n: int = 4000) -> np.ndarray:
    """Left-skewed data whose covering clamp binds mildly (xi < 0 kept, no Gumbel fallback)."""
    rng = np.random.RandomState(11)
    return -np.exp(rng.normal(0.0, 0.6, size=n)) + 5.0


class GEVCoveringClampDisclosureTest(unittest.TestCase):
    """B12: a covering-clamped moment fit must not present as a clean, unrepaired MLE."""

    def test_catastrophic_clamp_recovers_to_gumbel_and_records_repair(self):
        x = _heavy_right_tail_with_deep_min()
        g = mixle.Model(D.GeneralizedExtremeValueEstimator()).fit(x.tolist()).fitted
        ll = float(np.sum(g.seq_log_density(np.asarray(x))))
        # Pre-fix this fit reported total log-likelihood -9.6e+35 with repairs=() -- one real
        # observation sat within 1e-6 sd of the fitted support endpoint. The estimate must now
        # recover (Gumbel limit) and stay in the same universe as a per-observation density.
        self.assertTrue(np.isfinite(ll))
        self.assertGreater(ll / len(x), -100.0)
        self.assertTrue(any("shape-covering" in r for r in g.numerical_repairs()))
        prov = g.fit_provenance()
        self.assertTrue(any("shape-covering" in r for r in prov.repairs))
        self.assertTrue(prov.is_approximate())

    def test_mild_clamp_keeps_shape_and_records_repair(self):
        x = _left_skewed_no_outlier()
        g = mixle.Model(D.GeneralizedExtremeValueEstimator()).fit(x.tolist()).fitted
        self.assertTrue(any(r.startswith("shape-covering-clamped(") for r in g.numerical_repairs()))
        self.assertNotEqual(g.shape, 0.0)  # the clamped (non-Gumbel) shape was kept
        ll = float(np.sum(g.seq_log_density(np.asarray(x))))
        self.assertTrue(np.isfinite(ll))
        # The estimate's own training data stays in support with a finite, sane density.
        self.assertTrue(np.isfinite(g.log_density(float(x.max()))))
        self.assertTrue(np.isfinite(g.log_density(float(x.min()))))

    def test_clean_fit_records_no_repairs(self):
        d = D.GeneralizedExtremeValueDistribution(0.0, 1.0, 0.1)
        y = np.asarray(d.sampler(seed=3).sample(4000))
        g = mixle.Model(D.GeneralizedExtremeValueEstimator()).fit(y.tolist()).fitted
        self.assertEqual(g.numerical_repairs(), ())
        self.assertEqual(g.fit_provenance().repairs, ())
        self.assertAlmostEqual(g.shape, 0.1, delta=0.05)

    def test_direct_estimate_covering_clamp_records_repair(self):
        # Same defect through the direct estimator API (no EM loop attaching provenance).
        x = _heavy_right_tail_with_deep_min()
        est = GeneralizedExtremeValueEstimator()
        acc = est.accumulator_factory().make()
        acc.seq_update(np.asarray(x), np.ones(len(x)), None)
        fitted = est.estimate(None, acc.value())
        self.assertTrue(any("shape-covering" in r for r in fitted.numerical_repairs()))
        self.assertTrue(np.isfinite(fitted.log_density(float(x.min()))))


class GammaZeroSupportTest(unittest.TestCase):
    """t4-major: zeros used to collapse the fit to k=1 (an Exponential) silently."""

    def _fares_with_zeros(self) -> np.ndarray:
        rng = np.random.RandomState(3)
        f = np.round(rng.gamma(1.05, 31.0, size=891), 4)
        f[:15] = 0.0
        return f

    def test_zeros_fail_closed_named(self):
        f = self._fares_with_zeros()
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.GammaEstimator()).fit(f)
        msg = str(ctx.exception)
        self.assertIn("support x > 0", msg)
        self.assertIn("are exactly 0.0", msg)  # count is per encoded chunk, deliberately "at least"

    def test_scalar_update_zero_named(self):
        acc = GammaAccumulator()
        with self.assertRaises(ValueError) as ctx:
            acc.update(0.0, 1.0, None)
        self.assertIn("exactly 0.0", str(ctx.exception))

    def test_positive_part_estimate_unchanged(self):
        f = self._fares_with_zeros()
        pos = f[f > 0.0]
        g = mixle.Model(D.GammaEstimator()).fit(pos).fitted
        from scipy import stats

        a, _, sc = stats.gamma.fit(pos, floc=0)
        self.assertAlmostEqual(g.k, a, places=4)
        self.assertAlmostEqual(g.theta, sc, places=3)

    def test_nan_named_as_missing_data(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.GammaEstimator()).fit(np.array([1.0, np.nan, 2.0]))
        msg = str(ctx.exception)
        self.assertIn("NaN", msg)
        self.assertIn("missing", msg)
        self.assertNotIn("support x > 0,", msg)

    def test_negative_keeps_standard_support_error(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.GammaEstimator()).fit(np.array([-1.0, 1.0, 2.0]))
        self.assertIn("GammaDistribution has support x > 0.", str(ctx.exception))


class WeibullRayleighZeroFitTest(unittest.TestCase):
    """t4-minor: the zero-rejection must name the zeros, not the EM internals."""

    def test_weibull_zero_error_names_zeros(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.WeibullEstimator()).fit(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        msg = str(ctx.exception)
        self.assertIn("exactly 0.0", msg)
        self.assertNotIn("fused EM", msg)

    def test_weibull_negative_error_unchanged(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.WeibullEstimator()).fit(np.array([-1.0, 1.0, 2.0]))
        self.assertIn("WeibullDistribution requires observations x >= 0.", str(ctx.exception))

    def test_weibull_nan_named_as_missing_data(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.WeibullEstimator()).fit(np.array([1.0, np.nan, 2.0]))
        self.assertIn("NaN", str(ctx.exception))
        self.assertIn("missing", str(ctx.exception))

    def test_weibull_zero_weight_zero_is_admitted(self):
        # A zero carrying zero weight is no evidence (mixture components must survive it).
        acc = WeibullAccumulator()
        acc.update(0.0, 0.0, None)
        acc.seq_update((np.array([0.0, 1.0]), np.array([-np.inf, 0.0])), np.array([0.0, 1.0]), None)
        self.assertEqual(acc.count, 1.0)

    def test_weibull_scoring_zero_still_works(self):
        d = D.WeibullDistribution(2.0, 1.5)
        self.assertEqual(d.log_density(0.0), -np.inf)
        enc = d.dist_to_encoder().seq_encode(np.array([0.0, 1.0]))
        self.assertEqual(float(d.seq_log_density(enc)[0]), -np.inf)

    def test_rayleigh_zero_error_names_zeros(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.RayleighEstimator()).fit(np.array([0.0, 1.0, 2.0, 3.0]))
        msg = str(ctx.exception)
        self.assertIn("exactly 0.0", msg)
        self.assertNotIn("fused EM", msg)

    def test_rayleigh_nan_named_as_missing_data(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.RayleighEstimator()).fit(np.array([1.0, np.nan, 2.0]))
        self.assertIn("NaN", str(ctx.exception))

    def test_rayleigh_zero_weight_zero_is_admitted(self):
        acc = RayleighAccumulator()
        acc.update(0.0, 0.0, None)
        x = np.array([0.0, 2.0])
        acc.seq_update((x, x * x, np.log(np.where(x > 0, x, 1.0))), np.array([0.0, 1.0]), None)
        self.assertEqual(acc.count, 1.0)

    def test_rayleigh_healthy_fit_unchanged(self):
        d = D.RayleighDistribution(3.0)
        y = np.asarray(d.sampler(seed=5).sample(20000))
        r = mixle.Model(D.RayleighEstimator()).fit(y).fitted
        self.assertAlmostEqual(r.sigma, 3.0, delta=0.05)


class BetaBoundaryFitTest(unittest.TestCase):
    """t4-minor: boundary values must be named, not reported as bad 'sufficient statistics'."""

    def test_boundary_values_named(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.BetaEstimator()).fit(np.array([0.0, 0.2, 0.4, 0.6, 1.0, 0.5]))
        msg = str(ctx.exception)
        self.assertIn("exactly 0.0 or 1.0", msg)
        self.assertNotIn("sufficient statistics", msg)

    def test_scalar_update_boundary_named(self):
        acc = BetaAccumulator()
        with self.assertRaises(ValueError) as ctx:
            acc.update(1.0, 1.0, None)
        self.assertIn("exactly 0.0 or 1.0", str(ctx.exception))

    def test_zero_weight_boundary_does_not_poison_statistics(self):
        acc = BetaAccumulator()
        enc = D.BetaDistribution(2.0, 2.0).dist_to_encoder().seq_encode(np.array([0.0, 0.5, 1.0]))
        acc.seq_update(enc, np.array([0.0, 1.0, 0.0]), None)
        stats = acc.value()
        self.assertTrue(all(np.isfinite(v) for v in stats))
        self.assertEqual(stats[0], 1.0)

    def test_nan_named_as_missing_data(self):
        with self.assertRaises(ValueError) as ctx:
            mixle.Model(D.BetaEstimator()).fit(np.array([0.5, np.nan, 0.7]))
        self.assertIn("NaN", str(ctx.exception))

    def test_healthy_fit_unchanged(self):
        d = D.BetaDistribution(2.0, 5.0)
        y = np.asarray(d.sampler(seed=9).sample(20000))
        b = mixle.Model(D.BetaEstimator()).fit(y).fitted
        self.assertAlmostEqual(b.a, 2.0, delta=0.15)
        self.assertAlmostEqual(b.b, 5.0, delta=0.35)


class VonMisesWrapAroundDisclosureTest(unittest.TestCase):
    """t4-minor: the wrap-around semantics must be stated where a user will find them."""

    def test_module_and_class_docs_state_wraparound(self):
        self.assertIn("modulo", von_mises_module.__doc__)
        self.assertIn("not comparable", von_mises_module.__doc__)
        self.assertIn("modulo", VonMisesDistribution.__doc__)
        self.assertIn("wrap", VonMisesEstimator.__doc__.lower())

    def test_log_density_is_periodic_as_documented(self):
        d = VonMisesDistribution(0.7, 2.5)
        for x in (0.0, 1.3, -2.2, 100.0):
            self.assertAlmostEqual(d.log_density(x), d.log_density(x + 2.0 * np.pi), places=10)
            self.assertAlmostEqual(d.log_density(x), d.log_density(x - 6.0 * np.pi), places=10)


if __name__ == "__main__":
    unittest.main()
