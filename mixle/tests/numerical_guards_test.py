"""Regression guards for silent-corruption numerical edge cases (cleanup audit, Phase 1).

Each test reproduces a concrete pre-fix failure: a crash / NaN / +inf / negative-variance on a valid
but degenerate input that previously slipped through with no error.
"""

import unittest
import warnings

import numpy as np

from mixle.stats import GaussianDistribution
from mixle.stats.bayes.dict_dirichlet import DictDirichletDistribution
from mixle.stats.bayes.normal_gamma import NormalGammaDistribution
from mixle.stats.multivariate.diagonal_gaussian import DiagonalGaussianDistribution
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution
from mixle.stats.univariate.continuous.gumbel import GumbelDistribution
from mixle.stats.univariate.continuous.log_gaussian import LogGaussianDistribution


class GumbelOverflowTest(unittest.TestCase):
    def test_far_left_tail_scalar_no_overflow(self):
        # math.exp(-z) raised OverflowError on the far-left tail; must return the -inf limit instead.
        g = GumbelDistribution(0.0, 1.0)
        self.assertEqual(g.log_density(-1000.0), -np.inf)

    def test_far_left_tail_seq_no_warning(self):
        g = GumbelDistribution(0.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any overflow warning becomes a test failure
            out = g.seq_log_density(np.array([-1000.0, 0.0, 5.0]))
        self.assertEqual(out[0], -np.inf)
        self.assertTrue(np.all(np.isfinite(out[1:])))

    def test_matches_scipy_on_normal_range(self):
        from scipy.stats import gumbel_r

        g = GumbelDistribution(0.5, 2.0)
        for x in (-3.0, -0.5, 0.5, 2.0, 6.0):
            self.assertAlmostEqual(g.log_density(x), gumbel_r.logpdf(x, loc=0.5, scale=2.0), places=10)


class DictDirichletBoundaryTest(unittest.TestCase):
    def test_mixed_boundary_is_not_nan(self):
        # alpha<1 zero (+inf) mixed with alpha>1 zero (-inf) gave +inf + -inf = NaN; +inf must win.
        d = DictDirichletDistribution({"a": 0.5, "b": 2.0, "c": 2.0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            v = d.log_density({"a": 0.0, "b": 0.0, "c": 1.0})
        self.assertEqual(v, np.inf)

    def test_boundary_precedence_matches_array_dirichlet(self):
        d = DictDirichletDistribution({"a": 0.5, "b": 2.0, "c": 2.0})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(d.log_density({"a": 0.0, "b": 0.5, "c": 0.5}), np.inf)  # only alpha<1 zero
            self.assertEqual(d.log_density({"a": 0.5, "b": 0.0, "c": 0.5}), -np.inf)  # only alpha>1 zero
            self.assertTrue(np.isfinite(d.log_density({"a": 0.2, "b": 0.3, "c": 0.5})))  # interior

    def test_symmetric_alpha_lt_one_boundary(self):
        d = DictDirichletDistribution(0.5)  # symmetric alpha < 1
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(d.log_density({"a": 0.0, "b": 0.4, "c": 0.6}), np.inf)
            self.assertTrue(np.isfinite(d.log_density({"a": 0.3, "b": 0.3, "c": 0.4})))


class ConjugateVarianceFloorTest(unittest.TestCase):
    def test_scalar_negative_scatter_no_crash(self):
        # near-constant / large-offset data makes the reduced-suff-stat scatter round negative, which
        # previously drove the conjugate variance negative -> ValueError. Must floor instead.
        est = GaussianDistribution(0.0, 1.0, prior=NormalGammaDistribution(0.0, 1e-12, 1e-9, 1e-12)).estimator()
        sum_x, n = 3.0e8, 3.0
        suff = (sum_x, sum_x * sum_x / n - 1.0e9, n, n)  # scatter forced negative
        d = est.estimate(None, suff)
        self.assertTrue(np.isfinite(d.sigma2) and d.sigma2 > 0.0)
        self.assertTrue(np.isfinite(d.log_density(sum_x / n)))

    def test_scalar_identical_data_no_crash(self):
        # exactly-identical data (true variance 0) through the conjugate fit must not crash or NaN.
        est = GaussianDistribution(0.0, 1.0, prior=NormalGammaDistribution(0.0, 1e-9, 1e-6, 1e-9)).estimator()
        acc = est.accumulator_factory().make()
        for _ in range(5):
            acc.update(1.0e8, 1.0, None)
        d = est.estimate(None, acc.value())
        self.assertTrue(np.isfinite(d.sigma2) and d.sigma2 > 0.0)

    def test_log_gaussian_negative_scatter_no_crash(self):
        # LogGaussianEstimator._estimate_conjugate mirrors GaussianEstimator's: same
        # cancellation-prone reduced-suff-stat scatter, previously unfloored. Unlike the scalar
        # Gaussian (whose constructor rejects sigma2 <= 0 outright), log_gaussian.py already had a
        # crude "new_sigma2 if > 0 else 1.0" fallback that swallows the negative value without
        # crashing -- so a plain isfinite/>0 check alone would pass even pre-fix. Assert the actual
        # (correctly floored) value instead, which differs sharply from the old constant-1.0 escape hatch.
        est = LogGaussianDistribution(0.0, 1.0, prior=NormalGammaDistribution(0.0, 1e-12, 1e-9, 1e-12)).estimator()
        sum_x, n = 3.0e8, 3.0
        suff = (sum_x, sum_x * sum_x / n - 1.0e9, n, n)  # scatter forced negative
        d = est.estimate(None, suff)
        self.assertTrue(np.isfinite(d.sigma2) and d.sigma2 > 0.0)
        self.assertTrue(np.isfinite(d.log_density(sum_x / n)))
        self.assertAlmostEqual(d.sigma2, 4999.999994998334, places=3)
        self.assertNotAlmostEqual(d.sigma2, 1.0, places=2)  # the old unguarded fallback's exact constant

    def test_log_gaussian_zero_denom_no_crash(self):
        # old_a=0.5 with no observations makes new_a - 0.5 == 0; dividing new_b/denom directly
        # produced +inf, which passed the old "> 0 else 1.0" guard unfiltered and crashed the
        # LogGaussianDistribution constructor (requires finite sigma2 > 0).
        est = LogGaussianDistribution(0.0, 1.0, prior=NormalGammaDistribution(0.0, 1.0, 0.5, 1.0)).estimator()
        suff = (0.0, 0.0, 0.0, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any divide-by-zero warning becomes a test failure
            d = est.estimate(None, suff)
        self.assertTrue(np.isfinite(d.sigma2) and d.sigma2 > 0.0)

    def test_diagonal_negative_scatter_no_nan(self):
        from mixle.stats import DiagonalGaussianDistribution
        from mixle.stats.bayes.multivariate_normal_gamma import MultivariateNormalGammaDistribution

        prior = MultivariateNormalGammaDistribution(
            np.zeros(2), 1e-12 * np.ones(2), 1e-9 * np.ones(2), 1e-12 * np.ones(2)
        )
        est = DiagonalGaussianDistribution(np.zeros(2), np.ones(2), prior=prior).estimator()
        sx = np.array([3.0e8, 1.0e7])
        suff = (sx, sx * sx / 3.0 - np.array([1.0e9, 1.0e6]), 3.0)  # both coords negative scatter
        d = est.estimate(None, suff)
        self.assertTrue(np.all(np.isfinite(d.log_density(sx / 3.0))))


class DiagonalGaussianConstructorValidationTest(unittest.TestCase):
    """DiagonalGaussianDistribution.__init__ previously accepted covar entries that were zero,
    negative, or non-finite with no validation (unlike the scalar GaussianDistribution), silently
    producing a NaN log-density instead of raising.
    """

    def test_zero_covar_entry_raises(self):
        with self.assertRaises(ValueError):
            DiagonalGaussianDistribution(mu=[0.0, 0.0], covar=[1.0, 0.0])

    def test_negative_covar_entry_raises(self):
        with self.assertRaises(ValueError):
            DiagonalGaussianDistribution(mu=[0.0, 0.0], covar=[1.0, -2.0])

    def test_nonfinite_covar_entry_raises(self):
        with self.assertRaises(ValueError):
            DiagonalGaussianDistribution(mu=[0.0, 0.0], covar=[1.0, np.nan])
        with self.assertRaises(ValueError):
            DiagonalGaussianDistribution(mu=[0.0, 0.0], covar=[1.0, np.inf])

    def test_positive_finite_covar_still_constructs(self):
        # guards against an overcorrection that rejects valid covariances too.
        d = DiagonalGaussianDistribution(mu=[0.0, 0.0], covar=[1.0, 2.0])
        self.assertTrue(np.isfinite(d.log_density(np.array([0.5, 0.5]))))


class DiagonalGaussianPseudoCountTest(unittest.TestCase):
    """DiagonalGaussianDistribution.estimator(pseudo_count=...) previously omitted
    suff_stat=(self.mu, self.covar), unlike the analogous MultivariateGaussianDistribution.estimator.
    DiagonalGaussianEstimator.estimate only smooths toward the prior mean/covar when
    prior_mu/prior_covar (set from suff_stat) are not None, so pseudo_count was a silent no-op.
    """

    def test_pseudo_count_pulls_estimate_toward_prior_mean(self):
        dg = DiagonalGaussianDistribution(mu=[10.0, 10.0], covar=[1.0, 1.0])
        est = dg.estimator(pseudo_count=1000.0)
        self.assertIsNotNone(est.prior_mu)
        self.assertIsNotNone(est.prior_covar)

        sum_x = np.array([0.0, 0.0])
        sum_xx = np.array([0.0, 0.0])
        d = est.estimate(None, (sum_x, sum_xx, 1.0))
        # pseudo_count=1000 vastly outweighs 1 real observation at 0, so the fitted mean should sit
        # very close to the prior mean [10, 10] rather than the raw MLE [0, 0].
        expected_mu = (1000.0 * 10.0) / (1.0 + 1000.0)
        np.testing.assert_allclose(d.mu, [expected_mu, expected_mu], atol=1e-10)
        self.assertFalse(np.allclose(d.mu, [0.0, 0.0]))

    def test_pseudo_count_matches_multivariate_sibling(self):
        # MultivariateGaussianDistribution.estimator already passes suff_stat=(mu, covar); the
        # diagonal case should behave identically on a diagonal covariance.
        mvg = MultivariateGaussianDistribution(mu=[10.0, 10.0], covar=[[1.0, 0.0], [0.0, 1.0]])
        dg = DiagonalGaussianDistribution(mu=[10.0, 10.0], covar=[1.0, 1.0])
        sum_x = np.array([0.0, 0.0])
        sum_xx = np.array([0.0, 0.0])
        d_mvg = mvg.estimator(pseudo_count=1000.0).estimate(None, (sum_x, sum_xx, 1.0))
        d_dg = dg.estimator(pseudo_count=1000.0).estimate(None, (sum_x, sum_xx, 1.0))
        np.testing.assert_allclose(d_dg.mu, d_mvg.mu, atol=1e-8)


class PackageDunderAllTest(unittest.TestCase):
    def test_import_star_resolves(self):
        # mixle.__all__ listed "parallel" and "src", which do not resolve -> from mixle import * crashed.
        import mixle

        self.assertNotIn("parallel", mixle.__all__)
        self.assertNotIn("src", mixle.__all__)
        ns: dict = {}
        exec("from mixle import *", ns)  # must not raise AttributeError
        for name in mixle.__all__:
            self.assertIn(name, ns, f"{name!r} in __all__ but not exported by import *")


if __name__ == "__main__":
    unittest.main()
