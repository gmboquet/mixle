"""Regression guards for silent-corruption numerical edge cases (cleanup audit, Phase 1).

Each test reproduces a concrete pre-fix failure: a crash / NaN / +inf / negative-variance on a valid
but degenerate input that previously slipped through with no error.
"""

import unittest
import warnings

import numpy as np

from pysp.stats import GaussianDistribution
from pysp.stats.bayes.dict_dirichlet import DictDirichletDistribution
from pysp.stats.bayes.normal_gamma import NormalGammaDistribution
from pysp.stats.univariate.continuous.gumbel import GumbelDistribution


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

    def test_diagonal_negative_scatter_no_nan(self):
        from pysp.stats import DiagonalGaussianDistribution
        from pysp.stats.bayes.multivariate_normal_gamma import MultivariateNormalGammaDistribution

        prior = MultivariateNormalGammaDistribution(
            np.zeros(2), 1e-12 * np.ones(2), 1e-9 * np.ones(2), 1e-12 * np.ones(2)
        )
        est = DiagonalGaussianDistribution(np.zeros(2), np.ones(2), prior=prior).estimator()
        sx = np.array([3.0e8, 1.0e7])
        suff = (sx, sx * sx / 3.0 - np.array([1.0e9, 1.0e6]), 3.0)  # both coords negative scatter
        d = est.estimate(None, suff)
        self.assertTrue(np.all(np.isfinite(d.log_density(sx / 3.0))))


class PackageDunderAllTest(unittest.TestCase):
    def test_import_star_resolves(self):
        # pysp.__all__ listed "parallel" and "src", which do not resolve -> from pysp import * crashed.
        import pysp

        self.assertNotIn("parallel", pysp.__all__)
        self.assertNotIn("src", pysp.__all__)
        ns: dict = {}
        exec("from pysp import *", ns)  # must not raise AttributeError
        for name in pysp.__all__:
            self.assertIn(name, ns, f"{name!r} in __all__ but not exported by import *")


if __name__ == "__main__":
    unittest.main()
