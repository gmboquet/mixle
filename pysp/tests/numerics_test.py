"""Numerical correctness tests for pysp.

Covers:
  - utility functions in pysp.utils.vector and pysp.utils.special,
  - log-density values checked against closed forms / scipy.stats references,
  - consistency between scalar log_density(), density(), and vectorized seq_log_density(),
  - parameter recovery of estimators on sampled data,
  - empirical KL divergence with degenerate (-inf) likelihood values.
"""
import unittest

import numpy as np
import scipy.special
import scipy.stats

from pysp.stats import (
    initialize, estimate, seq_encode,
    BetaDistribution, BinomialDistribution, CategoricalDistribution, CompositeDistribution, ConditionalDistribution,
    DiagonalGaussianDistribution, DirichletDistribution, ExponentialDistribution,
    GammaDistribution, GaussianDistribution, GeometricDistribution,
    IntegerCategoricalDistribution, LogGaussianDistribution, MixtureDistribution,
    MultivariateGaussianDistribution, OptionalDistribution, ParetoDistribution, PoissonDistribution,
    SequenceDistribution,
)
from pysp.stats.geometric import GeometricEstimator
from pysp.utils.estimation import empirical_kl_divergence
from pysp.utils.special import digammainv, logpdet, trigamma
from pysp.utils.vector import (
    log_posterior, log_posterior_sum, log_sum, make_pdf, posterior, row_choice,
    weighted_log_sum,
)


class VectorUtilsTestCase(unittest.TestCase):

    def test_log_sum_matches_scipy(self):
        rng = np.random.RandomState(1)
        for _ in range(10):
            x = rng.randn(20) * 50
            self.assertAlmostEqual(log_sum(x.copy()), scipy.special.logsumexp(x), places=10)

    def test_log_sum_extreme_values_no_overflow(self):
        x = np.array([-1e5, -1e5 + 1.0])
        self.assertAlmostEqual(log_sum(x.copy()), scipy.special.logsumexp(x), places=10)
        x = np.array([1e4, 1e4 - 1.0])
        self.assertTrue(np.isfinite(log_sum(x.copy())))

    def test_log_sum_all_neg_inf(self):
        self.assertEqual(log_sum(np.array([-np.inf, -np.inf])), -np.inf)

    def test_weighted_log_sum(self):
        x = np.log(np.array([0.2, 0.3, 0.5]))
        w = np.log(np.array([0.5, 0.25, 0.25]))
        expected = np.log(np.exp(x + w).sum())
        self.assertAlmostEqual(weighted_log_sum(x.copy(), w), expected, places=12)

    def test_posterior_normalizes(self):
        rng = np.random.RandomState(2)
        for _ in range(10):
            x = rng.randn(8) * 100
            p = posterior(x.copy())
            self.assertAlmostEqual(p.sum(), 1.0, places=10)
            ref = np.exp(x - scipy.special.logsumexp(x))
            self.assertTrue(np.allclose(p, ref))

    def test_posterior_degenerate_uniform(self):
        p = posterior(np.array([-np.inf, -np.inf, -np.inf]))
        self.assertTrue(np.allclose(p, np.ones(3) / 3))

    def test_posterior_log_sum_value(self):
        x = np.log(np.array([0.1, 0.4]))
        p, ls = posterior(x.copy(), log_sum=True)
        self.assertAlmostEqual(ls, np.log(0.5), places=12)

    def test_log_posterior_normalizes(self):
        x = np.random.RandomState(3).randn(6)
        lp = log_posterior(x.copy())
        self.assertAlmostEqual(np.exp(lp).sum(), 1.0, places=10)

    def test_log_posterior_sum_returns_tuple_when_degenerate(self):
        rv = log_posterior_sum(np.array([-np.inf, -np.inf]))
        self.assertIsInstance(rv, tuple)
        lp, mass = rv
        self.assertAlmostEqual(np.exp(lp).sum(), 1.0, places=10)

    def test_make_pdf_normalizes(self):
        log_p = np.log(np.array([0.2, 0.3, 0.5])) + 7.3  # unnormalized
        rv = make_pdf(log_p)
        self.assertAlmostEqual(np.exp(rv).sum(), 1.0, places=12)
        self.assertTrue(np.allclose(np.exp(rv), [0.2, 0.3, 0.5]))

    def test_make_pdf_all_neg_inf(self):
        rv = make_pdf(np.array([-np.inf, -np.inf]))
        self.assertTrue(np.allclose(rv, -np.log(2.0)))

    def test_row_choice_matches_inverse_cdf(self):
        for m in [2, 5, 17]:
            rng = np.random.RandomState(m)
            n = 500
            p_mat = rng.dirichlet(np.ones(m), size=n)
            idx = row_choice(p_mat, np.random.RandomState(m + 100))
            u = np.random.RandomState(m + 100).rand(n)
            ref = np.array([min(np.searchsorted(np.cumsum(p_mat[i]), u[i], side='right'), m - 1)
                            for i in range(n)])
            self.assertTrue(np.array_equal(idx, ref),
                            'row_choice disagrees with inverse-CDF sampling for m=%d' % m)


class SpecialUtilsTestCase(unittest.TestCase):

    def test_logpdet_full_rank(self):
        rng = np.random.RandomState(4)
        for d in [2, 3, 6]:
            a = rng.randn(d, d)
            x = a @ a.T + np.eye(d)
            self.assertAlmostEqual(logpdet(x), np.linalg.slogdet(x)[1], places=8)

    def test_logpdet_singular_uses_nonzero_eigs(self):
        # rank-1 matrix: pseudo-determinant is its nonzero eigenvalue
        v = np.array([3.0, 4.0])
        x = np.outer(v, v)
        self.assertAlmostEqual(logpdet(x), np.log(25.0), places=8)

    def test_digammainv_roundtrip_scalar(self):
        for v in [0.05, 0.5, 1.0, 3.7, 50.0]:
            y = scipy.special.digamma(v)
            self.assertAlmostEqual(digammainv(y), v, places=6)

    def test_digammainv_roundtrip_array(self):
        v = np.array([0.1, 0.9, 2.5, 10.0, 100.0])
        y = scipy.special.digamma(v)
        rv = digammainv(y)
        self.assertTrue(np.allclose(rv, v, rtol=1e-6))

    def test_trigamma_matches_polygamma(self):
        v = np.array([0.2, 1.0, 4.5, 30.0])
        self.assertTrue(np.allclose(trigamma(v), scipy.special.polygamma(1, v)))


class ReferenceLogDensityTestCase(unittest.TestCase):
    """Compare log_density() against scipy.stats reference implementations."""

    def assert_close(self, a, b, msg=None):
        self.assertAlmostEqual(a, b, places=10, msg=msg)

    def test_gaussian(self):
        d = GaussianDistribution(mu=1.5, sigma2=4.0)
        for x in [-3.0, 0.0, 1.5, 10.0]:
            self.assert_close(d.log_density(x), scipy.stats.norm.logpdf(x, loc=1.5, scale=2.0))

    def test_log_gaussian(self):
        d = LogGaussianDistribution(mu=0.5, sigma2=2.25)
        for x in [0.1, 1.0, 5.0, 40.0]:
            ref = scipy.stats.lognorm.logpdf(x, s=1.5, scale=np.exp(0.5))
            self.assert_close(d.log_density(x), ref)

    def test_exponential(self):
        d = ExponentialDistribution(beta=3.0)
        for x in [0.01, 1.0, 9.0]:
            self.assert_close(d.log_density(x), scipy.stats.expon.logpdf(x, scale=3.0))

    def test_gamma(self):
        d = GammaDistribution(k=2.5, theta=1.7)
        for x in [0.1, 1.0, 6.0]:
            self.assert_close(d.log_density(x), scipy.stats.gamma.logpdf(x, a=2.5, scale=1.7))

    def test_poisson(self):
        d = PoissonDistribution(lam=4.2)
        for x in [0, 1, 4, 15]:
            self.assert_close(d.log_density(x), scipy.stats.poisson.logpmf(x, mu=4.2))

    def test_geometric(self):
        d = GeometricDistribution(p=0.3)
        for x in [1, 2, 7]:
            self.assert_close(d.log_density(x), scipy.stats.geom.logpmf(x, p=0.3))

    def test_binomial(self):
        d = BinomialDistribution(p=0.35, n=12)
        for x in [0, 3, 12]:
            self.assert_close(d.log_density(x), scipy.stats.binom.logpmf(x, n=12, p=0.35))

    def test_dirichlet(self):
        alpha = np.array([1.1, 2.8, 4.5])
        d = DirichletDistribution(alpha)
        x = np.array([0.2, 0.3, 0.5])
        self.assert_close(d.log_density(x), scipy.stats.dirichlet.logpdf(x, alpha))

    def test_multivariate_gaussian(self):
        mu = np.array([1.0, -2.0])
        covar = np.array([[2.0, 0.6], [0.6, 1.0]])
        d = MultivariateGaussianDistribution(mu, covar)
        for x in [np.array([0.0, 0.0]), np.array([3.0, -1.0])]:
            ref = scipy.stats.multivariate_normal.logpdf(x, mean=mu, cov=covar)
            self.assert_close(d.log_density(x), ref)

    def test_diagonal_gaussian(self):
        mu = [1.0, -1.0, 0.5]
        s2 = [1.0, 4.0, 0.25]
        d = DiagonalGaussianDistribution(mu, s2)
        x = np.array([0.0, 1.0, 1.0])
        ref = scipy.stats.norm.logpdf(x, loc=mu, scale=np.sqrt(s2)).sum()
        self.assert_close(d.log_density(x), ref)

    def test_categorical(self):
        pm = {'a': 0.4, 'b': 0.35, 'c': 0.25}
        d = CategoricalDistribution(pm)
        for k, v in pm.items():
            self.assert_close(d.log_density(k), np.log(v))

    def test_mixture_log_sum_exp(self):
        comps = [GaussianDistribution(mu=-100.0, sigma2=1.0), GaussianDistribution(mu=100.0, sigma2=1.0)]
        d = MixtureDistribution(comps, [0.5, 0.5])
        # component densities underflow individually; LSE must keep this finite & correct
        ref = scipy.stats.norm.logpdf(100.0, loc=100.0, scale=1.0) + np.log(0.5)
        self.assert_close(d.log_density(100.0), ref)

    def test_invalid_support_returns_negative_infinity(self):
        invalid_cases = [
            (PoissonDistribution(lam=2.0), [-1, 1.5, 'bad']),
            (GeometricDistribution(p=0.4), [0, -1, 1.5, 'bad']),
            (BinomialDistribution(p=0.4, n=5), [-1, 6, 2.5, 'bad']),
            (BetaDistribution(a=2.0, b=3.0), [0.0, 1.0, np.nan, 'bad']),
            (GammaDistribution(k=2.0, theta=3.0), [0.0, -1.0, np.nan, 'bad']),
            (LogGaussianDistribution(mu=0.0, sigma2=1.0), [0.0, -1.0]),
            (ExponentialDistribution(beta=2.0), [-1.0]),
            (ParetoDistribution(xm=2.0, alpha=3.0), [1.0, np.nan, 'bad']),
            (DirichletDistribution([2.0, 3.0, 4.0]), [[0.2, 0.3, 0.4], [-0.1, 0.6, 0.5]]),
        ]
        for dist, values in invalid_cases:
            for x in values:
                self.assertEqual(dist.log_density(x), -np.inf, msg='%s at %s' % (dist, x))

        self.assertEqual(LogGaussianDistribution(0.0, 1.0).density(0.0), 0.0)

    def test_invalid_distribution_parameters_raise(self):
        invalid = [
            (GaussianDistribution, (0.0, 0.0)),
            (GaussianDistribution, (np.nan, 1.0)),
            (GammaDistribution, (0.0, 1.0)),
            (GammaDistribution, (1.0, np.inf)),
            (ExponentialDistribution, (0.0,)),
            (PoissonDistribution, (0.0,)),
            (GeometricDistribution, (0.0,)),
            (BinomialDistribution, (0.4, 1.5)),
            (BetaDistribution, (0.0, 1.0)),
        ]
        for cls, args in invalid:
            with self.assertRaises(ValueError, msg='%s%r' % (cls.__name__, args)):
                cls(*args)

    def test_count_encoders_reject_fractional_counts(self):
        for dist in [PoissonDistribution(2.0), GeometricDistribution(0.4), BinomialDistribution(0.3, 5)]:
            with self.assertRaises(ValueError, msg=str(dist)):
                dist.dist_to_encoder().seq_encode([1.5])

    def test_count_seq_log_density_handles_out_of_support_values(self):
        b = BinomialDistribution(0.4, 5)
        enc_b = b.dist_to_encoder().seq_encode([0, 5, 6])
        np.testing.assert_array_equal(np.isneginf(b.seq_log_density(enc_b)), [False, False, True])

        p = PoissonDistribution(2.0)
        enc_p = (np.asarray([0.0, 1.0, 1.5]), scipy.special.gammaln(np.asarray([1.0, 2.0, 2.5])))
        np.testing.assert_array_equal(np.isneginf(p.seq_log_density(enc_p)), [False, False, True])

        g = GeometricDistribution(0.4)
        np.testing.assert_array_equal(np.isneginf(g.seq_log_density(np.asarray([0.0, 1.0, 1.5, 2.0]))),
                                      [True, False, True, False])

    def test_conditional_seq_log_density_without_default_matches_scalar(self):
        d = ConditionalDistribution({'seen': GaussianDistribution(0.0, 1.0)}, default_dist=None)
        data = [('seen', 0.25), ('missing', 0.25)]
        enc = d.dist_to_encoder().seq_encode(data)
        np.testing.assert_allclose(d.seq_log_density(enc), np.asarray([d.log_density(x) for x in data]))


class ConsistencyTestCase(unittest.TestCase):
    """Cross-checks between density(), log_density(), and seq_log_density()."""

    def battery(self):
        return [
            GaussianDistribution(mu=1.0, sigma2=2.0),
            LogGaussianDistribution(mu=0.2, sigma2=1.5),
            ExponentialDistribution(beta=2.5),
            GammaDistribution(k=3.0, theta=2.0),
            PoissonDistribution(lam=6.0),
            GeometricDistribution(p=0.4),
            BinomialDistribution(p=0.3, n=8),
            CategoricalDistribution({'a': 0.4, 'b': 0.3, 'c': 0.2, 'd': 0.1}),
            IntegerCategoricalDistribution(0, [0.1, 0.4, 0.3, 0.2]),
            DiagonalGaussianDistribution([1.8, 4.3, -1.5], [1.1, 4.8, 9.1]),
            MultivariateGaussianDistribution([1.0, 2.0], [[2.0, 0.5], [0.5, 1.0]]),
            DirichletDistribution([1.1, 2.8, 4.5]),
            MixtureDistribution([GaussianDistribution(mu=-3.0, sigma2=1.0),
                                 GaussianDistribution(mu=3.0, sigma2=2.0)], [0.4, 0.6]),
            CompositeDistribution((ExponentialDistribution(3.1), PoissonDistribution(3.2))),
            SequenceDistribution(GeometricDistribution(0.8),
                                 len_dist=CategoricalDistribution({5: 1.0})),
            OptionalDistribution(PoissonDistribution(4.7), p=0.1),
        ]

    def test_seq_log_density_matches_scalar(self):
        for dist in self.battery():
            data = dist.sampler(seed=10).sample(size=50)
            enc = dist.dist_to_encoder().seq_encode(data)
            seq_ll = dist.seq_log_density(enc)
            scalar_ll = np.array([dist.log_density(x) for x in data])
            self.assertTrue(
                np.allclose(seq_ll, scalar_ll, rtol=1e-10, atol=1e-12),
                'seq/scalar log-density mismatch for %s: max diff %s' % (
                    str(dist), np.max(np.abs(seq_ll - scalar_ll))))

    def test_density_matches_exp_log_density(self):
        for dist in self.battery():
            if not hasattr(dist, 'density'):
                continue
            data = dist.sampler(seed=11).sample(size=20)
            for x in data:
                ld = dist.log_density(x)
                self.assertAlmostEqual(
                    dist.density(x), np.exp(ld), places=10,
                    msg='density != exp(log_density) for %s at %s' % (str(dist), repr(x)))

    def test_sampler_reproducible(self):
        for dist in self.battery():
            s1 = dist.sampler(seed=42).sample(size=10)
            s2 = dist.sampler(seed=42).sample(size=10)
            self.assertEqual(list(map(str, s1)), list(map(str, s2)),
                             'sampler not reproducible for %s' % str(dist))


class EstimationRecoveryTestCase(unittest.TestCase):
    """Sample from a known model, estimate, and check parameter recovery."""

    @staticmethod
    def fit(dist, n=20000, seed=5):
        data = dist.sampler(seed=seed).sample(size=n)
        est = dist.estimator()
        init = initialize(data, est, rng=np.random.RandomState(1), p=1.0)
        return estimate(data, est, init)

    def test_gaussian_recovery(self):
        fit = self.fit(GaussianDistribution(mu=2.0, sigma2=9.0))
        self.assertAlmostEqual(fit.mu, 2.0, delta=0.1)
        self.assertAlmostEqual(fit.sigma2, 9.0, delta=0.3)

    def test_gaussian_variance_stability_large_offset(self):
        # E[x^2]-mean^2 cancellation check: small variance on a large mean
        fit = self.fit(GaussianDistribution(mu=1.0e6, sigma2=1.0))
        self.assertGreater(fit.sigma2, 0.5)
        self.assertLess(fit.sigma2, 1.5)

    def test_exponential_recovery(self):
        fit = self.fit(ExponentialDistribution(beta=4.0))
        self.assertAlmostEqual(fit.beta, 4.0, delta=0.2)

    def test_poisson_recovery(self):
        fit = self.fit(PoissonDistribution(lam=3.5))
        self.assertAlmostEqual(fit.lam, 3.5, delta=0.15)

    def test_geometric_recovery(self):
        fit = self.fit(GeometricDistribution(p=0.3))
        self.assertAlmostEqual(fit.p, 0.3, delta=0.02)

    def test_gamma_recovery(self):
        fit = self.fit(GammaDistribution(k=2.0, theta=3.0))
        self.assertAlmostEqual(fit.k, 2.0, delta=0.2)
        self.assertAlmostEqual(fit.theta, 3.0, delta=0.4)

    def test_categorical_recovery(self):
        truth = {'a': 0.5, 'b': 0.3, 'c': 0.2}
        fit = self.fit(CategoricalDistribution(truth))
        for k, v in truth.items():
            self.assertAlmostEqual(np.exp(fit.log_density(k)), v, delta=0.02)

    def test_geometric_estimator_suff_stat_clamped_to_unit_interval(self):
        self.assertEqual(GeometricEstimator(pseudo_count=1.0, suff_stat=0.7).suff_stat, 0.7)
        self.assertEqual(GeometricEstimator(pseudo_count=1.0, suff_stat=1.7).suff_stat, 1.0)
        self.assertEqual(GeometricEstimator(pseudo_count=1.0, suff_stat=-0.3).suff_stat, 0.0)


class EmpiricalKLTestCase(unittest.TestCase):

    def test_identical_distributions_zero_kl(self):
        d = GaussianDistribution(mu=0.0, sigma2=1.0)
        data = d.sampler(seed=8).sample(size=200)
        enc = seq_encode(data, encoder=d.dist_to_encoder())
        kl, bad1, bad2 = empirical_kl_divergence(d, d, enc)
        self.assertAlmostEqual(kl, 0.0, places=12)
        self.assertEqual(bad1, 0)
        self.assertEqual(bad2, 0)

    def test_handles_neg_inf_log_densities(self):
        d1 = CategoricalDistribution({'a': 0.5, 'b': 0.5})
        d2 = CategoricalDistribution({'a': 1.0}, default_value=0.0)
        data = ['a', 'b', 'a', 'a']
        enc = seq_encode(data, encoder=d1.dist_to_encoder())
        kl, bad1, bad2 = empirical_kl_divergence(d1, d2, enc)
        self.assertTrue(np.isfinite(kl))
        self.assertEqual(bad1, 0)
        self.assertEqual(bad2, 1)


if __name__ == '__main__':
    unittest.main()
