"""F-9 completion: entropy() for the six families PR #434 left unfinished.

PR #434 (2026-07-13) added closed-form entropy() to StudentT, Poisson, GEV, GPD, and
InverseGaussian (see completeness_review_fixes_test.py) but its own commit message overstated
the ledger item: audit/CODEBASE_REVIEW_LEDGER.md F-9 also names SkewNormal, NegativeBinomial,
Rician, Nakagami, Skellam, and LogSeries, none of which gained entropy() in that PR. This module
finishes F-9 for those six.

Every check here is independent of the closed-form derivation used in the source: the
"numerical reference" tests recompute the entropy integral/series from scratch (fresh quadrature
or truncated summation, not a call into the production formula), and the "monte carlo" tests use
only sampler() + log_density()/seq_log_density() -- never entropy() itself -- so a wrong constant,
sign, or parameterization in the shipped formula cannot pass silently. This matters here: scipy's
own generic ``.entropy()`` (a numerical fallback via ``rv_discrete.expect``) turns out to raise
"sum did not converge" and returns a silently wrong value for NegativeBinomial/LogSeries at
strongly over-dispersed parameters (see test_extreme_parameters_where_scipy_entropy_is_unreliable),
which is exactly the failure mode a pure scipy cross-check (as entropy_methods_test.py otherwise
uses) would miss.
"""

import math
import unittest

import numpy as np
from scipy import integrate
from scipy.special import gammaln, ive

from mixle.stats.univariate.continuous.nakagami import NakagamiDistribution
from mixle.stats.univariate.continuous.rician import RicianDistribution
from mixle.stats.univariate.continuous.skew_normal import SkewNormalDistribution
from mixle.stats.univariate.discrete.logseries import LogSeriesDistribution
from mixle.stats.univariate.discrete.negative_binomial import NegativeBinomialDistribution
from mixle.stats.univariate.discrete.skellam import SkellamDistribution

MC_N = 300_000  # at this size the plug-in MC estimator's standard error is ~0.001-0.003 nats
MC_SEED = 0
MC_DELTA = 0.02  # >= 6x the empirically observed standard error for every case below


def _mc_entropy(dist, n: int = MC_N, seed: int = MC_SEED) -> float:
    """Plug-in Monte Carlo entropy estimate -mean(log f(X)) for X ~ dist.sampler().

    Independent of ``dist.entropy()``: it only exercises the (separately tested) sampler and
    log-density paths, so it cannot be fooled by a bug that is only inside entropy() itself.
    """
    data = np.asarray(dist.sampler(seed=seed).sample(n))
    enc = dist.dist_to_encoder().seq_encode(data)
    logp = np.asarray(dist.seq_log_density(enc), dtype=np.float64)
    return float(-np.mean(logp))


def _nb_log_pmf(k: np.ndarray, r: float, p: float) -> np.ndarray:
    """Negative-binomial log-pmf, coded fresh from the textbook definition (Johnson, Kemp & Kotz)."""
    return gammaln(k + r) - gammaln(r) - gammaln(k + 1.0) + r * math.log(p) + k * math.log1p(-p)


def _skellam_log_pmf(k: np.ndarray, mu1: float, mu2: float) -> np.ndarray:
    """Skellam log-pmf, coded fresh from the textbook definition (Skellam, 1946)."""
    log_ratio_half = 0.5 * (math.log(mu1) - math.log(mu2))
    sqrt_diff_sq = (math.sqrt(mu1) - math.sqrt(mu2)) ** 2
    two_sqrt_prod = 2.0 * math.sqrt(mu1 * mu2)
    with np.errstate(divide="ignore"):
        log_bessel = np.log(ive(np.abs(k), two_sqrt_prod))
    return -sqrt_diff_sq + k * log_ratio_half + log_bessel


def _logseries_log_pmf(k: np.ndarray, p: float) -> np.ndarray:
    """Log-series log-pmf, coded fresh from the textbook definition (Fisher, Corbet & Williams)."""
    return k * math.log(p) - np.log(k) - math.log(-math.log1p(-p))


class SkewNormalEntropyTest(unittest.TestCase):
    def test_matches_independent_quadrature(self):
        for loc, scale, alpha in [
            (0.0, 1.0, 0.0),
            (0.5, 1.5, 2.0),
            (-1.0, 0.7, -3.0),
            (2.0, 2.0, 20.0),
            (0.0, 1.0, -0.3),
        ]:
            d = SkewNormalDistribution(loc, scale, alpha)

            def f(x, d=d):
                logf = d.log_density(x)
                return -math.exp(logf) * logf

            ref, _ = integrate.quad(f, -np.inf, np.inf, limit=200)
            with self.subTest(loc=loc, scale=scale, alpha=alpha):
                self.assertAlmostEqual(d.entropy(), ref, places=6)

    def test_matches_monte_carlo(self):
        for loc, scale, alpha in [(0.5, 1.5, 2.0), (2.0, 2.0, 20.0), (-1.0, 0.7, -3.0)]:
            d = SkewNormalDistribution(loc, scale, alpha)
            with self.subTest(loc=loc, scale=scale, alpha=alpha):
                self.assertAlmostEqual(d.entropy(), _mc_entropy(d), delta=MC_DELTA)


class NegativeBinomialEntropyTest(unittest.TestCase):
    def test_matches_independent_truncated_sum(self):
        for r, p in [(1.0, 0.5), (5.0, 0.3), (20.0, 0.6), (2.0, 0.05), (0.5, 0.2)]:
            d = NegativeBinomialDistribution(r, p)
            kmax = int(d.quantile(1.0 - 1.0e-16)) + 200
            k = np.arange(kmax + 1, dtype=np.float64)
            lp = _nb_log_pmf(k, r, p)
            ref = float(-np.sum(np.exp(lp) * lp))
            with self.subTest(r=r, p=p):
                self.assertAlmostEqual(d.entropy(), ref, places=6)

    def test_matches_monte_carlo(self):
        for r, p in [(5.0, 0.3), (2.0, 0.05)]:
            d = NegativeBinomialDistribution(r, p)
            with self.subTest(r=r, p=p):
                self.assertAlmostEqual(d.entropy(), _mc_entropy(d), delta=MC_DELTA)


class RicianEntropyTest(unittest.TestCase):
    def test_matches_independent_quadrature(self):
        for nu, sigma in [(0.0, 1.0), (0.0, 2.5), (1.0, 1.0), (3.0, 1.0), (5.0, 2.0), (0.1, 0.5)]:
            d = RicianDistribution(nu, sigma)

            def f(x, d=d):
                logf = d.log_density(x)
                return 0.0 if not np.isfinite(logf) else -math.exp(logf) * logf

            ref, _ = integrate.quad(f, 0.0, np.inf, limit=200)
            with self.subTest(nu=nu, sigma=sigma):
                self.assertAlmostEqual(d.entropy(), ref, places=6)

    def test_nu_zero_matches_rayleigh_closed_form(self):
        for sigma in [1.0, 2.5, 0.3]:
            d = RicianDistribution(0.0, sigma)
            rayleigh = 1.0 + math.log(sigma / math.sqrt(2.0)) + np.euler_gamma / 2.0
            with self.subTest(sigma=sigma):
                self.assertAlmostEqual(d.entropy(), rayleigh, places=10)

    def test_matches_monte_carlo(self):
        for nu, sigma in [(3.0, 1.0), (5.0, 2.0)]:
            d = RicianDistribution(nu, sigma)
            with self.subTest(nu=nu, sigma=sigma):
                self.assertAlmostEqual(d.entropy(), _mc_entropy(d), delta=MC_DELTA)


class NakagamiEntropyTest(unittest.TestCase):
    def test_matches_independent_quadrature(self):
        for m, omega in [(0.5, 1.0), (1.0, 1.0), (1.0, 3.0), (2.5, 2.0), (5.0, 0.7), (0.7, 4.5)]:
            d = NakagamiDistribution(m, omega)

            def f(x, d=d):
                logf = d.log_density(x)
                return -math.exp(logf) * logf

            ref, _ = integrate.quad(f, 0.0, np.inf, limit=200)
            with self.subTest(m=m, omega=omega):
                self.assertAlmostEqual(d.entropy(), ref, places=6)

    def test_m_one_matches_rayleigh_closed_form(self):
        # Nakagami(m=1, omega) is exactly Rayleigh(sigma=sqrt(omega/2)).
        for omega in [1.0, 3.0, 0.5]:
            d = NakagamiDistribution(1.0, omega)
            sigma = math.sqrt(omega / 2.0)
            rayleigh = 1.0 + math.log(sigma / math.sqrt(2.0)) + np.euler_gamma / 2.0
            with self.subTest(omega=omega):
                self.assertAlmostEqual(d.entropy(), rayleigh, places=10)

    def test_matches_monte_carlo(self):
        for m, omega in [(2.5, 2.0), (0.7, 4.5)]:
            d = NakagamiDistribution(m, omega)
            with self.subTest(m=m, omega=omega):
                self.assertAlmostEqual(d.entropy(), _mc_entropy(d), delta=MC_DELTA)


class SkellamEntropyTest(unittest.TestCase):
    def test_matches_independent_truncated_sum(self):
        for mu1, mu2 in [(2.0, 1.0), (5.0, 5.0), (0.5, 8.0), (15.0, 3.0)]:
            d = SkellamDistribution(mu1, mu2)
            tol = 1.0e-16
            lo = int(d.quantile(tol)) - 50
            hi = int(d.quantile(1.0 - tol)) + 50
            k = np.arange(lo, hi + 1, dtype=np.float64)
            lp = _skellam_log_pmf(k, mu1, mu2)
            finite = np.isfinite(lp)
            ref = float(-np.sum(np.exp(lp[finite]) * lp[finite]))
            with self.subTest(mu1=mu1, mu2=mu2):
                self.assertAlmostEqual(d.entropy(), ref, places=6)

    def test_matches_monte_carlo(self):
        for mu1, mu2 in [(5.0, 5.0), (15.0, 3.0)]:
            d = SkellamDistribution(mu1, mu2)
            with self.subTest(mu1=mu1, mu2=mu2):
                self.assertAlmostEqual(d.entropy(), _mc_entropy(d), delta=MC_DELTA)


class LogSeriesEntropyTest(unittest.TestCase):
    def test_matches_independent_truncated_sum(self):
        for p in [0.1, 0.5, 0.7, 0.9]:
            d = LogSeriesDistribution(p)
            kmax = int(d.quantile(1.0 - 1.0e-16)) + 200
            k = np.arange(1, kmax + 1, dtype=np.float64)
            lp = _logseries_log_pmf(k, p)
            ref = float(-np.sum(np.exp(lp) * lp))
            with self.subTest(p=p):
                self.assertAlmostEqual(d.entropy(), ref, places=6)

    def test_matches_monte_carlo(self):
        for p in [0.5, 0.9]:
            d = LogSeriesDistribution(p)
            with self.subTest(p=p):
                self.assertAlmostEqual(d.entropy(), _mc_entropy(d), delta=MC_DELTA)


class ExtremeParametersWhereScipyEntropyIsUnreliableTest(unittest.TestCase):
    """scipy.stats.{nbinom,logser}.entropy() falls back to a generic numerical series (via
    ``rv_discrete.expect``) that raises "sum did not converge" and returns a value off by more
    than 1.5 nats at these parameters -- so these two cases are checked only against the
    from-scratch truncated sum and Monte Carlo, which agree with each other and with mixle.
    """

    def test_negative_binomial_heavy_tailed(self):
        d = NegativeBinomialDistribution(0.5, 0.01)
        kmax = int(d.quantile(1.0 - 1.0e-16)) + 200
        k = np.arange(kmax + 1, dtype=np.float64)
        lp = _nb_log_pmf(k, 0.5, 0.01)
        ref = float(-np.sum(np.exp(lp) * lp))
        self.assertAlmostEqual(d.entropy(), ref, places=6)
        self.assertAlmostEqual(d.entropy(), _mc_entropy(d, n=500_000), delta=MC_DELTA)

    def test_logseries_near_boundary(self):
        d = LogSeriesDistribution(0.99)
        kmax = int(d.quantile(1.0 - 1.0e-16)) + 200
        k = np.arange(1, kmax + 1, dtype=np.float64)
        lp = _logseries_log_pmf(k, 0.99)
        ref = float(-np.sum(np.exp(lp) * lp))
        self.assertAlmostEqual(d.entropy(), ref, places=6)
        self.assertAlmostEqual(d.entropy(), _mc_entropy(d, n=500_000), delta=MC_DELTA)


if __name__ == "__main__":
    unittest.main()
