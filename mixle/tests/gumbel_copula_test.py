"""Gumbel copula core (mixle.stats.multivariate.gumbel_copula): Archimedean, UPPER-tail dependence -- the
complement to Clayton. Density integrates to ~1, sample-then-refit recovers theta, upper-tail dependence
shows up (mirror of Clayton's lower-tail), and it plugs into CopulaDistribution."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.combinator.copula import CopulaDistribution
from mixle.stats.multivariate.clayton_copula import ClaytonCopulaDistribution
from mixle.stats.multivariate.gumbel_copula import GumbelCopulaDistribution


def _unit_square_integral(core, m=300):
    g = np.linspace(1e-3, 1 - 1e-3, m)
    u, v = np.meshgrid(g, g)
    dens = np.exp(core.seq_log_density(np.c_[u.ravel(), v.ravel()])).reshape(m, m)
    return float(np.trapezoid(np.trapezoid(dens, g, axis=1), g))


class GumbelCopulaTest(unittest.TestCase):
    def test_density_integrates_to_one(self):
        self.assertAlmostEqual(_unit_square_integral(GumbelCopulaDistribution(2, 2.0)), 1.0, delta=0.03)

    def test_sample_then_refit_recovers_theta(self):
        c = GumbelCopulaDistribution(2, 2.0)
        s = c.sampler(0).sample(5000)
        est = c.estimator().estimate(None, (s, np.ones(len(s))))
        self.assertAlmostEqual(est.theta, 2.0, delta=0.3)

    def test_has_upper_tail_dependence(self):
        s = GumbelCopulaDistribution(2, 2.5).sampler(0).sample(12000)
        lower = np.mean((s[:, 0] < 0.05) & (s[:, 1] < 0.05))
        upper = np.mean((s[:, 0] > 0.95) & (s[:, 1] > 0.95))
        # dependence concentrated in the UPPER tail (the mirror of Clayton's lower-tail); the asymptotic
        # coefficient lambda_U = 2 - 2^(1/theta) ~ 0.68 here, so upper-tail co-occurrence clearly dominates.
        self.assertGreater(upper, 1.4 * lower)

    def test_theta_one_is_independence(self):
        c = GumbelCopulaDistribution(2, 1.0)
        np.testing.assert_allclose(c.seq_log_density(np.array([[0.3, 0.7], [0.1, 0.9]])), 0.0, atol=1e-9)

    def test_theta_below_one_is_clamped(self):
        self.assertEqual(GumbelCopulaDistribution(2, 0.5).theta, 1.0)  # theta < 1 is not a valid Gumbel

    def test_is_bivariate_only(self):
        with self.assertRaises(ValueError):
            GumbelCopulaDistribution(3, 2.0)

    def test_plugs_into_copula_distribution_and_beats_clayton_on_upper_tail_data(self):
        # Gumbel-generated (upper-tail) data with heterogeneous marginals
        u = GumbelCopulaDistribution(2, 2.5).sampler(0).sample(1000)
        x0 = spgamma.ppf(np.clip(u[:, 0], 1e-9, 1 - 1e-9), a=2.0, scale=2.0)
        x1 = norm.ppf(np.clip(u[:, 1], 1e-9, 1 - 1e-9), loc=5.0, scale=2.0)
        data = list(zip(x0.tolist(), x1.tolist()))

        def _fit_ll(core):
            proto = CopulaDistribution([st.GammaDistribution(1.0, 1.0), st.GaussianDistribution(0.0, 1.0)], core)
            fit = optimize(data, proto.estimator(), prev_estimate=proto, max_its=4, out=None)
            self.assertAlmostEqual(float(fit.marginals[1].mu), 5.0, delta=0.4)
            return float(np.sum(fit.seq_log_density(fit.dist_to_encoder().seq_encode(data))))

        ll_gumbel = _fit_ll(GumbelCopulaDistribution(2, 1.5))
        ll_clayton = _fit_ll(ClaytonCopulaDistribution(2, 0.5))
        self.assertGreater(ll_gumbel, ll_clayton)  # the upper-tail core wins on upper-tail data


if __name__ == "__main__":
    unittest.main()
