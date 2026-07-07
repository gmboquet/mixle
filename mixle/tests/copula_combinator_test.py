"""CopulaDistribution (mixle.stats.combinator.copula): glue arbitrary marginals to a copula core via
Sklar's theorem, fit by IFM. Recovers a known correlation + heterogeneous marginals, beats independence,
samples with the right marginals and rank dependence, and composes as a mixle five-piece distribution."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm, spearmanr

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.combinator.copula import CopulaDistribution
from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution


def _correlated_heterogeneous(seed, n=800, r=0.7):
    # latent Gaussian dependence pushed through a Gamma(2,2) marginal and a Gaussian(5,2) marginal
    rng = np.random.RandomState(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, r], [r, 1.0]], size=n)
    u = norm.cdf(z)
    x0 = spgamma.ppf(u[:, 0], a=2.0, scale=2.0)
    x1 = norm.ppf(u[:, 1], loc=5.0, scale=2.0)
    return list(zip(x0.tolist(), x1.tolist()))


def _proto():
    return CopulaDistribution(
        [st.GammaDistribution(1.0, 1.0), st.GaussianDistribution(0.0, 1.0)],
        GaussianCopulaDistribution(np.eye(2)),
    )


class CopulaDistributionTest(unittest.TestCase):
    def test_ifm_recovers_the_correlation_and_the_marginals(self):
        data = _correlated_heterogeneous(0, r=0.7)
        fit = optimize(data, _proto().estimator(), prev_estimate=_proto(), max_its=5, out=None)
        self.assertAlmostEqual(float(fit.copula.corr[0, 1]), 0.7, delta=0.08)
        # Gamma(shape k=2, scale theta=2); GaussianDistribution stores (mean mu, variance sigma2) = (5, 4)
        self.assertAlmostEqual(float(fit.marginals[0].k), 2.0, delta=0.4)
        self.assertAlmostEqual(float(fit.marginals[1].mu), 5.0, delta=0.3)

    def test_beats_the_independence_copula_on_dependent_data(self):
        data = _correlated_heterogeneous(1, r=0.7)
        fit = optimize(data, _proto().estimator(), prev_estimate=_proto(), max_its=5, out=None)
        indep = CopulaDistribution(fit.marginals, GaussianCopulaDistribution(np.eye(2)))
        ll_fit = float(np.sum(fit.seq_log_density(fit.dist_to_encoder().seq_encode(data))))
        ll_indep = float(np.sum(indep.seq_log_density(indep.dist_to_encoder().seq_encode(data))))
        self.assertGreater(ll_fit, ll_indep + 50.0)  # dependence is real and worth many nats

    def test_scalar_log_density_matches_the_sklar_decomposition(self):
        cop = CopulaDistribution(
            [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
            GaussianCopulaDistribution(np.array([[1.0, 0.5], [0.5, 1.0]])),
        )
        x = (3.0, 4.5)
        u = np.clip([cop.marginals[0].cdf(x[0]), cop.marginals[1].cdf(x[1])], 1e-12, 1 - 1e-12)
        expected = cop.marginals[0].log_density(x[0]) + cop.marginals[1].log_density(x[1]) + cop.copula.log_density(u)
        self.assertAlmostEqual(cop.log_density(x), expected, places=10)

    def test_seq_and_scalar_log_density_agree(self):
        data = _correlated_heterogeneous(2, n=50)
        cop = CopulaDistribution(
            [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
            GaussianCopulaDistribution(np.array([[1.0, 0.6], [0.6, 1.0]])),
        )
        seq = cop.seq_log_density(cop.dist_to_encoder().seq_encode(data))
        scalar = np.array([cop.log_density(x) for x in data])
        np.testing.assert_allclose(seq, scalar, atol=1e-9)

    def test_sampling_has_the_right_marginals_and_dependence(self):
        cop = CopulaDistribution(
            [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
            GaussianCopulaDistribution(np.array([[1.0, 0.8], [0.8, 1.0]])),
        )
        s = np.array(cop.sampler(0).sample(3000))
        self.assertAlmostEqual(s[:, 0].mean(), 4.0, delta=0.4)  # Gamma mean = shape*scale = 4
        self.assertAlmostEqual(s[:, 1].mean(), 5.0, delta=0.3)  # Gaussian mean = 5
        rho, _ = spearmanr(s[:, 0], s[:, 1])
        self.assertGreater(rho, 0.6)  # strong positive rank dependence, as the copula prescribes

    def test_requires_at_least_two_marginals(self):
        with self.assertRaises(ValueError):
            CopulaDistribution([st.GaussianDistribution(0.0, 1.0)], GaussianCopulaDistribution(np.eye(1)))

    def test_composes_inside_a_mixture(self):
        # two dependence regimes: a positively- and a negatively-correlated cluster, same marginals
        rng = np.random.RandomState(3)
        a = _correlated_heterogeneous(10, n=300, r=0.8)
        b = _correlated_heterogeneous(11, n=300, r=-0.8)
        data = a + b
        rng.shuffle(data)
        comp = [
            CopulaDistribution(
                [st.GammaDistribution(2.0, 2.0), st.GaussianDistribution(5.0, 4.0)],
                GaussianCopulaDistribution(np.array([[1.0, s], [s, 1.0]])),
            )
            for s in (0.5, -0.5)
        ]
        mix = st.MixtureDistribution(comp, [0.5, 0.5])
        fit = optimize(data, mix.estimator(), prev_estimate=mix, max_its=8, out=None)
        corrs = sorted(float(c.copula.corr[0, 1]) for c in fit.components)
        self.assertLess(corrs[0], -0.3)  # one regime recovered as negatively dependent
        self.assertGreater(corrs[1], 0.3)  # the other as positively dependent


if __name__ == "__main__":
    unittest.main()
