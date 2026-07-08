"""Automatic inference reaches a COPULA for dependent continuous records (mixle.inference.estimation).

When flat all-continuous records have dependence and heterogeneous marginals, optimize(data) with no
estimator recommends a CopulaDistribution over the auto-detected marginals + a Gaussian dependence core --
but only when it beats the independent composite by BIC. Independent data stays a composite; the historical
Bayesian-network path for non-continuous records is untouched."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm

from mixle.inference import optimize
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.copula import CopulaDistribution


def _dependent_heterogeneous(seed, n=1500, r=0.75):
    rng = np.random.RandomState(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, r], [r, 1.0]], size=n)
    u = norm.cdf(z)
    x0 = spgamma.ppf(u[:, 0], a=2.0, scale=2.0)  # Gamma marginal
    x1 = norm.ppf(u[:, 1], loc=5.0, scale=2.0)  # Gaussian marginal
    return [(float(a), float(b)) for a, b in zip(x0, x1)]


class AutomaticCopulaTest(unittest.TestCase):
    def test_dependent_continuous_records_get_a_copula(self):
        model = optimize(_dependent_heterogeneous(0), out=None)
        self.assertIsInstance(model, CopulaDistribution)
        self.assertAlmostEqual(float(model.copula.corr[0, 1]), 0.75, delta=0.1)  # recovers the dependence
        # marginals are the auto-detected families, not forced Gaussian
        self.assertEqual(type(model.marginals[0]).__name__, "GammaDistribution")
        self.assertEqual(type(model.marginals[1]).__name__, "GaussianDistribution")

    def test_the_copula_beats_the_independent_composite_on_dependent_data(self):
        data = _dependent_heterogeneous(1)
        cop = optimize(data, out=None)
        comp = optimize(data, structure="off", out=None)  # forces the historical independent composite
        enc_c = cop.dist_to_encoder().seq_encode(data)
        enc_i = comp.dist_to_encoder().seq_encode(data)
        ll_cop = float(np.sum(cop.seq_log_density(enc_c)))
        ll_comp = float(np.sum(comp.seq_log_density(enc_i)))
        self.assertGreater(ll_cop, ll_comp + 50.0)  # dependence is real and worth many nats

    def test_independent_continuous_records_stay_a_composite(self):
        # no dependence -> the copula's correlation params don't pay for themselves -> composite wins.
        rng = np.random.RandomState(2)
        x0 = spgamma.rvs(2.0, scale=2.0, size=1500, random_state=rng)
        x1 = norm.rvs(5.0, 2.0, size=1500, random_state=rng)
        data = [(float(a), float(b)) for a, b in zip(x0, x1)]
        model = optimize(data, out=None)
        self.assertNotIsInstance(model, CopulaDistribution)
        self.assertIsInstance(model, CompositeDistribution)

    def test_structure_off_restores_the_composite(self):
        model = optimize(_dependent_heterogeneous(3), structure="off", out=None)
        self.assertIsInstance(model, CompositeDistribution)

    def test_discrete_records_do_not_get_a_copula(self):
        # integer/categorical records are not continuous margins; the copula path must not fire (the
        # historical Bayesian-network-vs-composite behavior is unchanged for them).
        rng = np.random.RandomState(4)
        a = rng.randint(0, 5, size=800)
        b = (a + rng.randint(0, 2, size=800)) % 5  # dependent discrete
        data = [(int(x), int(y)) for x, y in zip(a, b)]
        model = optimize(data, out=None)
        self.assertNotIsInstance(model, CopulaDistribution)


if __name__ == "__main__":
    unittest.main()
