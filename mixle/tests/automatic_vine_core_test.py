"""Automatic inference picks the right copula CORE by BIC (mixle.inference.estimation): a regular vine when
the joint has tail dependence (a Clayton-style joint-crash coupling), the elliptical Gaussian copula when it
doesn't, and neither when the columns are independent -- all decided by BIC, never forced."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm

from mixle.inference import optimize
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.combinator.copula import CopulaDistribution
from mixle.stats.multivariate.clayton_copula import ClaytonCopulaDistribution
from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution
from mixle.stats.multivariate.rvine_copula import RVineCopulaDistribution


def _heterogeneous(u):
    """Push copula scores u through a Gamma marginal and a Gaussian marginal."""
    x0 = spgamma.ppf(np.clip(u[:, 0], 1e-9, 1 - 1e-9), a=2.0, scale=2.0)
    x1 = norm.ppf(np.clip(u[:, 1], 1e-9, 1 - 1e-9), loc=5.0, scale=2.0)
    return [(float(a), float(b)) for a, b in zip(x0, x1)]


class AutomaticVineCoreTest(unittest.TestCase):
    def test_tail_dependent_data_gets_a_vine(self):
        # Clayton coupling has strong LOWER-tail dependence the elliptical Gaussian copula cannot represent;
        # the vine's per-edge family selection should recover it and win on BIC.
        u = ClaytonCopulaDistribution(2, theta=3.0).sampler(0).sample(2000)
        model = optimize(_heterogeneous(u), out=None)
        self.assertIsInstance(model, CopulaDistribution)
        self.assertIsInstance(model.copula, RVineCopulaDistribution)
        fams = [e.copula.family for tree in model.copula.trees for e in tree]
        self.assertIn("clayton", fams)  # the tail-dependence family was selected, not forced Gaussian

    def test_elliptical_data_keeps_the_gaussian_copula(self):
        # Gaussian coupling is tail-independent: the vine's extra per-edge parameters don't pay for themselves,
        # so BIC keeps the simpler Gaussian copula core.
        rng = np.random.RandomState(0)
        z = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.7], [0.7, 1.0]], size=2000)
        model = optimize(_heterogeneous(norm.cdf(z)), out=None)
        self.assertIsInstance(model, CopulaDistribution)
        self.assertIsInstance(model.copula, GaussianCopulaDistribution)

    def test_independent_data_gets_no_copula_and_no_vine(self):
        rng = np.random.RandomState(1)
        x0 = spgamma.rvs(2.0, scale=2.0, size=2000, random_state=rng)
        x1 = norm.rvs(5.0, 2.0, size=2000, random_state=rng)
        model = optimize([(float(a), float(b)) for a, b in zip(x0, x1)], out=None)
        self.assertIsInstance(model, CompositeDistribution)  # no dependence -> no vine even attempted

    def test_vine_captures_tail_dependence_a_gaussian_copula_misses(self):
        # sanity that the vine is genuinely better here: on Clayton data it out-scores the Gaussian copula.
        u = ClaytonCopulaDistribution(2, theta=4.0).sampler(0).sample(2000)
        data = _heterogeneous(u)
        vine = optimize(data, out=None)
        # refit a Gaussian-copula-cored model on the same marginals for comparison
        marg = list(vine.marginals)
        gc = CopulaDistribution(marg, GaussianCopulaDistribution(np.eye(2)))
        gc = optimize(data, gc.estimator(), prev_estimate=gc, out=None)
        ll_vine = float(np.sum(vine.seq_log_density(vine.dist_to_encoder().seq_encode(data))))
        ll_gc = float(np.sum(gc.seq_log_density(gc.dist_to_encoder().seq_encode(data))))
        self.assertGreater(ll_vine, ll_gc)  # the tail-dependence core fits the joint-crash coupling better


if __name__ == "__main__":
    unittest.main()
