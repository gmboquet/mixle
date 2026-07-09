"""D-vine copula + the expanded pair-copula families (Frank, Gumbel, Student-t) in mixle.stats.multivariate.
vine_copula. The D-vine is the second canonical vine (a path of pair copulas); anchored by 'a Gaussian D-vine
== the Gaussian copula', with correct sampling and per-edge family selection across all six families."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm, spearmanr

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.combinator.copula import CopulaDistribution
from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution
from mixle.stats.multivariate.vine_copula import (
    ClaytonPairCopula,
    CVineCopulaDistribution,
    DVineCopulaDistribution,
    FrankPairCopula,
    GaussianPairCopula,
    GumbelPairCopula,
    StudentTPairCopula,
)


def _fit_gaussian_copula(gc, u):
    est = gc.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(gc.dist_to_encoder().seq_encode(u), np.ones(len(u)), None)
    return est.estimate(None, acc.value())


class NewPairCopulaTest(unittest.TestCase):
    def test_h_is_the_conditional_cdf_and_h_inv_inverts_it(self):
        # d/da h(a|b) must equal the copula density c(a,b), i.e. h is exactly F(a|b); and h_inv undoes h.
        rng = np.random.RandomState(0)
        a, b = rng.uniform(0.05, 0.95, 500), rng.uniform(0.05, 0.95, 500)
        aa, bb = np.array([0.3, 0.5, 0.7]), np.array([0.6, 0.4, 0.2])
        eps = 1e-6
        for pc in (FrankPairCopula(5.0), FrankPairCopula(-4.0), GumbelPairCopula(2.5), StudentTPairCopula(0.5, 5.0)):
            dh = (pc.h(aa + eps, bb) - pc.h(aa - eps, bb)) / (2 * eps)
            self.assertLess(float(np.max(np.abs(dh - np.exp(pc.logpdf(aa, bb))))), 1e-4)
            self.assertLess(float(np.max(np.abs(pc.h_inv(pc.h(a, b), b) - a))), 1e-6)

    def test_densities_integrate_to_one(self):
        g = np.linspace(1e-3, 1 - 1e-3, 250)
        u, v = np.meshgrid(g, g)
        for pc in (FrankPairCopula(6.0), GumbelPairCopula(2.0), StudentTPairCopula(0.5, 4.0)):
            dens = np.exp(pc.logpdf(u.ravel(), v.ravel())).reshape(250, 250)
            self.assertAlmostEqual(float(np.trapezoid(np.trapezoid(dens, g, axis=1), g)), 1.0, delta=0.03)


class DVineCopulaTest(unittest.TestCase):
    def test_gaussian_dvine_equals_the_gaussian_copula(self):
        # the anchor: a Gaussian D-vine and a Gaussian copula are the SAME model (d=4).
        R = np.array([[1, 0.6, 0.3, 0.2], [0.6, 1, 0.5, 0.3], [0.3, 0.5, 1, 0.4], [0.2, 0.3, 0.4, 1.0]])
        gc = GaussianCopulaDistribution(R)
        u = gc.sampler(0).sample(5000)
        dvine = (
            DVineCopulaDistribution(4, {}, candidates=("gaussian",)).estimator().estimate(None, (u, np.ones(len(u))))
        )
        gc_fit = _fit_gaussian_copula(gc, u)
        test = gc.sampler(1).sample(500)
        np.testing.assert_allclose(
            dvine.seq_log_density(test),
            gc_fit.seq_log_density(gc_fit.dist_to_encoder().seq_encode(test)),
            atol=0.05,
        )

    def test_sampling_reproduces_pairwise_dependence(self):
        R = np.array([[1, 0.6, 0.3, 0.2], [0.6, 1, 0.5, 0.3], [0.3, 0.5, 1, 0.4], [0.2, 0.3, 0.4, 1.0]])
        u = GaussianCopulaDistribution(R).sampler(0).sample(5000)
        dvine = (
            DVineCopulaDistribution(4, {}, candidates=("gaussian",)).estimator().estimate(None, (u, np.ones(len(u))))
        )
        s = dvine.sampler(0).sample(5000)
        for i, j in ((0, 1), (1, 2), (2, 3), (0, 3)):
            self.assertAlmostEqual(spearmanr(s[:, i], s[:, j])[0], spearmanr(u[:, i], u[:, j])[0], delta=0.06)

    def test_per_edge_family_selection_recovers_heterogeneous_structure(self):
        true = DVineCopulaDistribution(
            3, {(1, 1): GumbelPairCopula(3.0), (1, 2): ClaytonPairCopula(3.0), (2, 1): FrankPairCopula(4.0)}
        )
        u = true.sampler(0).sample(4000)
        fit = DVineCopulaDistribution(3, {}).estimator().estimate(None, (u, np.ones(len(u))))
        self.assertEqual(fit.pairs[(1, 1)].family, "gumbel")  # upper-tail edge recovered
        self.assertEqual(fit.pairs[(1, 2)].family, "clayton")  # lower-tail edge recovered

    def test_empty_dvine_is_independence(self):
        v = DVineCopulaDistribution(4, {})
        np.testing.assert_allclose(v.seq_log_density(np.array([[0.2, 0.4, 0.6, 0.8]])), 0.0, atol=1e-12)

    def test_rejects_dim_below_two(self):
        with self.assertRaises(ValueError):
            DVineCopulaDistribution(1, {})

    def test_plugs_into_copula_distribution(self):
        true = DVineCopulaDistribution(
            3, {(1, 1): GumbelPairCopula(3.0), (1, 2): ClaytonPairCopula(3.0), (2, 1): FrankPairCopula(3.0)}
        )
        u = true.sampler(0).sample(1500)
        x0 = spgamma.ppf(np.clip(u[:, 0], 1e-9, 1 - 1e-9), a=2.0, scale=2.0)
        x1 = norm.ppf(np.clip(u[:, 1], 1e-9, 1 - 1e-9), loc=5.0, scale=2.0)
        x2 = spgamma.ppf(np.clip(u[:, 2], 1e-9, 1 - 1e-9), a=1.5, scale=3.0)
        data = list(zip(x0.tolist(), x1.tolist(), x2.tolist()))
        proto = CopulaDistribution(
            [st.GammaDistribution(1.0, 1.0), st.GaussianDistribution(0.0, 1.0), st.GammaDistribution(1.0, 1.0)],
            DVineCopulaDistribution(3, {}),
        )
        fit = optimize(data, proto.estimator(), prev_estimate=proto, max_its=3, out=None)
        self.assertAlmostEqual(float(fit.marginals[1].mu), 5.0, delta=0.4)
        self.assertIsInstance(fit.copula, DVineCopulaDistribution)


class ExpandedCVineTest(unittest.TestCase):
    def test_cvine_selects_from_all_six_families(self):
        true = CVineCopulaDistribution(
            3, {(1, 1): GumbelPairCopula(3.0), (1, 2): GaussianPairCopula(0.5), (2, 1): FrankPairCopula(3.0)}
        )
        u = true.sampler(0).sample(4000)
        fit = CVineCopulaDistribution(3, {}).estimator().estimate(None, (u, np.ones(len(u))))
        self.assertEqual(fit.pairs[(1, 1)].family, "gumbel")  # C-vine now reaches the Archimedean upper-tail family


if __name__ == "__main__":
    unittest.main()
