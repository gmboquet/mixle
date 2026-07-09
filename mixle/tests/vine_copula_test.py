"""Vine (C-vine) copula (mixle.stats.multivariate.vine_copula): high-dimensional dependence from a cascade of
bivariate pair copulas with per-edge families. Correctness is anchored by the exact equivalence 'a Gaussian
C-vine == the Gaussian copula'; the payoff test shows per-edge family selection beating any single family."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import norm, spearmanr

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.combinator.copula import CopulaDistribution
from mixle.stats.multivariate.clayton_copula import ClaytonCopulaDistribution
from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution
from mixle.stats.multivariate.vine_copula import (
    ClaytonPairCopula,
    CVineCopulaDistribution,
    GaussianPairCopula,
    IndependencePairCopula,
)


def _fit_single_core_ll(core, u):
    est = core.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(core.dist_to_encoder().seq_encode(u), np.ones(len(u)), None)
    fit = est.estimate(None, acc.value())
    return float(np.sum(fit.seq_log_density(fit.dist_to_encoder().seq_encode(u))))


class PairCopulaTest(unittest.TestCase):
    def test_h_and_h_inv_are_inverses(self):
        rng = np.random.RandomState(0)
        a, b = rng.uniform(0.01, 0.99, 2000), rng.uniform(0.01, 0.99, 2000)
        for pc in (GaussianPairCopula(0.6), GaussianPairCopula(-0.4), ClaytonPairCopula(2.0), IndependencePairCopula()):
            self.assertLess(float(np.max(np.abs(pc.h_inv(pc.h(a, b), b) - a))), 1e-8)


class CVineCopulaTest(unittest.TestCase):
    def test_gaussian_cvine_equals_the_gaussian_copula(self):
        # the anchor: a Gaussian C-vine and a Gaussian copula are the SAME model, different parameterization.
        R = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
        gc = GaussianCopulaDistribution(R)
        u = gc.sampler(0).sample(4000)
        vine = CVineCopulaDistribution(3, {}, candidates=("gaussian",)).estimator().estimate(None, (u, np.ones(len(u))))
        gc_fit = _fit_gaussian_copula(gc, u)
        test = gc.sampler(1).sample(400)
        ld_vine = vine.seq_log_density(test)
        ld_gc = gc_fit.seq_log_density(gc_fit.dist_to_encoder().seq_encode(test))
        np.testing.assert_allclose(ld_vine, ld_gc, atol=0.05)  # identical model -> matching densities

    def test_per_edge_family_selection_recovers_a_heterogeneous_structure(self):
        # true vine: one Clayton edge (lower-tail) + two Gaussian edges. Selection should recover the families.
        true = CVineCopulaDistribution(
            3,
            {(1, 1): ClaytonPairCopula(4.0), (1, 2): GaussianPairCopula(0.3), (2, 1): GaussianPairCopula(0.2)},
        )
        u = true.sampler(0).sample(3000)
        vine = CVineCopulaDistribution(3, {}).estimator().estimate(None, (u, np.ones(len(u))))
        self.assertEqual(vine.pairs[(1, 1)].family, "clayton")  # the strong lower-tail edge is picked as Clayton

    def test_vine_beats_any_single_family_on_heterogeneous_data(self):
        true = CVineCopulaDistribution(
            3,
            {(1, 1): ClaytonPairCopula(5.0), (1, 2): GaussianPairCopula(0.4), (2, 1): GaussianPairCopula(0.2)},
        )
        u = true.sampler(0).sample(3000)
        vine = CVineCopulaDistribution(3, {}).estimator().estimate(None, (u, np.ones(len(u))))
        ll_vine = float(np.sum(vine.seq_log_density(u)))
        ll_gauss = _fit_single_core_ll(GaussianCopulaDistribution(np.eye(3)), u)
        ll_clayton = _fit_single_core_ll(ClaytonCopulaDistribution(3, 0.5), u)
        self.assertGreater(ll_vine, ll_gauss)  # per-edge families beat one elliptical family
        self.assertGreater(ll_vine, ll_clayton)  # and one Archimedean family

    def test_sampling_reproduces_pairwise_dependence(self):
        R = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.5], [0.2, 0.5, 1.0]])
        u = GaussianCopulaDistribution(R).sampler(0).sample(4000)
        vine = CVineCopulaDistribution(3, {}, candidates=("gaussian",)).estimator().estimate(None, (u, np.ones(len(u))))
        s = vine.sampler(0).sample(4000)
        for i, j in ((0, 1), (1, 2), (0, 2)):
            self.assertAlmostEqual(spearmanr(s[:, i], s[:, j])[0], spearmanr(u[:, i], u[:, j])[0], delta=0.08)

    def test_empty_vine_is_the_independence_copula(self):
        v = CVineCopulaDistribution(4, {})
        np.testing.assert_allclose(v.seq_log_density(np.array([[0.2, 0.4, 0.6, 0.8]])), 0.0, atol=1e-12)

    def test_rejects_dim_below_two(self):
        with self.assertRaises(ValueError):
            CVineCopulaDistribution(1, {})

    def test_plugs_into_copula_distribution_with_heterogeneous_marginals(self):
        true = CVineCopulaDistribution(
            3, {(1, 1): ClaytonPairCopula(4.0), (1, 2): GaussianPairCopula(0.3), (2, 1): GaussianPairCopula(0.2)}
        )
        u = true.sampler(0).sample(1500)
        x0 = spgamma.ppf(np.clip(u[:, 0], 1e-9, 1 - 1e-9), a=2.0, scale=2.0)
        x1 = norm.ppf(np.clip(u[:, 1], 1e-9, 1 - 1e-9), loc=5.0, scale=2.0)
        x2 = spgamma.ppf(np.clip(u[:, 2], 1e-9, 1 - 1e-9), a=1.5, scale=3.0)
        data = list(zip(x0.tolist(), x1.tolist(), x2.tolist()))
        proto = CopulaDistribution(
            [st.GammaDistribution(1.0, 1.0), st.GaussianDistribution(0.0, 1.0), st.GammaDistribution(1.0, 1.0)],
            CVineCopulaDistribution(3, {}),
        )
        fit = optimize(data, proto.estimator(), prev_estimate=proto, max_its=3, out=None)
        self.assertAlmostEqual(float(fit.marginals[1].mu), 5.0, delta=0.4)  # marginals recovered
        self.assertEqual(fit.copula.pairs[(1, 1)].family, "clayton")  # vine structure recovered through the combinator


def _fit_gaussian_copula(gc, u):
    est = gc.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(gc.dist_to_encoder().seq_encode(u), np.ones(len(u)), None)
    return est.estimate(None, acc.value())


if __name__ == "__main__":
    unittest.main()
