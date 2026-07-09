"""Regular vine (R-vine) copula (mixle.stats.multivariate.rvine_copula): the general vine with automatic
Dissmann structure selection. Anchored by 'a Gaussian R-vine == the Gaussian copula' (at d=5, whatever
structure is selected); the payoff test shows auto-selection beating a fixed-order C-vine on chain data;
sampling is validated by recovering the full joint (correlation / Kendall tau) for Gaussian AND Clayton vines."""

import unittest

import numpy as np
from scipy.stats import gamma as spgamma
from scipy.stats import kendalltau, norm

import mixle.stats as st
from mixle.inference import optimize
from mixle.stats.combinator.copula import CopulaDistribution
from mixle.stats.multivariate.clayton_copula import ClaytonCopulaDistribution
from mixle.stats.multivariate.gaussian_copula import GaussianCopulaDistribution
from mixle.stats.multivariate.rvine_copula import RVineCopulaDistribution
from mixle.stats.multivariate.vine_copula import CVineCopulaDistribution


def _fit_gaussian_copula(gc, u):
    est = gc.estimator()
    acc = est.accumulator_factory().make()
    acc.seq_update(gc.dist_to_encoder().seq_encode(u), np.ones(len(u)), None)
    return est.estimate(None, acc.value())


def _chain_corr(d, rho=0.8):
    r = np.eye(d)
    for i in range(d - 1):
        r[i, i + 1] = r[i + 1, i] = rho
    r = r @ r.T
    s = np.sqrt(np.diag(r))
    return r / np.outer(s, s)


class RVineCopulaTest(unittest.TestCase):
    def test_gaussian_rvine_equals_the_gaussian_copula_at_d5(self):
        # anchor: any correct vine of Gaussians fit to Gaussian-copula data reproduces the Gaussian copula.
        R = np.array(
            [
                [1, 0.7, 0.4, 0.3, 0.2],
                [0.7, 1, 0.5, 0.4, 0.3],
                [0.4, 0.5, 1, 0.6, 0.4],
                [0.3, 0.4, 0.6, 1, 0.5],
                [0.2, 0.3, 0.4, 0.5, 1.0],
            ]
        )
        gc = GaussianCopulaDistribution(R)
        u = gc.sampler(0).sample(6000)
        rvine = (
            RVineCopulaDistribution(5, [], candidates=("gaussian",)).estimator().estimate(None, (u, np.ones(len(u))))
        )
        gc_fit = _fit_gaussian_copula(gc, u)
        test = gc.sampler(1).sample(500)
        np.testing.assert_allclose(
            rvine.seq_log_density(test),
            gc_fit.seq_log_density(gc_fit.dist_to_encoder().seq_encode(test)),
            atol=0.05,
        )

    def test_auto_structure_beats_a_fixed_order_cvine_on_chain_data(self):
        # chain dependence 0-1-2-...-5: the best vine is a path; a fixed-root C-vine is a poor fit.
        u = GaussianCopulaDistribution(_chain_corr(6)).sampler(0).sample(4000)
        cand = ("gaussian", "independence")
        rvine = RVineCopulaDistribution(6, [], candidates=cand).estimator().estimate(None, (u, np.ones(len(u))))
        cvine = CVineCopulaDistribution(6, {}, candidates=cand).estimator().estimate(None, (u, np.ones(len(u))))
        ll_r = float(np.sum(rvine.seq_log_density(u)))
        ll_c = float(np.sum(cvine.seq_log_density(u)))
        self.assertGreater(ll_r, ll_c)  # Dissmann selection finds the chain the fixed C-vine root misses

    def test_sampling_recovers_the_full_gaussian_joint(self):
        R = np.array([[1, 0.7, 0.4, 0.3], [0.7, 1, 0.5, 0.4], [0.4, 0.5, 1, 0.6], [0.3, 0.4, 0.6, 1.0]])
        gc = GaussianCopulaDistribution(R)
        u = gc.sampler(0).sample(6000)
        rvine = (
            RVineCopulaDistribution(4, [], candidates=("gaussian",)).estimator().estimate(None, (u, np.ones(len(u))))
        )
        s = rvine.sampler(1).sample(8000)
        Rhat = _fit_gaussian_copula(gc, s).corr
        self.assertLess(float(np.max(np.abs(Rhat - R))), 0.06)  # full correlation matrix reproduced

    def test_sampling_recovers_a_nongaussian_joint_with_tail_dependence(self):
        u = ClaytonCopulaDistribution(4, theta=2.5).sampler(0).sample(5000)
        cand = ("clayton", "gaussian", "frank", "independence")
        rvine = RVineCopulaDistribution(4, [], candidates=cand).estimator().estimate(None, (u, np.ones(len(u))))
        s = rvine.sampler(1).sample(6000)
        worst = max(
            abs(kendalltau(u[:, i], u[:, j])[0] - kendalltau(s[:, i], s[:, j])[0])
            for i in range(4)
            for j in range(i + 1, 4)
        )
        self.assertLess(worst, 0.05)  # every pairwise Kendall tau reproduced
        self.assertGreater(np.mean((s < 0.1).all(axis=1)), 0.03)  # Clayton lower-tail dependence preserved

    def test_selects_per_edge_families(self):
        u = ClaytonCopulaDistribution(4, theta=3.0).sampler(0).sample(4000)
        rvine = RVineCopulaDistribution(4, []).estimator().estimate(None, (u, np.ones(len(u))))
        fams = [e.copula.family for tree in rvine.trees for e in tree]
        self.assertIn("clayton", fams)  # the lower-tail structure is recovered as Clayton edges

    def test_empty_rvine_is_independence(self):
        rvine = RVineCopulaDistribution(4, [])
        np.testing.assert_allclose(rvine.seq_log_density(np.array([[0.2, 0.4, 0.6, 0.8]])), 0.0, atol=1e-12)

    def test_rejects_dim_below_two(self):
        with self.assertRaises(ValueError):
            RVineCopulaDistribution(1, [])

    def test_plugs_into_copula_distribution(self):
        u = ClaytonCopulaDistribution(3, theta=2.5).sampler(0).sample(1500)
        x0 = spgamma.ppf(np.clip(u[:, 0], 1e-9, 1 - 1e-9), a=2.0, scale=2.0)
        x1 = norm.ppf(np.clip(u[:, 1], 1e-9, 1 - 1e-9), loc=5.0, scale=2.0)
        x2 = spgamma.ppf(np.clip(u[:, 2], 1e-9, 1 - 1e-9), a=1.5, scale=3.0)
        data = list(zip(x0.tolist(), x1.tolist(), x2.tolist()))
        proto = CopulaDistribution(
            [st.GammaDistribution(1.0, 1.0), st.GaussianDistribution(0.0, 1.0), st.GammaDistribution(1.0, 1.0)],
            RVineCopulaDistribution(3, []),
        )
        fit = optimize(data, proto.estimator(), prev_estimate=proto, max_its=3, out=None)
        self.assertAlmostEqual(float(fit.marginals[1].mu), 5.0, delta=0.4)
        self.assertIsInstance(fit.copula, RVineCopulaDistribution)


if __name__ == "__main__":
    unittest.main()
