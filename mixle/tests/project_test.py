"""Closed-form variational projections: exact moment-matching, checked against analytic invariants.

These are closed-form operations, so the bar is machine precision (not stochastic tolerance): the
collapse must equal the analytic law-of-total-variance moments, reduction must preserve the mixture's
overall mean and covariance exactly, and reduce-to-one must equal collapse. One test also confirms the
closed-form collapse agrees with the existing SAMPLING projection (mixle.ops.project) to sampling error.
"""

import unittest

import numpy as np

from mixle.inference.project import collapse_mixture, fisher_merge, gaussian_kl, moment_project, reduce_mixture
from mixle.stats.latent.gaussian_mixture import GaussianMixtureDistribution
from mixle.stats.latent.mixture import MixtureDistribution
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


def _analytic_moments(w, mus, covs):
    """The reference overall mean and covariance of a Gaussian mixture (law of total variance)."""
    w = np.asarray(w, float)
    w = w / w.sum()
    mus, covs = np.asarray(mus, float), np.asarray(covs, float)
    mu = w @ mus
    d = mus - mu
    cov = np.einsum("k,kij->ij", w, covs) + np.einsum("k,ki,kj->ij", w, d, d)
    return mu, cov


class GaussianKLTest(unittest.TestCase):
    def test_self_kl_is_zero_and_positive_otherwise(self):
        p = MultivariateGaussianDistribution(np.array([1.0, -2.0]), np.array([[2.0, 0.3], [0.3, 1.0]]))
        q = MultivariateGaussianDistribution(np.array([0.0, 0.0]), np.eye(2))
        self.assertAlmostEqual(gaussian_kl(p, p), 0.0, places=10)
        self.assertGreater(gaussian_kl(p, q), 0.0)

    def test_matches_closed_form_univariate(self):
        # KL(N(m0,v0) || N(m1,v1)) = ln(√(v1/v0)) + (v0 + (m0-m1)²)/(2 v1) - 1/2
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(1.0, 4.0)
        want = np.log(np.sqrt(4.0 / 1.0)) + (1.0 + 1.0) / (2 * 4.0) - 0.5
        self.assertAlmostEqual(gaussian_kl(p, q), want, places=10)


class CollapseTest(unittest.TestCase):
    def _gmm(self):
        mus = np.array([[0.0, 0.0], [3.0, 1.0], [-2.0, 4.0]])
        covs = np.array([np.diag([1.0, 2.0]), [[2.0, 0.5], [0.5, 1.0]], np.diag([0.5, 0.5])])
        w = np.array([0.5, 0.3, 0.2])
        return GaussianMixtureDistribution(mus, covs, w), w, mus, covs

    def test_collapse_equals_law_of_total_variance(self):
        gmm, w, mus, covs = self._gmm()
        want_mu, want_cov = _analytic_moments(w, mus, covs)
        got = collapse_mixture(gmm)
        np.testing.assert_allclose(got.mu, want_mu, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got.covar, want_cov, rtol=0, atol=1e-12)

    def test_collapse_univariate_returns_gaussian(self):
        comps = [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 0.5)]
        mix = MixtureDistribution(comps, np.array([0.6, 0.4]))
        got = collapse_mixture(mix)
        self.assertIsInstance(got, GaussianDistribution)
        # mean = 0.6*-1 + 0.4*2 = 0.2 ; var = E[v] + Var[mean]
        self.assertAlmostEqual(got.mu, 0.2, places=12)
        want_var = 0.6 * 1.0 + 0.4 * 0.5 + 0.6 * (-1.2) ** 2 + 0.4 * (1.8) ** 2
        self.assertAlmostEqual(got.sigma2, want_var, places=12)

    def test_collapse_agrees_with_sampling_projection(self):
        # the closed-form collapse must equal mixle.ops.project (which SAMPLES) to sampling error
        from mixle.ops import project

        comps = [GaussianDistribution(-1.0, 1.0), GaussianDistribution(3.0, 2.0)]
        mix = MixtureDistribution(comps, np.array([0.7, 0.3]))
        exact = collapse_mixture(mix)
        sampled = project(mix, GaussianDistribution(0.0, 1.0).estimator(), n_samples=40000, seed=0)
        self.assertAlmostEqual(exact.mu, sampled.mu, delta=0.05)
        self.assertAlmostEqual(exact.sigma2, sampled.sigma2, delta=0.3)


class ReduceTest(unittest.TestCase):
    def _gmm(self, seed=0):
        rng = np.random.RandomState(seed)
        k, d = 6, 3
        mus = rng.randn(k, d) * 3
        covs = np.stack([np.diag(rng.uniform(0.5, 2.0, d)) for _ in range(k)])
        w = rng.dirichlet(np.ones(k))
        return GaussianMixtureDistribution(mus, covs, w), w, mus, covs

    def test_reduce_preserves_overall_mean_and_covariance(self):
        # every Runnalls merge is moment-preserving, so the reduced mixture keeps the exact global moments
        gmm, w, mus, covs = self._gmm()
        want_mu, want_cov = _analytic_moments(w, mus, covs)
        for m in (5, 3, 2, 1):
            red = reduce_mixture(gmm, m)
            self.assertLessEqual(red.num_components, m)
            got_mu, got_cov = _analytic_moments(red.w, red.mu, red.sig2)
            np.testing.assert_allclose(got_mu, want_mu, rtol=0, atol=1e-10)
            np.testing.assert_allclose(got_cov, want_cov, rtol=0, atol=1e-10)

    def test_reduce_to_one_equals_collapse(self):
        gmm, *_ = self._gmm(seed=3)
        one = reduce_mixture(gmm, 1)
        collapsed = collapse_mixture(gmm)
        self.assertEqual(one.num_components, 1)
        np.testing.assert_allclose(one.mu[0], collapsed.mu, rtol=0, atol=1e-10)
        np.testing.assert_allclose(one.sig2[0], collapsed.covar, rtol=0, atol=1e-10)

    def test_reduce_noop_when_already_small_enough(self):
        gmm, *_ = self._gmm()
        same = reduce_mixture(gmm, 10)  # target exceeds K -> unchanged
        self.assertEqual(same.num_components, gmm.num_components)

    def test_runnalls_merges_the_closest_pair(self):
        # two nearly-identical components + one far away; reducing 3->2 must merge the two close ones,
        # leaving the far component essentially untouched
        mus = np.array([[0.0, 0.0], [0.05, -0.03], [10.0, 10.0]])
        covs = np.array([np.eye(2), np.eye(2), np.eye(2)])
        w = np.array([0.4, 0.4, 0.2])
        gmm = GaussianMixtureDistribution(mus, covs, w)
        red = reduce_mixture(gmm, 2)
        self.assertEqual(red.num_components, 2)
        # the far component (mean ~[10,10], weight 0.2) survives with its weight and mean intact
        far = min(range(2), key=lambda k: abs(red.w[k] - 0.2))
        np.testing.assert_allclose(red.mu[far], [10.0, 10.0], atol=1e-9)
        self.assertAlmostEqual(red.w[far], 0.2, places=9)
        # the merged component carries the other 0.8 of the mass, centered near the two close means
        merged = 1 - far
        self.assertAlmostEqual(red.w[merged], 0.8, places=9)
        np.testing.assert_allclose(red.mu[merged], [0.025, -0.015], atol=1e-9)


class MomentProjectDispatchTest(unittest.TestCase):
    def test_gaussian_mixture_takes_the_exact_path(self):
        gmm = GaussianMixtureDistribution(np.array([[0.0], [2.0]]), np.array([[[1.0]], [[0.5]]]), np.array([0.5, 0.5]))
        got = moment_project(gmm)  # target=None -> exact collapse
        exact = collapse_mixture(gmm)
        np.testing.assert_allclose(got.mu, exact.mu, atol=1e-12)

    def test_non_gaussian_needs_a_target_and_delegates_to_sampling(self):
        from mixle.capability import CapabilityError

        gmm = MixtureDistribution(
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(2.0, 1.0)], np.array([0.5, 0.5])
        )
        # a Gaussian source WITH a target still works via the sampling path (exact=False)
        got = moment_project(gmm, GaussianDistribution(0.0, 1.0).estimator(), exact=False, n_samples=20000, seed=0)
        self.assertIsInstance(got, GaussianDistribution)
        # no target and not collapsible -> a clear capability error
        with self.assertRaises(CapabilityError):
            moment_project("not a distribution", None)


class FisherMergeTest(unittest.TestCase):
    def test_precision_weighted_mean_scalar(self):
        # merge two 1-D estimates with precisions 1 and 3 -> (1*0 + 3*4)/(1+3) = 3.0
        got = fisher_merge([np.array([0.0]), np.array([4.0])], [1.0, 3.0])
        self.assertAlmostEqual(got[0], 3.0, places=12)

    def test_equal_fisher_is_plain_average(self):
        got = fisher_merge([np.array([1.0, 2.0]), np.array([3.0, 6.0])], [2.5, 2.5])
        np.testing.assert_allclose(got, [2.0, 4.0], atol=1e-12)
        # and None (unit Fisher) is also the plain average
        np.testing.assert_allclose(fisher_merge([[1.0, 2.0], [3.0, 6.0]]), [2.0, 4.0], atol=1e-12)

    def test_diagonal_fisher_is_per_coordinate(self):
        # coordinate 0: precisions (1,1) -> mean; coordinate 1: precisions (9,1) -> pulled to first
        got = fisher_merge([np.array([0.0, 0.0]), np.array([2.0, 2.0])], [np.array([1.0, 9.0]), np.array([1.0, 1.0])])
        self.assertAlmostEqual(got[0], 1.0, places=12)
        self.assertAlmostEqual(got[1], (9 * 0 + 1 * 2) / 10.0, places=12)

    def test_matches_gaussian_product_of_experts_mean(self):
        # Fisher of a Gaussian mean is its precision 1/sigma^2; the Fisher-merge of two means must equal
        # the product-of-experts mean (the exact Bayesian combination of two Gaussian beliefs)
        from mixle.ops import product_of_experts
        from mixle.stats.univariate.continuous.gaussian import GaussianDistribution

        m0, v0, m1, v1 = -1.0, 2.0, 3.0, 0.5
        merged = fisher_merge([np.array([m0]), np.array([m1])], [1.0 / v0, 1.0 / v1])
        poe = product_of_experts([GaussianDistribution(m0, v0), GaussianDistribution(m1, v1)])
        self.assertAlmostEqual(merged[0], poe.mu, places=12)

    def test_full_fisher_matrix_solve(self):
        # two estimates with full-matrix Fishers -> (F0+F1)^-1 (F0 t0 + F1 t1)
        t0, t1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        f0 = np.array([[2.0, 0.0], [0.0, 1.0]])
        f1 = np.array([[1.0, 0.0], [0.0, 4.0]])
        want = np.linalg.solve(f0 + f1, f0 @ t0 + f1 @ t1)
        np.testing.assert_allclose(fisher_merge([t0, t1], [f0, f1]), want, atol=1e-10)

    def test_single_estimate_is_identity(self):
        np.testing.assert_allclose(fisher_merge([np.array([5.0, -1.0])], [np.array([3.0, 3.0])]), [5.0, -1.0])


if __name__ == "__main__":
    unittest.main()
