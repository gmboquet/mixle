"""Tests for ExponentialFamilyForm.fisher_information (WS-B1).

The Fisher information in natural coordinates of an exponential family is the covariance of the
sufficient statistic, ``I(eta) = Cov[T(x)] = grad^2 A(eta)`` -- the second-order companion to
``mean_parameters`` (``grad A = E[T]``). Validated by a closed-form 1-D case (Exponential:
``I = Var[x] = 1/lambda^2``) and by symmetry / positive-semidefiniteness / sample-consistency on a
2-D family (Gaussian, ``T = (x, x^2)``).
"""

import unittest

import numpy as np

from mixle.stats.compute.exp_family import to_exponential_family
from mixle.stats.univariate.continuous.exponential import ExponentialDistribution
from mixle.stats.univariate.continuous.gamma import GammaDistribution
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class ExponentialFamilyFisherTest(unittest.TestCase):
    def test_sample_budget_and_error_receipt(self):
        form = to_exponential_family(ExponentialDistribution(1.0))
        for invalid in (0, 1, -1):
            with self.subTest(n_samples=repr(invalid)):
                with self.assertRaises(ValueError):
                    form.fisher_information(n_samples=invalid)
        with self.assertRaises(TypeError):
            form.fisher_information(n_samples=2.5)

        estimate = form.estimate_fisher_information(n_samples=10, seed=4)
        self.assertEqual(estimate.value.shape, (1, 1))
        self.assertEqual(estimate.error_estimate.shape, (1, 1))
        self.assertTrue(np.all(np.isfinite(estimate.error_estimate)))

    def test_error_estimate_matches_the_explicit_outer_product_reduction(self):
        """The covariance and its per-entry standard error are reductions of the outer products
        T_i T_j. They are computed from two (dim, dim) moment matrices rather than by materializing
        the (n_samples, dim, dim) product tensor, which costs dim times the sample matrix itself.
        This pins the reduced form to the definition it replaces.
        """
        form = to_exponential_family(GaussianDistribution(1.0, 2.0))
        count = 5000
        estimate = form.estimate_fisher_information(n_samples=count, seed=11)

        samples = form.distribution.sampler(11).sample(count)
        stats = np.asarray(form.sufficient_statistics(samples), dtype=np.float64)
        centered = stats - stats.mean(axis=0)
        products = centered[:, :, None] * centered[:, None, :]
        np.testing.assert_allclose(estimate.value, products.sum(axis=0) / (count - 1), rtol=1e-10)
        np.testing.assert_allclose(estimate.error_estimate, products.std(axis=0, ddof=1) / np.sqrt(count), rtol=1e-8)

    def test_exponential_matches_closed_form(self):
        # ExponentialDistribution is parameterized by its mean beta, so Var[x] = beta^2.
        beta = 1.5
        form = to_exponential_family(ExponentialDistribution(beta))
        info = form.fisher_information(n_samples=200000, seed=0)
        self.assertEqual(info.shape, (1, 1))
        # I(eta) = Cov[T(x)] = Var[x] = beta^2 for the Exponential (T(x) = x).
        np.testing.assert_allclose(info[0, 0], beta**2, rtol=0.03)

    def test_gaussian_is_symmetric_psd_and_consistent(self):
        d = GaussianDistribution(1.0, 2.0)
        form = to_exponential_family(d)
        info = form.fisher_information(n_samples=200000, seed=0)
        self.assertEqual(info.shape, (2, 2))
        np.testing.assert_allclose(info, info.T, atol=1e-9)  # symmetric
        eigvals = np.linalg.eigvalsh(info)
        self.assertGreaterEqual(float(eigvals.min()), -1e-8)  # PSD

        # Consistent with an independent-sample covariance of the sufficient statistic.
        samples = d.sampler(123).sample(200000)
        t = np.asarray(form.sufficient_statistics(samples), dtype=np.float64)
        np.testing.assert_allclose(info, np.cov(t, rowvar=False), rtol=0.08, atol=0.05)

    def test_gamma_is_symmetric_psd(self):
        form = to_exponential_family(GammaDistribution(2.0, 1.3))
        info = form.fisher_information(n_samples=200000, seed=1)
        self.assertEqual(info.shape, (2, 2))
        np.testing.assert_allclose(info, info.T, atol=1e-9)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(info).min()), -1e-8)


if __name__ == "__main__":
    unittest.main()
