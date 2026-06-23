"""Tests for ExponentialFamilyForm.fisher_information (WS-B1).

The Fisher information in natural coordinates of an exponential family is the covariance of the
sufficient statistic, ``I(eta) = Cov[T(x)] = grad^2 A(eta)`` -- the second-order companion to
``mean_parameters`` (``grad A = E[T]``). Validated by a closed-form 1-D case (Exponential:
``I = Var[x] = 1/lambda^2``) and by symmetry / positive-semidefiniteness / sample-consistency on a
2-D family (Gaussian, ``T = (x, x^2)``).
"""

import unittest

import numpy as np

from pysp.stats.exp_family import to_exponential_family
from pysp.stats.univariate.continuous.exponential import ExponentialDistribution
from pysp.stats.univariate.continuous.gamma import GammaDistribution
from pysp.stats.univariate.continuous.gaussian import GaussianDistribution


class ExponentialFamilyFisherTest(unittest.TestCase):
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
