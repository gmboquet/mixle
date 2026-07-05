"""Regression: MVN covariance Cholesky self-heals a non-PD matrix (float32/GPU precision loss)."""

import unittest

import numpy as np

import mixle.stats as st
from mixle.stats.multivariate.multivariate_gaussian import _robust_cho_factor


class RobustCholeskyTest(unittest.TestCase):
    def test_pd_covariance_is_untouched(self):
        # a genuinely PD covariance takes the fast path -> identical to plain cho_factor
        import scipy.linalg

        cov = np.array([[2.0, 0.3], [0.3, 1.5]])
        got = _robust_cho_factor(cov)[0]
        ref = scipy.linalg.cho_factor(cov)[0]
        self.assertTrue(np.array_equal(got, ref))

    def test_non_pd_covariance_self_heals(self):
        # a matrix that is symmetric but has a tiny negative eigenvalue (float precision artifact)
        bad = np.array([[1.0, 1.0], [1.0, 1.0 - 1e-9]])  # nearly singular / non-PD
        factor = _robust_cho_factor(bad)  # must not raise
        self.assertIsNotNone(factor)

    def test_construction_survives_non_pd_input(self):
        d = st.MultivariateGaussianDistribution(np.zeros(2), np.array([[1.0, 1.0], [1.0, 1.0]]))
        self.assertIsNotNone(d.chol)  # a rank-deficient covariance no longer crashes construction

    def test_mvn_mixture_fit_survives_float32(self):
        # the reported crash: MPS/float32 MVN mixture at higher dim -> non-PD covariance
        torch = __import__("importlib").import_module("torch") if _has("torch") else None
        if torch is None or not torch.backends.mps.is_available():
            self.skipTest("needs torch+mps for the float32 path")
        from mixle.engines import TorchEngine
        from mixle.inference import optimize

        rng = np.random.RandomState(1)
        dim, k, n = 64, 8, 20000
        comps = [st.MultivariateGaussianDistribution(rng.randn(dim) * 4, np.eye(dim)) for _ in range(k)]
        data = st.MixtureDistribution(comps, list(rng.dirichlet(np.ones(k)))).sampler(1).sample(n)
        est = st.MixtureEstimator([st.MultivariateGaussianEstimator(dim=dim) for _ in range(k)])
        init = st.MixtureDistribution(
            [st.MultivariateGaussianDistribution(rng.randn(dim), np.eye(dim)) for _ in range(k)], [1.0 / k] * k
        )
        m = optimize(data, est, prev_estimate=init, max_its=5, out=None, engine=TorchEngine(device="mps"))
        self.assertIsNotNone(m)


def _has(mod):
    import importlib.util

    return importlib.util.find_spec(mod) is not None


if __name__ == "__main__":
    unittest.main()
