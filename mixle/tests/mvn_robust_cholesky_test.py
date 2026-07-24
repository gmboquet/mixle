"""Regression: MVN covariance Cholesky self-heals a non-PD matrix (float32/GPU precision loss).

Also covers the follow-up bug where scoring and sampling silently disagreed about the effective
covariance after an invalid (asymmetric, or only-PD-after-healing) covariance was accepted: scoring
(log_density/seq_log_density) always derived its covariance from ``chol``/``inv_covar``, but
``self.covar`` -- what the sampler and everything else reads -- kept the raw, un-healed input.
``_robust_cho_factor`` now returns (and construction stores) the exact matrix it factorized, so
every consumer agrees on one effective covariance.
"""

import unittest

import numpy as np

import mixle.stats as st
from mixle.stats.multivariate.multivariate_gaussian import _robust_cho_factor


def _covar_from_cho_factor(chol: tuple[np.ndarray, bool]) -> np.ndarray:
    """Reconstruct the covariance a ``scipy.linalg.cho_factor`` result actually corresponds to."""
    c, lower = chol
    if lower:
        tri = np.tril(c)
        return tri @ tri.T
    tri = np.triu(c)
    return tri.T @ tri


class RobustCholeskyTest(unittest.TestCase):
    def test_pd_covariance_is_untouched(self):
        # a genuinely PD covariance takes the fast path -> identical to plain cho_factor
        import scipy.linalg

        cov = np.array([[2.0, 0.3], [0.3, 1.5]])
        chol, effective_covar = _robust_cho_factor(cov)
        ref = scipy.linalg.cho_factor(cov)[0]
        self.assertTrue(np.array_equal(chol[0], ref))
        # the fast path returns the covariance itself (bit-exact, already symmetric)
        self.assertTrue(np.array_equal(effective_covar, cov))

    def test_non_pd_covariance_self_heals(self):
        # a matrix that is symmetric but has a tiny negative eigenvalue (float precision artifact)
        bad = np.array([[1.0, 1.0], [1.0, 1.0 - 1e-9]])  # nearly singular / non-PD
        chol, effective_covar = _robust_cho_factor(bad)  # must not raise
        self.assertIsNotNone(chol)
        # the healed covariance must be the one the returned factor actually corresponds to
        np.testing.assert_allclose(effective_covar, _covar_from_cho_factor(chol), atol=1e-12)

    def test_asymmetric_covariance_is_symmetrized_before_factoring(self):
        # cho_factor only reads one triangle of its input, so an asymmetric-but-implied-PD covariance
        # "succeeds" on the fast path against a matrix that differs from the raw input entirely (the
        # other triangle is silently ignored). The returned effective covariance must be the symmetric
        # matrix actually factorized, not the raw asymmetric input.
        asym = np.array([[1.0, 0.5], [0.6, 1.0]])
        chol, effective_covar = _robust_cho_factor(asym)
        np.testing.assert_allclose(effective_covar, np.array([[1.0, 0.55], [0.55, 1.0]]), atol=1e-12)
        np.testing.assert_allclose(effective_covar, _covar_from_cho_factor(chol), atol=1e-12)

    def test_construction_survives_non_pd_input(self):
        d = st.MultivariateGaussianDistribution(np.zeros(2), np.array([[1.0, 1.0], [1.0, 1.0]]))
        self.assertIsNotNone(d.chol)  # a rank-deficient covariance no longer crashes construction

    def test_scoring_and_sampling_agree_on_effective_covariance(self):
        # The bug this guards against: log_density (scoring) and sampler.sample (sampling) must
        # evaluate/draw from the IDENTICAL effective covariance, even when construction is handed an
        # accepted-but-invalid covariance. The textbook Gaussian log-density is recomputed directly
        # from d.covar -- what sampler.sample() draws from -- via independent numpy primitives
        # (slogdet/solve, not mixle's own chol/inv_covar), so a scoring/sampling desync would be
        # caught rather than masked by reusing the same internals on both sides. (scipy's own
        # multivariate_normal.logpdf is deliberately NOT used here: for the near-singular case below
        # it applies its own eigenvalue-cutoff regularization, which differs from mixle's jitter
        # healing for reasons unrelated to this bug and would make the two incomparable.)
        for bad_covar in (
            np.array([[1.0, 0.5], [0.6, 1.0]]),  # asymmetric, PD via cho_factor's fast path
            np.array([[1.0, 1.0], [1.0, 1.0 - 1e-9]]),  # symmetric, tiny negative eigenvalue
        ):
            mu = np.zeros(2)
            d = st.MultivariateGaussianDistribution(mu, bad_covar)
            x = np.array([0.3, -0.2])
            diff = x - mu

            sign, logdet = np.linalg.slogdet(d.covar)
            self.assertGreater(sign, 0.0)
            sampling_ll = -0.5 * (2.0 * np.log(2.0 * np.pi) + logdet + diff @ np.linalg.solve(d.covar, diff))

            scoring_ll = d.log_density(x)
            seq_ll = d.seq_log_density(x[None, :])[0]

            np.testing.assert_allclose(scoring_ll, sampling_ll, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(seq_ll, sampling_ll, rtol=1e-6, atol=1e-6)

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
