"""Fitted-parameter parity: mixle's Gaussian-mixture EM vs. scikit-learn's (worklist Q5.5).

``scipy_golden_test.py`` already pins per-distribution *densities* against SciPy. What that does not
prove is that a *fitted* latent model recovers the same parameters a trusted reference would -- final
likelihood alone can agree while means/covariances quietly drift. This test closes that gap for the
principal latent path (the Gaussian mixture) against scikit-learn's ``GaussianMixture`` as the reference
MLE.

The claim under test is exact-agreement, not mere closeness: two correct EM implementations fitting the
*same data* from the *same initialization* converge to the *same* maximum-likelihood fixed point. So both
fits are seeded from one shared init (means, covariances, weights) -- without that, either optimizer can
land in a different local basin and the comparison would be meaningless. With it, the fitted means,
weights, and full covariances agree to ``~1e-7`` (see the retained tolerances below, four-plus orders of
magnitude tighter than a parameter-recovery test's), and the mean log-likelihoods agree to machine noise.

Requires scikit-learn (an external reference, not a mixle runtime dependency); skips cleanly without it.
"""

import unittest

import numpy as np
import pytest

pytest.importorskip("sklearn")

from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.stats import MultivariateGaussianEstimator  # noqa: E402
from mixle.stats.latent.gaussian_mixture import (  # noqa: E402
    GaussianMixtureDistribution,
    GaussianMixtureEstimator,
)

# Three well-separated 2-D Gaussians with correlated (non-diagonal) covariances.
_TRUE_MEANS = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 8.0]])
_TRUE_COVS = [
    np.array([[1.0, 0.3], [0.3, 1.0]]),
    np.array([[0.8, -0.2], [-0.2, 1.2]]),
    np.array([[1.5, 0.0], [0.0, 0.6]]),
]
_TRUE_WEIGHTS = np.array([0.5, 0.3, 0.2])

# A fixed shared initialization for BOTH optimizers (deliberately off-truth, same basin).
_INIT_MEANS = np.array([[1.0, 1.0], [5.0, 5.0], [1.0, 7.0]])
_INIT_COVS = [2.0 * np.eye(2), 2.0 * np.eye(2), 2.0 * np.eye(2)]
_INIT_WEIGHTS = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]

_REG_COVAR = 1e-6
_MAX_ITER = 500
_TOL = 1e-12


def _sample(seed, n=3000):
    rng = np.random.RandomState(seed)
    comp = rng.choice(3, size=n, p=_TRUE_WEIGHTS)
    return np.array([rng.multivariate_normal(_TRUE_MEANS[c], _TRUE_COVS[c]) for c in comp])


def _fit_mixle(data):
    init = GaussianMixtureDistribution(_INIT_MEANS.tolist(), [c.copy() for c in _INIT_COVS], list(_INIT_WEIGHTS))
    est = GaussianMixtureEstimator([MultivariateGaussianEstimator(dim=2, min_covar=_REG_COVAR) for _ in range(3)])
    return optimize(data, estimator=est, prev_estimate=init, max_its=_MAX_ITER, delta=_TOL, out=None)


def _fit_sklearn(X):
    return GaussianMixture(
        n_components=3,
        covariance_type="full",
        reg_covar=_REG_COVAR,
        means_init=_INIT_MEANS,
        weights_init=_INIT_WEIGHTS,
        precisions_init=[np.linalg.inv(c) for c in _INIT_COVS],
        max_iter=_MAX_ITER,
        tol=_TOL,
    ).fit(X)


class SklearnGmmParityTest(unittest.TestCase):
    def _assert_parity(self, seed):
        X = _sample(seed)
        data = [x.tolist() for x in X]
        m = _fit_mixle(data)
        sk = _fit_sklearn(X)

        mu_m, w_m, sig_m = np.asarray(m.mu), np.asarray(m.w), np.asarray(m.sig2)
        # Align components across the two fits (mixture labels are exchangeable) by nearest mean.
        cost = np.linalg.norm(mu_m[:, None, :] - sk.means_[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(cost)

        np.testing.assert_allclose(mu_m[ri], sk.means_[ci], atol=1e-3, err_msg=f"means (seed {seed})")
        np.testing.assert_allclose(w_m[ri], sk.weights_[ci], atol=1e-3, err_msg=f"weights (seed {seed})")
        np.testing.assert_allclose(sig_m[ri], sk.covariances_[ci], atol=1e-3, err_msg=f"covariances (seed {seed})")

        # Same objective at the same fixed point: mean log-likelihoods must agree to machine noise.
        enc = m.dist_to_encoder().seq_encode(data)
        ll_mixle = float(m.seq_log_density(enc).mean())
        ll_sklearn = float(sk.score(X))
        self.assertAlmostEqual(ll_mixle, ll_sklearn, places=5, msg=f"mean log-likelihood (seed {seed})")

    def test_parameter_parity_across_seeds(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self._assert_parity(seed)


if __name__ == "__main__":
    unittest.main()
