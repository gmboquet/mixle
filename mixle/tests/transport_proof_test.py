"""CARD TRANSPORT-a -- the F0 transport-proof gate: can mixle learn a calibrated conditional
transport p(x | y) at all? The cheap, up-front go/no-go for the entire cross-modal (workstream F)
plan, checked on two checkable inverses BEFORE any belief graph (F1-F7) is built:

  1. A linear-Gaussian inverse y = A x + noise, where the true posterior p(x | y) is closed-form
     (a standard Bayesian linear-Gaussian / Kalman update).
  2. A small nonlinear inverse y = x^2 + noise (a classic sign-ambiguous, BIMODAL posterior), with a
     dense-grid reference posterior (exact up to grid resolution, since x is 1-D).

Two metrics only, per the card: (a) posterior fidelity -- the learned transport's sampled posterior
mean/std against the true/reference posterior; (b) calibration -- do the transport's own credible
intervals cover the truth at their nominal rate on held-out (x, y), checked with a binomial test
(mirroring mixle.task.solve.Solution.health()'s use of the same test for calibration drift).

TransportProofGoNoGoTest.test_go_no_go_report prints the card's required explicit go/no-go verdict,
computed from the same module-level fits the other two test classes verify.
"""

import unittest

import numpy as np
from scipy.stats import binomtest

from mixle.inference import optimize

try:
    import torch  # noqa: F401

    from mixle.models.mixture_density import NeuralConditionalDensity, build_mdn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

ALPHA = 0.10  # 90% nominal credible-interval coverage
COVERAGE_P_FLOOR = 0.01  # binomial test p-value floor: below this, coverage is NOT consistent with nominal


def _fit_mdn(data, *, x_dim, y_dim, k=3, max_its=30, m_steps=80, lr=3e-3, seed=0, delta=1.0e-9, reuse_estep_ll=True):
    """Fit p(cond | target) via a mixture density network; `data` is a list of (cond, target) pairs.

    ``delta``/``reuse_estep_ll`` default to ``optimize``'s own early-stopping behavior -- right for
    the linear-Gaussian case, which converges (and stays well-calibrated) in a handful of iterations.
    The nonlinear case needs the full iteration budget (pass ``delta=None, reuse_estep_ll=False``):
    early-stopped, it undertrains and its credible intervals undercover.
    """
    module = build_mdn(x_dim=x_dim, y_dim=y_dim, k=k, hidden=32, layers=2)
    leaf = NeuralConditionalDensity(module, m_steps=m_steps, lr=lr)
    return optimize(
        data,
        leaf.estimator(),
        max_its=max_its,
        delta=delta,
        reuse_estep_ll=reuse_estep_ll,
        out=None,
        rng=np.random.RandomState(seed),
    )


def _marginal_coverage(sampler, x_test, y_test, *, n_draws=200):
    """Per-dimension credible-interval coverage: for each held-out (x, y), does the [alpha/2, 1-alpha/2]
    quantile interval of n_draws posterior samples (given y) contain the true x, per dimension?"""
    d = x_test.shape[1]
    covered = [[] for _ in range(d)]
    for i in range(len(x_test)):
        # one batched forward pass for all n_draws of THIS point, instead of n_draws individual calls
        y_batch = np.repeat(np.atleast_2d(np.asarray(y_test[i], dtype=float)), n_draws, axis=0)
        draws = np.asarray(sampler.sample_given_batch(y_batch))
        lo = np.quantile(draws, ALPHA / 2, axis=0)
        hi = np.quantile(draws, 1 - ALPHA / 2, axis=0)
        for k in range(d):
            covered[k].append(bool(lo[k] <= x_test[i, k] <= hi[k]))
    return covered


def _coverage_consistent_with_nominal(covered_flags) -> tuple[float, float]:
    """(observed_rate, p_value) for a two-sided binomial test of coverage against the nominal 1-ALPHA."""
    n = len(covered_flags)
    hits = int(sum(covered_flags))
    p = float(binomtest(hits, n, 1.0 - ALPHA).pvalue)
    return hits / n, p


# --- case 1: linear-Gaussian inverse, closed-form posterior -----------------------------------------


def _linear_gaussian_setup(d_x=2, d_y=2, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.normal(size=(d_y, d_x)) * 0.9
    sigma0 = np.eye(d_x)  # prior covariance
    r = np.eye(d_y) * 0.25  # noise covariance
    return A, sigma0, r


def _linear_gaussian_sample(A, sigma0, r, n, rng):
    d_x = sigma0.shape[0]
    x = rng.multivariate_normal(np.zeros(d_x), sigma0, size=n)
    noise = rng.multivariate_normal(np.zeros(r.shape[0]), r, size=n)
    y = x @ A.T + noise
    return x, y


def _true_posterior(A, sigma0, r, y):
    sigma0_inv = np.linalg.inv(sigma0)
    r_inv = np.linalg.inv(r)
    sigma_post = np.linalg.inv(sigma0_inv + A.T @ r_inv @ A)
    mu_post = sigma_post @ A.T @ r_inv @ y
    return mu_post, sigma_post


# --- case 2: nonlinear (bimodal) inverse, dense-grid reference posterior --------------------------

SIGMA_PRIOR, SIGMA_NOISE = 1.5, 0.3


def _nonlinear_sample(n, rng):
    x = rng.normal(0, SIGMA_PRIOR, size=(n, 1))
    y = x**2 + rng.normal(0, SIGMA_NOISE, size=(n, 1))
    return x, y


def _nonlinear_reference_posterior(y, grid=np.linspace(-5, 5, 4001)):
    logp = -0.5 * (grid / SIGMA_PRIOR) ** 2 - 0.5 * ((y - grid**2) / SIGMA_NOISE) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum() * (grid[1] - grid[0])
    mean = float(np.sum(grid * p) * (grid[1] - grid[0]))
    std = float(np.sqrt(np.sum((grid - mean) ** 2 * p) * (grid[1] - grid[0])))
    return mean, std


# --- fit both toy transports ONCE at import time; every test class (including the go/no-go report)
# reads these same fits rather than repeating ~20s of training per case. Skipped entirely (rather than
# raising on import) when torch is absent -- the classes below are marked skipUnless(_HAS_TORCH). ---

_LIN_A = _LIN_SIGMA0 = _LIN_R = None
_LIN_SAMPLER = _LIN_X_TEST = _LIN_Y_TEST = None
_NL_SAMPLER = _NL_X_TEST = _NL_Y_TEST = None

if _HAS_TORCH:
    _LIN_A, _LIN_SIGMA0, _LIN_R = _linear_gaussian_setup()
    _lin_x_train, _lin_y_train = _linear_gaussian_sample(_LIN_A, _LIN_SIGMA0, _LIN_R, 3000, np.random.RandomState(0))
    _LIN_FIT = _fit_mdn(
        [(_lin_y_train[i], _lin_x_train[i]) for i in range(len(_lin_x_train))], x_dim=2, y_dim=2, seed=0
    )
    _LIN_SAMPLER = _LIN_FIT.sampler(seed=0)
    _LIN_X_TEST, _LIN_Y_TEST = _linear_gaussian_sample(_LIN_A, _LIN_SIGMA0, _LIN_R, 150, np.random.RandomState(1))

    _nl_x_train, _nl_y_train = _nonlinear_sample(4000, np.random.RandomState(0))
    _NL_FIT = _fit_mdn(
        [(_nl_y_train[i], _nl_x_train[i]) for i in range(len(_nl_x_train))],
        x_dim=1,
        y_dim=1,
        seed=0,
        delta=None,
        reuse_estep_ll=False,
    )
    _NL_SAMPLER = _NL_FIT.sampler(seed=0)
    _NL_X_TEST, _NL_Y_TEST = _nonlinear_sample(150, np.random.RandomState(2))


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class LinearGaussianTransportTest(unittest.TestCase):
    """Metric (a) posterior fidelity and (b) calibration on the closed-form linear-Gaussian inverse."""

    @classmethod
    def setUpClass(cls):
        cls.A, cls.sigma0, cls.r = _LIN_A, _LIN_SIGMA0, _LIN_R
        cls.sampler = _LIN_SAMPLER
        cls.x_test, cls.y_test = _LIN_X_TEST, _LIN_Y_TEST

    def test_posterior_mean_and_covariance_match_closed_form(self):
        errs_mean, errs_cov = [], []
        for i in range(10):
            y = self.y_test[i]
            mu_true, sigma_true = _true_posterior(self.A, self.sigma0, self.r, y)
            y_batch = np.repeat(np.atleast_2d(np.asarray(y, dtype=float)), 400, axis=0)
            draws = np.asarray(self.sampler.sample_given_batch(y_batch))
            mu_hat, sigma_hat = draws.mean(axis=0), np.cov(draws.T)
            errs_mean.append(np.linalg.norm(mu_hat - mu_true))
            errs_cov.append(np.linalg.norm(sigma_hat - sigma_true))
        self.assertLess(float(np.mean(errs_mean)), 0.15)
        self.assertLess(float(np.mean(errs_cov)), 0.1)

    def test_credible_intervals_cover_at_nominal_rate(self):
        covered = _marginal_coverage(self.sampler, self.x_test, self.y_test, n_draws=200)
        for dim_covered in covered:
            rate, p_value = _coverage_consistent_with_nominal(dim_covered)
            self.assertGreater(p_value, COVERAGE_P_FLOOR, msg=f"observed coverage {rate} inconsistent with 90%")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class NonlinearBimodalTransportTest(unittest.TestCase):
    """The small nonlinear case: y = x^2 + noise has a genuinely BIMODAL posterior (+/- x equally
    consistent), the case a single-Gaussian conditional head cannot represent but a mixture can."""

    @classmethod
    def setUpClass(cls):
        cls.sampler = _NL_SAMPLER
        cls.x_test, cls.y_test = _NL_X_TEST, _NL_Y_TEST

    def test_posterior_matches_dense_grid_reference_and_is_bimodal(self):
        y_probe = np.array([1.0])  # a moderate y: the true posterior has real mass at BOTH x ~= +1 and x ~= -1
        ref_mean, ref_std = _nonlinear_reference_posterior(y_probe[0])
        y_batch = np.repeat(np.atleast_2d(np.asarray(y_probe, dtype=float)), 2000, axis=0)
        draws = np.asarray(self.sampler.sample_given_batch(y_batch))[:, 0]

        self.assertLess(abs(float(draws.mean()) - ref_mean), 0.15)
        self.assertLess(abs(float(draws.std()) - ref_std), 0.15)

        near_pos = float(np.mean(np.abs(draws - 1.0) < 0.5))
        near_neg = float(np.mean(np.abs(draws + 1.0) < 0.5))
        self.assertGreater(near_pos, 0.2)  # both modes carry real mass -- NOT collapsed to one side
        self.assertGreater(near_neg, 0.2)

    def test_credible_intervals_cover_at_nominal_rate(self):
        covered = _marginal_coverage(self.sampler, self.x_test, self.y_test, n_draws=200)
        rate, p_value = _coverage_consistent_with_nominal(covered[0])
        self.assertGreater(p_value, COVERAGE_P_FLOOR, msg=f"observed coverage {rate} inconsistent with 90%")


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TransportProofGoNoGoTest(unittest.TestCase):
    """The card's own required deliverable: the report states the go/no-go explicitly.

    Reads the same module-level fits the other two test classes use (no repeated ~20s-per-case
    training) so the decision reflects exactly what those tests verified.
    """

    def test_go_no_go_report(self):
        lin_covered = _marginal_coverage(_LIN_SAMPLER, _LIN_X_TEST, _LIN_Y_TEST, n_draws=200)
        lin_rates = [_coverage_consistent_with_nominal(c) for c in lin_covered]

        nl_covered = _marginal_coverage(_NL_SAMPLER, _NL_X_TEST, _NL_Y_TEST, n_draws=200)
        nl_rate = _coverage_consistent_with_nominal(nl_covered[0])

        go = all(p > COVERAGE_P_FLOOR for _, p in lin_rates) and nl_rate[1] > COVERAGE_P_FLOOR
        decision = "GO" if go else "NO-GO"
        print(f"\nF0 TRANSPORT-PROOF GATE DECISION: {decision}")
        print(f"  linear-Gaussian per-dim coverage: {[round(r_, 3) for r_, _ in lin_rates]} (nominal {1 - ALPHA})")
        print(f"  nonlinear bimodal coverage: {round(nl_rate[0], 3)} (nominal {1 - ALPHA})")
        self.assertTrue(go, "F0 gate is NO-GO -- record notes/f0-transport-negative.md and do not build F1-F7")


if __name__ == "__main__":
    unittest.main()
