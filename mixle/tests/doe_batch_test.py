"""Rigorous batch (multi-point) Bayesian optimization (mixle.doe.batch)."""

import importlib.util
import unittest
import warnings

import numpy as np
from scipy.stats import norm

from mixle.doe.batch import monte_carlo_qei

HAS_TORCH = importlib.util.find_spec("torch") is not None  # the proposal drivers fit the torch GP surrogate


class MonteCarloQeiTest(unittest.TestCase):
    """The q-EI estimator is pure NumPy and checkable in closed form."""

    def test_q1_matches_analytic_ei(self):
        best, mu, sigma = 1.0, 0.5, 0.8
        z = (best - mu) / sigma
        analytic = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)  # EI, minimization
        mc = monte_carlo_qei([mu], [[sigma**2]], best, maximize=False, samples=200000, seed=0)
        self.assertAlmostEqual(mc, analytic, places=2)

    def test_correlated_duplicate_gives_no_batch_gain(self):
        # two perfectly-correlated identical points have the q-EI of a single point (the property
        # kriging-believer violates): batching duplicates is worthless.
        best, mu, var = 1.0, 0.5, 0.64
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        dup = monte_carlo_qei([mu, mu], [[var, var], [var, var]], best, samples=200000, seed=0)
        self.assertAlmostEqual(dup, single, places=2)

    def test_independent_points_increase_qei(self):
        best, mu, var = 1.0, 0.5, 0.64
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        indep = monte_carlo_qei([mu, mu], [[var, 0.0], [0.0, var]], best, samples=200000, seed=0)
        self.assertGreater(indep, single)


class SafeCholeskyValidationTest(unittest.TestCase):
    """MXR-080-0166: _safe_cholesky (via monte_carlo_qei) used to replace ANY covariance it couldn't
    factor -- including a genuinely invalid one -- with its diagonal, silently downgrading the module's
    central "true joint posterior" claim to an independent approximation while still returning a
    plausible-looking number. It must now distinguish "not a valid covariance at all" (reject) from
    "valid but numerically singular" (heal with small, quantified, reported jitter) and never blur the
    two.
    """

    def test_indefinite_covariance_is_rejected_not_silently_downgraded(self):
        # the audit's exact example: [[1,2],[2,1]] has eigenvalues 3 and -1 (ac=1 < b^2=4) -- not a
        # covariance matrix at all, for ANY jitter budget. Must raise, not quietly fall back to diagonal.
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            monte_carlo_qei([0.0, 0.0], [[1, 2], [2, 1]], best=1.0, samples=1000, seed=0)

    def test_a_severely_non_psd_covariance_is_also_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            monte_carlo_qei([0.0, 0.0], [[-1.0, 0.0], [0.0, -1.0]], best=1.0, samples=1000, seed=0)

    def test_near_singular_covariance_is_healed_with_quantified_reported_jitter(self):
        # an exactly rank-deficient (perfectly correlated duplicate) covariance IS a legitimate joint
        # covariance -- min eigenvalue 0.0, not negative -- but a direct Cholesky factorization fails on
        # it, so this must heal via jitter rather than raise. The healing must not be invisible: a
        # RuntimeWarning should report it, and the jitter must be small (a numerical nudge, not a change
        # of model) relative to the covariance's own scale.
        best, mu, var = 1.0, 0.5, 0.64
        cov = [[var, var], [var, var]]  # exactly singular: eigenvalues are 0 and 2*var
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = monte_carlo_qei([mu, mu], cov, best, samples=200000, seed=0)
        jitter_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(jitter_warnings), 1)
        message = str(jitter_warnings[0].message)
        self.assertIn("jitter=", message)
        reported_jitter = float(message.split("jitter=")[1].split()[0])
        self.assertGreater(reported_jitter, 0.0)
        self.assertLess(reported_jitter, 1e-4 * var)  # a small numerical nudge, not a material perturbation
        # healed value must still reflect the TRUE JOINT computation (perfectly-correlated duplicate has
        # the single-point q-EI), not the independent-model value the old diagonal fallback would give.
        single = monte_carlo_qei([mu], [[var]], best, samples=200000, seed=0)
        indep = monte_carlo_qei([mu, mu], [[var, 0.0], [0.0, var]], best, samples=200000, seed=0)
        self.assertAlmostEqual(val, single, places=2)
        self.assertGreater(indep - single, 0.01)  # confirms independent really is a different number here

    def test_well_conditioned_covariance_needs_no_jitter(self):
        # negative control: a genuinely well-conditioned covariance must factor on the first, unperturbed
        # attempt -- no warning, and the usual closed-form-matching value from test_q1_matches_analytic_ei.
        best, mu, sigma = 1.0, 0.5, 0.8
        z = (best - mu) / sigma
        analytic = (best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            val = monte_carlo_qei([mu], [[sigma**2]], best, samples=200000, seed=0)
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
        self.assertAlmostEqual(val, analytic, places=2)

    def test_non_finite_covariance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            monte_carlo_qei([0.0, 0.0], [[1.0, 0.0], [0.0, np.nan]], best=1.0, samples=100, seed=0)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class BatchProposalTest(unittest.TestCase):
    def _problem(self):
        rng = np.random.RandomState(0)
        x = rng.uniform(-3, 3, (12, 1))
        y = np.sin(3 * x[:, 0]) + 0.3 * x[:, 0] ** 2  # two basins
        return x, y, [(-3.0, 3.0)]

    def test_qei_batch_is_diverse(self):
        from mixle.doe import propose_qei_batch

        x, y, bounds = self._problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = propose_qei_batch(x, y, bounds, q=3, n_candidates=120, mc_samples=128, seed=1)
        self.assertEqual(b.shape, (3, 1))
        self.assertTrue((b >= -3).all() and (b <= 3).all())
        pdist = [abs(b[i, 0] - b[j, 0]) for i in range(3) for j in range(i + 1, 3)]
        self.assertGreater(min(pdist), 0.1)  # not near-duplicate points

    def test_local_penalization_is_diverse(self):
        from mixle.doe import propose_local_penalization

        x, y, bounds = self._problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = propose_local_penalization(x, y, bounds, q=3, n_candidates=400, seed=1)
        self.assertEqual(b.shape, (3, 1))
        pdist = [abs(b[i, 0] - b[j, 0]) for i in range(3) for j in range(i + 1, 3)]
        self.assertGreater(min(pdist), 0.3)  # floored spacing keeps the batch spread


if __name__ == "__main__":
    unittest.main()
