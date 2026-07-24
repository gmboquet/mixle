"""Forward uncertainty propagation: Monte Carlo + unscented transform (Phase 4)."""

import unittest
import warnings

import numpy as np

from mixle.doe import propagate, unscented_transform


class PropagateTest(unittest.TestCase):
    def setUp(self):
        self.a = np.array([1.0, -2.0, 0.5])
        self.mu = np.array([1.0, 2.0, 3.0])
        rng = np.random.RandomState(0)
        chol = rng.randn(3, 3)
        self.cov = chol @ chol.T + np.eye(3)
        self.f = lambda x: x @ self.a
        self.true_mean = self.a @ self.mu
        self.true_var = self.a @ self.cov @ self.a

    def test_unscented_is_exact_for_linear_models(self):
        out = propagate(self.f, self.mu, self.cov, method="unscented")
        self.assertAlmostEqual(out["mean"], self.true_mean, places=8)
        self.assertAlmostEqual(out["std"] ** 2, self.true_var, places=6)

    def test_monte_carlo_matches_linear_moments(self):
        out = propagate(self.f, self.mu, self.cov, n=200000, method="montecarlo", seed=1)
        self.assertAlmostEqual(out["mean"], self.true_mean, delta=0.05)
        self.assertAlmostEqual(out["std"], np.sqrt(self.true_var), delta=0.05)

    def test_monte_carlo_and_unscented_agree_on_mild_nonlinearity(self):
        g = lambda x: x[:, 0] ** 2 + x[:, 1]
        mu, cov = np.array([1.0, 0.0]), 0.1 * np.eye(2)
        mc = propagate(g, mu, cov, n=300000, method="montecarlo", seed=2)
        ut = propagate(g, mu, cov, method="unscented")
        self.assertAlmostEqual(mc["mean"], 1.1, delta=0.02)  # E[x1^2 + x2] = (1 + 0.1) + 0
        self.assertAlmostEqual(ut["mean"], 1.1, delta=0.02)
        self.assertAlmostEqual(mc["std"], ut["std"], delta=0.03)

    def test_monte_carlo_quantiles_are_ordered(self):
        out = propagate(self.f, self.mu, self.cov, n=50000, method="montecarlo", seed=3)
        q = out["quantiles"]
        self.assertLess(q[0.05], q[0.5])
        self.assertLess(q[0.5], q[0.95])

    def test_vector_output_covariance(self):
        h = lambda x: np.stack([x[:, 0] + x[:, 1], x[:, 0] - x[:, 1]], axis=1)
        m, c = unscented_transform(h, np.array([0.0, 0.0]), np.eye(2))
        self.assertEqual(np.shape(m), (2,))
        np.testing.assert_allclose(c, [[2.0, 0.0], [0.0, 2.0]], atol=1e-6)  # var(x0+x1)=2, uncorrelated

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            propagate(self.f, self.mu, self.cov, method="bogus")


class UnscentedCovarianceValidationTest(unittest.TestCase):
    """MXR-080-0190: ``_safe_cholesky`` (via ``unscented_transform``) used to replace ANY covariance it
    couldn't factor -- including a genuinely invalid one -- with its diagonal, silently discarding both
    the off-diagonal dependence structure and the fact the input wasn't even PSD, while still returning a
    plausible-looking number. Mirrors ``mixle.doe.batch._safe_cholesky``'s MXR-080-0166 fix (commit
    8dedcd54): reject invalid covariances outright, heal merely-singular ones with small, quantified,
    reported jitter, and never fall back to a diagonal (independent) approximation.
    """

    def test_indefinite_covariance_is_rejected_not_silently_downgraded(self):
        # the audit's exact example: [[1,2],[2,1]] has eigenvalues 3 and -1 (ac=1 < b^2=4) -- not a
        # covariance matrix at all, for ANY jitter budget. Pre-fix this returned (0.0, 2.0): identical to
        # what np.eye(2) produces (see the next test) -- i.e. silently "as if it were identity".
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            unscented_transform(lambda x: x.sum(axis=1), np.array([0.0, 0.0]), np.array([[1.0, 2.0], [2.0, 1.0]]))

    def test_identity_covariance_is_the_actual_identity_case(self):
        # grounds the "as if it were identity" claim above: this is what an actually-identity input gives.
        m, c = unscented_transform(lambda x: x.sum(axis=1), np.array([0.0, 0.0]), np.eye(2))
        self.assertAlmostEqual(m, 0.0)
        self.assertAlmostEqual(c, 2.0)

    def test_indefinite_covariance_rejected_through_propagate_unscented(self):
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            propagate(
                lambda x: x.sum(axis=1), np.array([0.0, 0.0]), np.array([[1.0, 2.0], [2.0, 1.0]]), method="unscented"
            )

    def test_indefinite_covariance_rejected_through_propagate_montecarlo(self):
        # the same invalid matrix must be refused by the OTHER propagation method too, not just warned
        # about and silently sampled from -- numpy's own multivariate_normal(check_valid='warn') default
        # only emits an easy-to-miss warning and proceeds anyway.
        with self.assertRaisesRegex(ValueError, "positive semi-definite"):
            propagate(
                lambda x: x.sum(axis=1),
                np.array([0.0, 0.0]),
                np.array([[1.0, 2.0], [2.0, 1.0]]),
                method="montecarlo",
                n=1000,
            )

    def test_near_singular_covariance_is_healed_with_quantified_reported_jitter(self):
        # a perfectly-correlated-duplicate covariance IS a legitimate joint covariance (min eigenvalue
        # 0.0, not negative) but fails a direct Cholesky factorization -- must heal via jitter, not raise.
        # alpha=0.5, kappa=2.0 (d=2) make the transform's internal (d+lambda) scale factor exactly 1.0 in
        # floating point, so the matrix _safe_cholesky actually receives is bit-identical to `cov` --
        # isolating the healing behavior from the transform's own (unrelated) scaling of its input.
        var = 0.64
        cov = [[var, var], [var, var]]  # exactly singular: eigenvalues 0 and 2*var
        mean = [0.5, 0.5]
        f = lambda x: x[:, 0] + x[:, 1]  # noqa: E731
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m, c = unscented_transform(f, np.array(mean), np.array(cov), alpha=0.5, kappa=2.0)
        jitter_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(jitter_warnings), 1)
        message = str(jitter_warnings[0].message)
        self.assertIn("jitter=", message)
        reported_jitter = float(message.split("jitter=")[1].split()[0])
        self.assertGreater(reported_jitter, 0.0)
        self.assertLess(reported_jitter, 1e-4 * var)  # a small numerical nudge, not a material perturbation
        # healed value must reflect the TRUE JOINT covariance (var(x0+x1)=4*var for perfectly correlated
        # x0,x1), not the independent-model value (var(x0+x1)=2*var) a diagonal fallback would give.
        self.assertAlmostEqual(m, 1.0, places=6)
        self.assertAlmostEqual(c, 4 * var, places=4)
        indep_m, indep_c = unscented_transform(f, np.array(mean), np.diag([var, var]), alpha=0.5, kappa=2.0)
        self.assertAlmostEqual(indep_c, 2 * var, places=6)
        self.assertGreater(c - indep_c, 0.1)  # confirms independent really is a different number here

    def test_well_conditioned_covariance_needs_no_jitter(self):
        # negative control: a genuinely well-conditioned covariance must factor on the first, unperturbed
        # attempt -- no warning, and the ordinary closed-form linear-model result.
        a = np.array([1.0, -2.0, 0.5])
        mu = np.array([1.0, 2.0, 3.0])
        rng = np.random.RandomState(0)
        chol = rng.randn(3, 3)
        cov = chol @ chol.T + np.eye(3)
        f = lambda x: x @ a  # noqa: E731
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m, c = unscented_transform(f, mu, cov)
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])
        self.assertAlmostEqual(m, a @ mu, places=8)
        self.assertAlmostEqual(c, a @ cov @ a, places=6)

    def test_non_finite_covariance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            unscented_transform(lambda x: x.sum(axis=1), np.array([0.0, 0.0]), np.array([[1.0, 0.0], [0.0, np.nan]]))


if __name__ == "__main__":
    unittest.main()
