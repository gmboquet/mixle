"""Active learning and Bayesian optimal design (mixle.doe.active)."""

import importlib.util
import unittest
import warnings

import numpy as np

from mixle.doe.active import (
    active_learning_design,
    alc_scores,
    alm_scores,
    expected_information_gain_linear,
    expected_information_gain_nmc,
    propose_active_learning,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ExpectedInformationGainTest(unittest.TestCase):
    def test_linear_eig_prefers_spread_design(self):
        spread = np.array([[1, -1.0], [1, -0.33], [1, 0.33], [1, 1.0]])
        clustered = np.array([[1, -0.05], [1, 0.0], [1, 0.02], [1, 0.05]])
        self.assertGreater(
            expected_information_gain_linear(spread, noise=0.5),
            expected_information_gain_linear(clustered, noise=0.5),
        )

    def test_large_p_small_n_does_not_warn_and_matches_the_pxp_formulation(self):
        """Regression test for a real bug (found via experiments/adaptive-gravity-survey-design):
        scoring a handful of candidate observations (n small) against a model with many parameters
        (p large -- e.g. hundreds of grid cells) used to form the full dense (p, p) matrix and throw
        spurious divide-by-zero/overflow RuntimeWarnings from slogdet. By Sylvester's identity the
        (n, n) formulation is mathematically identical and must actually be used when it's smaller."""
        rng = np.random.default_rng(0)
        p, n = 720, 4
        f = rng.normal(size=(n, p)) * 1e-5
        prior_cov = np.eye(p) * (350.0**2)
        noise = 0.02

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning fails the test, not just RuntimeWarning
            value = expected_information_gain_linear(f, noise=noise, prior_cov=prior_cov)

        # cross-check against the direct (p, p) formulation computed independently in the test itself
        # (not by calling back into the function under test) -- confirms the fast path is not just
        # silent, it's correct.
        m_pxp = np.eye(p) + (prior_cov @ (f.T @ f)) / (noise**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the (p, p) path is EXPECTED to warn; that's the bug being fixed
            _, logdet_pxp = np.linalg.slogdet(m_pxp)
        self.assertAlmostEqual(value, 0.5 * logdet_pxp, places=6)

    def test_small_p_large_n_still_matches_manual_pxp_computation(self):
        """The other direction (n > p, the common textbook case) must still work -- confirms the
        n<=p branch selection didn't regress the existing default path."""
        rng = np.random.default_rng(1)
        p, n = 3, 20
        f = rng.normal(size=(n, p))
        prior_cov = np.eye(p) * 2.0
        noise = 0.5
        value = expected_information_gain_linear(f, noise=noise, prior_cov=prior_cov)
        m_pxp = np.eye(p) + (prior_cov @ (f.T @ f)) / (noise**2)
        _, logdet_pxp = np.linalg.slogdet(m_pxp)
        self.assertAlmostEqual(value, 0.5 * logdet_pxp, places=9)

    def test_nmc_matches_linear_gaussian_closed_form(self):
        f, sigma = 1.5, 0.7
        analytic = 0.5 * np.log(1 + f**2 / sigma**2)

        def prior(rng, n):
            return rng.standard_normal((n, 1))

        def loglik(thetas, y):
            return -0.5 * ((y - thetas[:, 0] * f) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

        def sim(theta, rng):
            return np.array([theta[0] * f + sigma * rng.standard_normal()])

        nmc = expected_information_gain_nmc(prior, loglik, sim, n_outer=4000, n_inner=4000, seed=0)
        self.assertAlmostEqual(nmc, analytic, delta=0.05)


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class ActiveLearningTest(unittest.TestCase):
    def test_alm_proposes_into_the_data_gap(self):
        from mixle.doe import propose_active_learning

        x = np.array([[-2.0], [-1.8], [-1.6], [1.6], [1.8], [2.0]])  # gap in the middle
        y = np.sin(x[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xn = propose_active_learning(x, y, [(-2.0, 2.0)], method="alm", n_candidates=200, seed=1)
        self.assertLess(abs(xn[0]), 1.0)  # placed where uncertainty is highest

    def test_active_learning_loop_runs(self):
        from mixle.doe import active_learning_design

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            des = active_learning_design(
                lambda x: float(np.sin(3 * x[0])), [(-2.0, 2.0)], n_init=6, max_evals=14, method="alc", seed=2
            )
        self.assertEqual(des["X"].shape[0], 14)


class ActiveLearningBudgetValidationTest(unittest.TestCase):
    """MXR-080-0158: n_init/max_evals joint validation and exact positive counts.

    These raise before any (torch-dependent) GP fit is ever attempted, so -- unlike ActiveLearningTest
    above -- none of them need HAS_TORCH.
    """

    def test_rejects_n_init_greater_than_max_evals(self):
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)], n_init=10, max_evals=5, seed=0)

    def test_rejects_n_init_one_more_than_max_evals(self):
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)], n_init=6, max_evals=5, seed=0)

    def test_rejects_fractional_n_init(self):
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)], n_init=2.5, max_evals=10, seed=0)

    def test_rejects_fractional_max_evals(self):
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)], n_init=2, max_evals=10.5, seed=0)

    def test_rejects_explicit_zero_n_init_instead_of_silently_defaulting(self):
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)], n_init=0, max_evals=10, seed=0)

    def test_rejects_zero_max_evals(self):
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)], n_init=1, max_evals=0, seed=0)

    def test_rejects_negative_max_evals(self):
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)], n_init=1, max_evals=-3, seed=0)

    def test_default_n_init_is_still_subject_to_the_budget_check(self):
        # 3 dims -> default n_init = 2*3 = 6, which must itself fit within max_evals.
        with self.assertRaises(ValueError):
            active_learning_design(lambda x: float(np.sum(x)), [(-1.0, 1.0)] * 3, max_evals=4, seed=0)

    def test_propose_active_learning_rejects_fractional_n_candidates(self):
        rng = np.random.RandomState(0)
        x, y = rng.uniform(0, 1, size=(5, 2)), rng.normal(size=5)
        with self.assertRaises(ValueError):
            propose_active_learning(x, y, [(0.0, 1.0), (0.0, 1.0)], n_candidates=10.5)

    def test_propose_active_learning_rejects_zero_n_candidates(self):
        rng = np.random.RandomState(0)
        x, y = rng.uniform(0, 1, size=(5, 2)), rng.normal(size=5)
        with self.assertRaises(ValueError):
            propose_active_learning(x, y, [(0.0, 1.0), (0.0, 1.0)], n_candidates=0)

    def test_propose_active_learning_rejects_zero_n_reference(self):
        rng = np.random.RandomState(0)
        x, y = rng.uniform(0, 1, size=(5, 2)), rng.normal(size=5)
        with self.assertRaises(ValueError):
            propose_active_learning(x, y, [(0.0, 1.0), (0.0, 1.0)], method="alc", n_reference=0)

    def test_propose_active_learning_rejects_fractional_n_reference(self):
        rng = np.random.RandomState(0)
        x, y = rng.uniform(0, 1, size=(5, 2)), rng.normal(size=5)
        with self.assertRaises(ValueError):
            propose_active_learning(x, y, [(0.0, 1.0), (0.0, 1.0)], method="alc", n_reference=3.5)

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_propose_active_learning_alm_ignores_invalid_n_reference(self):
        # n_reference is irrelevant to method='alm' (only 'alc' integrates over a reference set) and
        # must not be spuriously validated/required when it will never be used.
        rng = np.random.RandomState(0)
        x, y = rng.uniform(0, 1, size=(5, 2)), rng.normal(size=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            point = propose_active_learning(
                x, y, [(0.0, 1.0), (0.0, 1.0)], method="alm", n_candidates=16, n_reference=0, seed=0
            )
        self.assertEqual(point.shape, (2,))

    @unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
    def test_negative_control_valid_budget_runs_exactly_to_max_evals(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            des = active_learning_design(
                lambda x: float(np.sin(3 * x[0])), [(-2.0, 2.0)], n_init=4, max_evals=9, method="alc", seed=11
            )
        self.assertEqual(des["X"].shape[0], 9)
        self.assertEqual(des["Y"].shape[0], 9)
        self.assertTrue(np.all(np.isfinite(des["Y"])))


class _StubGP:
    """A minimal duck-typed GP surrogate: ``.predict`` returns a caller-supplied ``(mean, cov)`` pair,
    so alc_scores/alm_scores can be unit-tested directly (including non-finite-covariance inputs that
    would be awkward to provoke from a real fit) without fitting a real, torch-based GP."""

    def __init__(self, cov):
        self._cov = np.asarray(cov, dtype=np.float64)

    def predict(self, x, y, x_new, return_cov=False):
        n = np.asarray(x_new).shape[0]
        mean = np.zeros(n)
        return (mean, self._cov) if return_cov else mean


class AlcAlmScoresValidationTest(unittest.TestCase):
    """MXR-080-0158: alc_scores/alm_scores reject empty candidate/reference sets and non-finite merits
    directly -- the actual root cause of the all-zero-merits/arbitrary-argmax bug, guarded here
    independently of whatever count validation propose_active_learning performs upstream (these two
    functions are public, in __all__, and callable directly -- e.g. mixle.task.emulate does)."""

    def test_alc_scores_rejects_empty_reference(self):
        gp = _StubGP(np.eye(3))
        with self.assertRaises(ValueError):
            alc_scores(gp, None, None, np.array([[0.1], [0.2], [0.3]]), np.empty((0, 1)))

    def test_alc_scores_rejects_empty_candidates(self):
        gp = _StubGP(np.eye(2))
        with self.assertRaises(ValueError):
            alc_scores(gp, None, None, np.empty((0, 1)), np.array([[0.1], [0.2]]))

    def test_alm_scores_rejects_empty_candidates(self):
        gp = _StubGP(np.empty((0, 0)))
        with self.assertRaises(ValueError):
            alm_scores(gp, None, None, np.empty((0, 1)))

    def test_alc_scores_rejects_non_finite_merits(self):
        n_ref, n_cand = 2, 3
        cov = np.eye(n_ref + n_cand)
        cov[0, n_ref] = np.inf  # a non-finite cross-covariance between reference[0] and candidate[0]
        gp = _StubGP(cov)
        with self.assertRaises(ValueError):
            alc_scores(gp, None, None, np.zeros((n_cand, 1)), np.zeros((n_ref, 1)))

    def test_alc_scores_negative_control_nonempty_reference_gives_finite_nonzero_merits(self):
        rng = np.random.RandomState(0)
        n_ref, n_cand = 4, 5
        n = n_ref + n_cand
        a = rng.normal(size=(n, n))
        cov = a @ a.T + np.eye(n) * 0.1  # a genuine, well-conditioned PD covariance
        gp = _StubGP(cov)
        scores = alc_scores(gp, None, None, np.zeros((n_cand, 1)), np.zeros((n_ref, 1)))
        self.assertEqual(scores.shape, (n_cand,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.any(scores > 0))

    def test_alm_scores_negative_control_gives_finite_merits(self):
        rng = np.random.RandomState(0)
        n_cand = 5
        a = rng.normal(size=(n_cand, n_cand))
        cov = a @ a.T + np.eye(n_cand) * 0.1
        gp = _StubGP(cov)
        scores = alm_scores(gp, None, None, np.zeros((n_cand, 1)))
        self.assertEqual(scores.shape, (n_cand,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))


if __name__ == "__main__":
    unittest.main()
