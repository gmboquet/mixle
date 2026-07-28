"""Max-value Entropy Search information-theoretic BO (mixle.doe.entropy)."""

import importlib.util
import unittest
import warnings

import numpy as np

from mixle.doe.entropy import (
    _fit_gumbel,
    _gumbel_quantile,
    max_value_entropy_search,
    propose_mes,
    sample_max_values,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class MaxValueEntropyTest(unittest.TestCase):
    def test_y_star_samples_above_best_mean(self):
        mu = np.array([0.0, 0.5, 0.9, 1.0])
        sd = np.array([0.5, 0.5, 0.5, 0.01])
        ystar = sample_max_values(mu, sd, 500, seed=0)
        self.assertGreaterEqual(ystar.min(), mu.max() - 1e-9)  # the max is never below the best mean

    def test_information_is_nonnegative_and_favors_uncertainty(self):
        mu = np.array([0.0, 0.5, 0.9, 1.0])
        sd = np.array([0.5, 0.5, 0.5, 0.01])
        ystar = sample_max_values(mu, sd, 500, seed=0)
        mes = max_value_entropy_search(mu, sd, ystar, maximize=True)
        self.assertTrue(np.all(mes >= -1e-9))
        # an uncertain near-optimal candidate (index 2: mu=0.9, sd=0.5) beats both a near-certain one
        # (index 3: mu=1.0, sd=0.01) and a certain-but-clearly-worse one (index 0: mu=0.0, sd=0.5) --
        # this is the comparison MXR-080-0177 (the Gumbel loc sign fix) made robust: it holds stably
        # across seeds and sample counts, unlike the previous index-1-vs-3 comparison this replaces,
        # which only passed by coincidence under the old, miscalibrated y* distribution.
        self.assertGreater(mes[2], mes[3])
        self.assertGreater(mes[2], mes[0])


class GumbelFitTest(unittest.TestCase):
    """MXR-080-0177: the Gumbel quantile fit used the wrong sign for ``loc``.

    ``y_r = loc - scale*C_r`` solves to ``loc = y50 + scale*C50`` at the median; the code instead
    computed ``loc = y50 - scale*C50``, which -- plugged back into the quantile function -- recovers a
    fitted median of ``y50 - 2*scale*C50`` rather than ``y50`` itself.
    """

    def test_inverse_cdf_recovers_known_quantiles(self):
        # Fit against quantiles generated from known Gumbel(loc0, scale0) pairs (not derived from the
        # fit itself) and confirm the fit recovers the exact loc/scale, then that plugging the fit back
        # into the quantile function reproduces the inputs -- the standard way to validate a
        # quantile-matching fit is self-consistent. Also doubles as the negative control that the fit
        # still works for non-degenerate cases beyond the single-candidate audit scenario below,
        # including a negative loc and a large-scale case.
        for loc0, scale0 in [(2.0, 3.0), (-1.5, 0.7), (0.0, 1.0), (100.0, 25.0)]:
            with self.subTest(loc0=loc0, scale0=scale0):
                y25 = float(_gumbel_quantile(loc0, scale0, 0.25))
                y50 = float(_gumbel_quantile(loc0, scale0, 0.50))
                y75 = float(_gumbel_quantile(loc0, scale0, 0.75))
                loc, scale = _fit_gumbel(y25, y50, y75)
                self.assertAlmostEqual(loc, loc0, places=9)
                self.assertAlmostEqual(scale, scale0, places=9)
                for r, y in [(0.25, y25), (0.50, y50), (0.75, y75)]:
                    self.assertAlmostEqual(float(_gumbel_quantile(loc, scale, r)), y, places=9)

    def test_median_recovery_matches_audit_reproduction(self):
        # The audit's own scenario: a single standard-normal candidate has an exactly known max-value
        # distribution (the max over one element is just that element), i.e. y* ~ N(0, 1) with true
        # median 0. Before the fix, 200,000 sampled y* had empirical median ~0.63 (the wrong loc sign
        # shifts the fitted quantile function's own median by -2*scale*C50, which is about +0.63 here);
        # after the fix it recovers ~0.
        ystar = sample_max_values(0.0, 1.0, 200_000, seed=0)
        self.assertAlmostEqual(float(np.median(ystar)), 0.0, delta=0.05)


class MesValidationTest(unittest.TestCase):
    """MXR-080-0178: MES silently accepted empty/mismatched/non-finite moments and optimum samples,
    silently floored a negative std to the same epsilon used for a legitimately tiny one, and could
    silently produce NaN information instead of raising.
    """

    # -- sample_max_values: posterior moments --

    def test_sample_max_values_rejects_empty_moments(self):
        with self.assertRaises(ValueError):
            sample_max_values(np.array([]), np.array([]), 10, seed=0)

    def test_sample_max_values_rejects_mismatched_moments(self):
        with self.assertRaises(ValueError):
            sample_max_values(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0]), 10, seed=0)

    def test_sample_max_values_rejects_non_finite_mean(self):
        with self.assertRaises(ValueError):
            sample_max_values(np.array([0.0, np.nan]), np.array([1.0, 1.0]), 10, seed=0)

    def test_sample_max_values_rejects_non_finite_std(self):
        with self.assertRaises(ValueError):
            sample_max_values(np.array([0.0, 1.0]), np.array([1.0, np.inf]), 10, seed=0)

    def test_sample_max_values_rejects_negative_std(self):
        # Distinct from the legitimate-tiny-std case below: a negative std is never valid (a standard
        # deviation cannot be negative), so it must be rejected outright, not silently floored to the
        # same epsilon used for a genuinely tiny-but-valid std.
        with self.assertRaises(ValueError):
            sample_max_values(np.array([0.0]), np.array([-5.0]), 10, seed=0)

    def test_sample_max_values_still_floors_legitimate_tiny_std(self):
        # Zero (and near-zero) std IS legitimate -- a candidate with no predictive uncertainty -- and
        # must still be floored for numerical stability rather than rejected.
        ystar = sample_max_values(np.array([0.0, 1.0]), np.array([0.0, 1e-15]), 200, seed=0)
        self.assertTrue(np.all(np.isfinite(ystar)))

    def test_sample_max_values_rejects_nonpositive_n_samples(self):
        for bad in (0, -3, 2.5):
            with self.subTest(n_samples=bad):
                with self.assertRaises(ValueError):
                    sample_max_values(np.array([0.0]), np.array([1.0]), bad, seed=0)

    def test_sample_max_values_rejects_unrepresentable_finite_bracket(self):
        with self.assertRaisesRegex(ValueError, "bracket"):
            sample_max_values(np.array([1e308]), np.array([1e308]), 10, seed=0)

    # -- max_value_entropy_search: posterior moments and optimum samples --

    def test_mes_rejects_empty_moments(self):
        with self.assertRaises(ValueError):
            max_value_entropy_search(np.array([]), np.array([]), np.array([1.0]))

    def test_mes_rejects_mismatched_moments(self):
        with self.assertRaises(ValueError):
            max_value_entropy_search(np.array([0.0, 1.0]), np.array([1.0]), np.array([1.0]))

    def test_mes_rejects_non_finite_mean(self):
        with self.assertRaises(ValueError):
            max_value_entropy_search(np.array([0.0, np.nan]), np.array([1.0, 1.0]), np.array([1.0]))

    def test_mes_rejects_negative_std(self):
        with self.assertRaises(ValueError):
            max_value_entropy_search(np.array([0.0]), np.array([-1.0]), np.array([1.0]))

    def test_mes_still_floors_legitimate_tiny_std(self):
        mes = max_value_entropy_search(np.array([0.0, 1.0]), np.array([0.0, 1e-15]), np.array([1.0, 2.0, 3.0]))
        self.assertTrue(np.all(np.isfinite(mes)))

    def test_mes_rejects_empty_max_samples(self):
        with self.assertRaises(ValueError):
            max_value_entropy_search(np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([]))

    def test_mes_rejects_non_finite_max_samples(self):
        with self.assertRaises(ValueError):
            max_value_entropy_search(np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([1.0, np.nan]))

    def test_mes_rejects_non_finite_information_from_extreme_but_finite_inputs(self):
        # mean, std, and max_samples are all individually finite, but gamma = (y* - mu)/sd overflows to
        # +/-inf, and inf * 0 in the gamma*pdf term silently produces NaN -- this must be rejected, not
        # returned to the caller (who would otherwise feed NaN into an argmax over candidates).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the gamma overflow is EXPECTED here; that's the bug being fixed
            with self.assertRaises(ValueError):
                max_value_entropy_search(np.array([1e300]), np.array([1e-9]), np.array([-1e300]), maximize=True)

    def test_mes_well_posed_call_is_sensible_and_finite(self):
        # Negative control: a normal, well-posed computation with legitimate posterior moments still
        # produces sensible, finite entropy/information values for every candidate.
        mu = np.array([0.0, 0.5, 0.9, 1.0])
        sd = np.array([0.5, 0.5, 0.5, 0.01])
        ystar = sample_max_values(mu, sd, 500, seed=0)
        mes = max_value_entropy_search(mu, sd, ystar, maximize=True)
        self.assertTrue(np.all(np.isfinite(mes)))
        self.assertTrue(np.all(mes >= -1e-9))


class MesProposalContractTest(unittest.TestCase):
    class _StubGP:
        def __init__(self, mean, covariance):
            self.mean = mean
            self.covariance = covariance

        def fit(self, x, y, **kwargs):
            return self

        def predict(self, x, y, points, return_cov=True):
            return self.mean, self.covariance

    def setUp(self):
        self.x = np.array([[-1.0], [0.0], [1.0]])
        self.y = np.array([1.0, 0.0, 1.0])
        self.bounds = [(-1.0, 1.0)]

    def test_rejects_fractional_and_boolean_candidate_counts(self):
        for invalid in (2.9, True, np.bool_(True)):
            with self.assertRaises((TypeError, ValueError)):
                propose_mes(self.x, self.y, self.bounds, n_candidates=invalid)

    def test_rejects_malformed_mean_and_covariance(self):
        invalid_posteriors = (
            (np.zeros(3), np.eye(2)),
            (np.zeros(2), np.array([[1.0, 0.2], [0.0, 1.0]])),
            (np.zeros(2), np.diag([1.0, -1.0])),
            (np.full(2, np.nan), np.eye(2)),
        )
        for mean, covariance in invalid_posteriors:
            with self.assertRaises(ValueError):
                propose_mes(
                    self.x,
                    self.y,
                    self.bounds,
                    n_candidates=2,
                    max_samples=4,
                    gp=self._StubGP(mean, covariance),
                    seed=0,
                )

    def test_valid_stub_posterior_returns_one_finite_candidate(self):
        point = propose_mes(
            self.x,
            self.y,
            self.bounds,
            n_candidates=3,
            max_samples=8,
            gp=self._StubGP(np.zeros(3), np.eye(3)),
            seed=0,
        )
        self.assertEqual(point.shape, (1,))
        self.assertTrue(np.all(np.isfinite(point)))


@unittest.skipUnless(HAS_TORCH, "GP surrogate requires torch")
class MesDriverTest(unittest.TestCase):
    def test_bo_loop_converges(self):
        from mixle.doe import propose_mes

        def f(x):
            return -(np.sin(3 * x) + 0.3 * x**2)  # maximize

        rng = np.random.RandomState(0)
        x = rng.uniform(-3, 3, (6, 1))
        y = f(x[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(8):
                xn = propose_mes(x, y, [(-3.0, 3.0)], n_candidates=200, max_samples=48, maximize=True, seed=i)
                x = np.vstack([x, xn])
                y = np.append(y, f(xn[0]))
        true_max = f(np.linspace(-3, 3, 4000)).max()
        self.assertGreater(y.max(), true_max - 0.1)  # reaches near the global optimum


if __name__ == "__main__":
    unittest.main()
