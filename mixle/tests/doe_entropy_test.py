"""Max-value Entropy Search information-theoretic BO (mixle.doe.entropy)."""

import importlib.util
import unittest
import warnings

import numpy as np

from mixle.doe.entropy import _fit_gumbel, _gumbel_quantile, max_value_entropy_search, sample_max_values

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
