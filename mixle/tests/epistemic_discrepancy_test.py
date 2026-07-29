"""mixle.epistemic.discrepancy: KL/JS/Wasserstein/MMD between distributions or samples (Card E1)."""

import unittest

import numpy as np

from mixle.epistemic.discrepancy import (
    _sample,
    discrepancy_report,
    js_divergence,
    kl_divergence,
    mmd,
    mmd_squared,
    wasserstein_distance,
)
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class SampleDispatchTest(unittest.TestCase):
    """_sample's dist.sample(n) / per-draw dist.sample() fallback dispatch."""

    def test_a_bug_inside_sample_is_not_masked_by_a_retry_looping_n_single_draws(self):
        # a sample(n) that accepts n must be called with it exactly once; a TypeError from inside
        # its own body must propagate, not be swallowed and silently retried as n separate sample()
        # calls -- which would be both wrong (n calls instead of 1) and far more expensive. n=None
        # has a default specifically so a naive try/except TypeError fallback's sample() retries
        # are themselves syntactically valid and reach the body (appending to calls) -- otherwise
        # those retries would fail at argument-binding before ever calling in, and this test could
        # not tell a single correct call apart from a silent duplicate retry loop.
        calls = []

        class BuggyDist:
            def sample(self, n=None):
                calls.append(n)
                return None + (n or 0)  # an internal bug unrelated to whether n is accepted

        with self.assertRaises(TypeError):
            _sample(BuggyDist(), 5, np.random.RandomState(0))
        self.assertEqual(calls, [5])  # called once, with n -- never retried as 5 separate calls

    def test_legacy_single_draw_sample_falls_back_correctly(self):
        class LegacySingleDrawDist:
            def __init__(self):
                self.i = 0

            def sample(self):
                self.i += 1
                return float(self.i)

        out = _sample(LegacySingleDrawDist(), 4, np.random.RandomState(0))
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0])


class KLDivergenceTest(unittest.TestCase):
    def test_kl_of_identical_gaussians_is_zero(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(0.0, 1.0)
        self.assertAlmostEqual(kl_divergence(p, q), 0.0, places=8)

    def test_kl_matches_closed_form_gaussian_formula(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(5.0, 1.0)
        mu_p, var_p, mu_q, var_q = 0.0, 1.0, 5.0, 1.0
        expected = 0.5 * (var_p / var_q + (mu_q - mu_p) ** 2 / var_q - 1.0 + np.log(var_q / var_p))
        self.assertAlmostEqual(kl_divergence(p, q), expected, places=8)


class JSDivergenceTest(unittest.TestCase):
    def test_symmetric_within_mc_tolerance(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(3.0, 1.0)
        a = js_divergence(p, q, n=20_000, seed=0)
        b = js_divergence(q, p, seed=0, n=20_000)
        self.assertAlmostEqual(a, b, delta=0.02)


class WassersteinDistanceTest(unittest.TestCase):
    def test_matches_known_toy_case(self):
        class PointMass:
            def __init__(self, value):
                self.value = value

            def sample(self, n):
                return np.full(n, self.value, dtype=np.float64)

        p = PointMass(0.0)
        q = PointMass(3.0)
        self.assertAlmostEqual(wasserstein_distance(p, q, n=100), 3.0, places=8)

    def test_multivariate_raises_not_implemented(self):
        class TwoD:
            def sample(self, n):
                return np.zeros((n, 2), dtype=np.float64)

        with self.assertRaises(NotImplementedError):
            wasserstein_distance(TwoD(), TwoD(), n=10)

    def test_column_vector_samples_are_sorted_before_matching(self):
        # np.sort defaults to the last axis; on a (n, 1) column vector -- what a distribution's
        # .sample(n) returns when each draw is its own length-1 row -- that axis has one element,
        # so sorting it in place is a silent no-op and the draws stay in their original order.
        # p=[100, 0] vs q=[49, 51]: the correct order statistics are p=[0, 100], q=[49, 51], giving
        # mean(|0-49|, |100-51|) = 49.0. Pairing the *unsorted* draws instead gives the wrong 51.0.
        class ColumnVectorDist:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float64).reshape(-1, 1)

            def sample(self, n):
                return self._values

        p = ColumnVectorDist([100.0, 0.0])
        q = ColumnVectorDist([49.0, 51.0])
        self.assertAlmostEqual(wasserstein_distance(p, q, n=2), 49.0, places=8)

    def test_column_vector_and_flat_samples_agree(self):
        # Negative control: a (n, 1) column vector and its (n,) flattened equivalent must give the
        # identical answer -- the extra trailing length-1 axis alone must not change the result now
        # that both shapes are squeezed to 1D before sorting.
        class FlatDist:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float64)

            def sample(self, n):
                return self._values

        class ColumnVectorDist:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float64).reshape(-1, 1)

            def sample(self, n):
                return self._values

        flat = wasserstein_distance(FlatDist([100.0, 0.0, 37.5]), FlatDist([49.0, 51.0, 2.5]), n=3)
        column = wasserstein_distance(ColumnVectorDist([100.0, 0.0, 37.5]), ColumnVectorDist([49.0, 51.0, 2.5]), n=3)
        self.assertAlmostEqual(flat, column, places=8)


class MMDTest(unittest.TestCase):
    def test_same_distribution_is_near_zero(self):
        rng = np.random.RandomState(0)
        xs = rng.normal(size=1000)
        value = mmd(xs[:500], xs[500:])
        self.assertLess(abs(value), 0.05)

    def test_different_distributions_is_clearly_positive(self):
        rng = np.random.RandomState(0)
        xs = rng.normal(size=1000)
        ys = rng.normal(loc=5.0, size=1000)
        self.assertGreater(mmd(xs, ys), 0.5)

    def test_unknown_kernel_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            mmd(np.zeros(5), np.zeros(5), kernel="polynomial")

    def test_empty_samples_raises_instead_of_nan(self):
        with self.assertRaises(ValueError):
            mmd(np.array([]), np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            mmd(np.array([1.0, 2.0, 3.0]), np.array([]))

    def test_nonpositive_bandwidth_raises_instead_of_silently_using_gamma_one(self):
        for bad_bandwidth in (0.0, -1.0):
            with self.subTest(bandwidth=repr(bad_bandwidth)), self.assertRaises(ValueError):
                mmd(np.zeros(5), np.ones(5), bandwidth=bad_bandwidth)


class DiscrepancyReportTest(unittest.TestCase):
    def test_not_degraded_for_registered_closed_form_pair(self):
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(1.0, 1.0)
        result = discrepancy_report(p, q)
        self.assertEqual(result.metric, "kl_divergence")
        self.assertFalse(result.degraded)

    def test_degraded_for_a_pair_without_a_closed_form(self):
        class SampleOnly:
            def __init__(self, loc):
                self.loc = loc

            def log_density(self, x):
                return float(-0.5 * (x - self.loc) ** 2)

            def sample(self, n):
                return np.random.RandomState(0).normal(loc=self.loc, size=n)

        result = discrepancy_report(SampleOnly(0.0), SampleOnly(1.0))
        self.assertEqual(result.metric, "kl_divergence")
        self.assertTrue(result.degraded)

    def test_explicit_metric_is_honored(self):
        result = discrepancy_report(np.zeros(50), np.ones(50), metric="mmd")
        self.assertEqual(result.metric, "mmd")
        self.assertTrue(result.degraded)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            discrepancy_report(np.zeros(5), np.zeros(5), metric="not_a_real_metric")

    def test_seeded_auto_dispatch_is_reproducible(self):
        # The exact branch that used to construct an unseeded internal RNG: predicted is a
        # distribution, observed is a plain array, so 'auto' routes to mmd(samples_from(predicted),
        # observed). Same seed in -> same value out, and the seed actually used is on the result.
        predicted = GaussianDistribution(0.0, 1.0)
        observed = np.random.RandomState(7).normal(loc=0.3, size=20)
        first = discrepancy_report(predicted, observed, metric="auto", seed=123)
        second = discrepancy_report(predicted, observed, metric="auto", seed=123)
        self.assertEqual(first.value, second.value)
        self.assertEqual(first.seed, 123)
        self.assertEqual(second.seed, 123)
        self.assertEqual(first.n_samples, 512)

    def test_unseeded_call_still_records_a_reproducible_seed(self):
        # No seed supplied: discrepancy_report must still generate one, use it, and record it, so a
        # caller can reproduce a specific unseeded call after the fact via result.seed even though
        # they never picked the seed themselves.
        predicted = GaussianDistribution(0.0, 1.0)
        observed = np.random.RandomState(7).normal(loc=0.3, size=20)
        result = discrepancy_report(predicted, observed, metric="auto")
        self.assertIsInstance(result.seed, int)
        reproduced = discrepancy_report(predicted, observed, metric="auto", seed=result.seed)
        self.assertEqual(result.value, reproduced.value)

    def test_different_seeds_typically_give_different_values(self):
        # Negative control: seed must actually drive the sampling, not just be cosmetically recorded
        # while the same draw happens underneath regardless of what's passed in.
        predicted = GaussianDistribution(0.0, 1.0)
        observed = np.random.RandomState(7).normal(loc=0.3, size=20)
        a = discrepancy_report(predicted, observed, metric="auto", seed=1)
        b = discrepancy_report(predicted, observed, metric="auto", seed=2)
        self.assertNotEqual(a.value, b.value)

    def test_exact_closed_form_records_no_seed_or_sample_count(self):
        # Negative control: the exact Gaussian-pair closed form never samples, so it must not claim
        # a seed or sample count -- that would misleadingly imply randomness that was never used.
        p = GaussianDistribution(0.0, 1.0)
        q = GaussianDistribution(1.0, 1.0)
        result = discrepancy_report(p, q)
        self.assertIsNone(result.seed)
        self.assertIsNone(result.n_samples)

    def test_seeded_reproducible_for_sample_based_kl_pair(self):
        # The *other* internal-sampling branch: both sides are distributions but not the exact
        # Gaussian pair, so 'auto' falls back to kl_divergence's Monte Carlo estimate. Uses mixle's
        # own .sampler(seed).sample(n) shape so the stub's draws genuinely depend on the RNG
        # discrepancy_report threads through, rather than a fixed internal seed that would make this
        # test pass trivially regardless of whether seed-threading actually works.
        class _NormalSampler:
            def __init__(self, loc, seed):
                self.rng = np.random.RandomState(seed)
                self.loc = loc

            def sample(self, n):
                return self.rng.normal(loc=self.loc, size=n)

        class SamplerBasedDist:
            def __init__(self, loc):
                self.loc = loc

            def log_density(self, x):
                return float(-0.5 * (x - self.loc) ** 2)

            def sampler(self, seed=None):
                return _NormalSampler(self.loc, seed)

        p, q = SamplerBasedDist(0.0), SamplerBasedDist(1.0)
        first = discrepancy_report(p, q, seed=42)
        second = discrepancy_report(p, q, seed=42)
        third = discrepancy_report(p, q, seed=43)
        self.assertEqual(first.value, second.value)
        self.assertEqual(first.seed, 42)
        self.assertNotEqual(first.value, third.value)


class MMDNonNegativityTest(unittest.TestCase):
    """MXR-080-1748: the public discrepancy must not be a signed squared estimator."""

    def test_identical_samples_never_give_a_negative_discrepancy(self):
        for values in ([0.0, 1.0], [3.0], [1.0, 2.0, 3.0, 4.0]):
            arr = np.array(values)
            with self.subTest(values=repr(values)):
                self.assertGreaterEqual(mmd(arr, arr), 0.0)

    def test_the_squared_estimator_is_still_available_and_still_signed(self):
        arr = np.array([0.0, 1.0])
        self.assertLess(mmd_squared(arr, arr), 0.0)  # unbiased under the null: it scatters both ways
        self.assertAlmostEqual(mmd(arr, arr), 0.0, places=12)

    def test_mmd_is_the_clipped_root_of_the_squared_estimator(self):
        rng = np.random.RandomState(0)
        xs, ys = rng.normal(size=200), rng.normal(loc=3.0, size=200)
        self.assertAlmostEqual(mmd(xs, ys), float(np.sqrt(mmd_squared(xs, ys))), places=12)

    def test_report_never_publishes_a_negative_discrepancy(self):
        arr = np.array([0.0, 1.0])
        self.assertGreaterEqual(discrepancy_report(arr, arr, metric="mmd").value, 0.0)


class SampleBudgetEvidenceTest(unittest.TestCase):
    """MXR-080-1750: recorded sample budgets must describe work that actually happened."""

    class _ShortSampler:
        def __init__(self, loc=0.0):
            self.loc = loc

        def log_density(self, x):
            return float(-0.5 * (x - self.loc) ** 2)

        def sample(self, n, rng=None):
            del n, rng
            return np.array([self.loc, self.loc + 1.0])  # always two draws, whatever was asked

    def test_a_sampler_returning_fewer_draws_than_requested_is_rejected(self):
        with self.assertRaises(ValueError):
            _sample(self._ShortSampler(), 10_000, np.random.RandomState(0))

    def test_a_short_sampler_cannot_back_a_10000_sample_report(self):
        with self.assertRaises(ValueError):
            discrepancy_report(self._ShortSampler(0.0), self._ShortSampler(1.0), metric="kl_divergence", seed=0)

    def test_wasserstein_refuses_unequal_empirical_measures(self):
        class _OneDraw(self._ShortSampler):
            def sample(self, n, rng=None):
                del n, rng
                return np.array([1.0])

        with self.assertRaises(ValueError):
            wasserstein_distance(self._ShortSampler(), _OneDraw(), n=2)


class SamplerReproducibilityTest(unittest.TestCase):
    """MXR-080-1749: a recorded seed must actually reproduce the value it is recorded on."""

    class _GlobalRngDist:
        """A direct sampler that ignores every RNG and reads NumPy's global state."""

        def __init__(self, loc):
            self.loc = loc

        def log_density(self, x):
            return float(-0.5 * (x - self.loc) ** 2)

        def sample(self, n):
            return np.random.normal(loc=self.loc, size=n)

    class _ControlledDist:
        def __init__(self, loc):
            self.loc = loc

        def log_density(self, x):
            return float(-0.5 * (x - self.loc) ** 2)

        def sample(self, n, rng=None):
            return (rng or np.random).normal(loc=self.loc, size=n)

    def test_an_uncontrolled_sampler_is_not_reported_as_reproducible(self):
        p, q = self._GlobalRngDist(0.0), self._GlobalRngDist(1.0)
        first = discrepancy_report(p, q, metric="kl_divergence", seed=123)
        second = discrepancy_report(p, q, metric="kl_divergence", seed=123)
        self.assertFalse(first.reproducible)
        self.assertIsNone(first.seed)  # no integer is recorded that fails to reproduce the value
        self.assertNotEqual(first.value, second.value)

    def test_an_rng_aware_sampler_is_seeded_and_reproduces_exactly(self):
        p, q = self._ControlledDist(0.0), self._ControlledDist(1.0)
        first = discrepancy_report(p, q, metric="kl_divergence", seed=123)
        second = discrepancy_report(p, q, metric="kl_divergence", seed=123)
        third = discrepancy_report(p, q, metric="kl_divergence", seed=124)
        self.assertTrue(first.reproducible)
        self.assertEqual(first.seed, 123)
        self.assertEqual(first.value, second.value)
        self.assertNotEqual(first.value, third.value)

    def test_the_mixle_sampler_shape_stays_reproducible(self):
        p = GaussianDistribution(0.0, 1.0)
        observed = np.random.RandomState(7).normal(size=20)
        result = discrepancy_report(p, observed, metric="auto", seed=5)
        self.assertTrue(result.reproducible)
        self.assertEqual(result.seed, 5)


class MMDGeometryAndBandwidthTest(unittest.TestCase):
    """MXR-080-1751: the public MMD path accepted geometry and bandwidths it cannot honour.

    ``_prepare`` reshaped only rank-1 input, so a 0-d sample reached ``shape[0]`` and raised a bare
    ``IndexError`` (an incidental crash, not a contract), while a rank-3 array was accepted and then
    failed inside the kernel -- which reduces only the last coordinate axis. Bandwidth was checked
    with ``bandwidth <= 0``, which NaN fails like every other comparison: a NaN bandwidth returned
    NaN as if it were a discrepancy, and an infinite bandwidth made ``gamma`` exactly 0.0 so every
    kernel entry became ``exp(0) == 1`` and the estimator reported an apparently exact zero -- "these
    two sample sets are identical" -- for any two sample sets at all.
    """

    def setUp(self):
        self.x = np.random.RandomState(0).normal(size=(20, 2))
        self.y = np.random.RandomState(1).normal(size=(20, 2))

    def test_unsupported_sample_rank_is_rejected_by_contract(self):
        for fn in (mmd, mmd_squared):
            with self.assertRaisesRegex(ValueError, r"\(n,\) or \(n, d\)"):
                fn(1.0, 2.0)
            with self.assertRaisesRegex(ValueError, r"\(n,\) or \(n, d\)"):
                fn(self.x.reshape(5, 4, 2), self.y.reshape(5, 4, 2))

    def test_nonfinite_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            mmd(np.full((20, 2), np.nan), self.y)
        with self.assertRaisesRegex(ValueError, "finite"):
            mmd(self.x, np.full((20, 2), np.inf))

    def test_mismatched_coordinate_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coordinate dimension"):
            mmd(self.x, self.y[:, :1])

    def test_nonfinite_bandwidth_is_rejected_instead_of_faking_a_verdict(self):
        for bandwidth in (np.nan, np.inf, -np.inf, 0.0, -1.0, True):
            with self.assertRaisesRegex(ValueError, "bandwidth"):
                mmd(self.x, self.y, bandwidth=bandwidth)

    def test_supported_geometry_is_unchanged(self):
        # negative control: (n,) and (n, d) both still work, and identical inputs still give 0.
        self.assertGreater(mmd(self.x, self.y), 0.0)
        self.assertEqual(mmd(self.x, self.x), 0.0)
        self.assertGreater(mmd(np.arange(20.0), np.arange(20.0) + 5.0), 0.0)
        self.assertGreater(mmd(self.x, self.y, bandwidth=1.5), 0.0)


if __name__ == "__main__":
    unittest.main()
