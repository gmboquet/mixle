"""K2: numerics-error receipts through accumulators.

Optional compensated (Kahan) accumulation carries a running numerics-error bound alongside the
accumulated sufficient statistics; the bound composes through ``combine()`` the same way the
statistics themselves do. ``MultivariateGaussianEstimator.estimate()`` grows an opt-in conditioning
receipt (covariance eigenvalue ratio / near-degenerate-variance flag). Both are OFF by default with
no measurable overhead over the pre-existing behavior.

Acceptance criteria under test:
  1. Planted ill-conditioning (and a well-conditioned control) is correctly flagged.
  2. Planted catastrophic cancellation is flagged with correct bound ORDERING, verified against a
     high-precision (``math.fsum``) reference -- not just naive-bound > kahan-bound in isolation.
  3. The disabled/default path is unaffected: behaviorally identical to the pre-existing
     implementation, and a large-workload timing check confirms no measurable regression.
"""

import math
import time
import unittest

import numpy as np

from mixle.stats.compute.error_receipts import CompensatedAccumulator, conditioning_receipt, error_bound
from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianEstimator
from mixle.stats.univariate.continuous.gaussian import GaussianAccumulator, GaussianEstimator

# Stated overhead threshold for the disabled/default path (acceptance criterion: "overhead < stated %
# when disabled"). This repo's test suite runs under pytest-xdist -n auto by default (see
# pyproject.toml), so wall-clock timing here is contended by sibling workers and cannot support a
# tight percentage threshold without flaking. The primary overhead guarantee is therefore the
# STRUCTURAL check (test_disabled_path_matches_naive_float64_reference /
# test_disabled_accumulator_carries_no_receipt_state below): the disabled path is bit-for-bit
# identical to the pre-existing np.dot-based implementation, so it cannot regress. The timing check
# is corroborating evidence with a generous, non-flaky threshold (min-of-many-repeats, large
# workload) -- generous enough to absorb worker contention while still catching a real regression
# (e.g. an accidental O(n) Python loop on the disabled path, which would blow the ratio far past it).
_DISABLED_OVERHEAD_THRESHOLD = 0.50  # 50%, deliberately generous under parallel test contention


# --------------------------------------------------------------------------------------------------
# 1. Planted ill-conditioning
# --------------------------------------------------------------------------------------------------
class ConditioningReceiptTest(unittest.TestCase):
    def _fit(self, data, **kwargs):
        est = MultivariateGaussianEstimator(dim=data.shape[1], track_conditioning=True, **kwargs)
        acc = est.accumulator_factory().make()
        acc.seq_update(data, np.ones(len(data)), None)
        return est.estimate(None, acc.value())

    def test_planted_near_degenerate_direction_is_flagged(self):
        # 3 well-spread dims + 1 dim with variance ~1e8x smaller -> a genuinely near-degenerate
        # covariance direction, not a numerical artifact.
        rng = np.random.RandomState(0)
        n = 5000
        well = rng.normal(loc=0.0, scale=1.0, size=(n, 3))
        degenerate = rng.normal(loc=0.0, scale=1.0e-4, size=(n, 1))  # variance ~1e-8 vs ~1
        data = np.hstack([well, degenerate])

        dist = self._fit(data)
        receipt = dist.conditioning_receipt
        self.assertIsNotNone(receipt)
        self.assertTrue(receipt.near_degenerate)
        self.assertGreater(receipt.condition_number, 1.0e4)
        # the smallest eigenvalue should correspond to the planted near-zero-variance dimension
        self.assertLess(float(np.min(receipt.eigenvalues)), 1.0e-4)

    def test_well_conditioned_control_is_not_flagged(self):
        rng = np.random.RandomState(1)
        n = 5000
        # isotropic-ish covariance: comparable variances on every axis -> healthy condition number
        data = rng.multivariate_normal(mean=np.zeros(4), cov=np.diag([1.0, 1.2, 0.8, 1.1]), size=n)

        dist = self._fit(data)
        receipt = dist.conditioning_receipt
        self.assertIsNotNone(receipt)
        self.assertFalse(receipt.near_degenerate)
        self.assertLess(receipt.condition_number, 100.0)

    def test_disabled_by_default_no_receipt_attached(self):
        rng = np.random.RandomState(2)
        data = rng.normal(size=(200, 3))
        est = MultivariateGaussianEstimator(dim=3)  # track_conditioning defaults to False
        acc = est.accumulator_factory().make()
        acc.seq_update(data, np.ones(len(data)), None)
        dist = est.estimate(None, acc.value())
        self.assertFalse(hasattr(dist, "conditioning_receipt"))

    def test_conditioning_receipt_function_matches_known_eigenvalues(self):
        # a diagonal covariance's eigenvalues are exactly its diagonal -- a direct correctness check
        # of conditioning_receipt() independent of the estimator plumbing.
        covar = np.diag([4.0, 1.0, 1.0e-9])
        receipt = conditioning_receipt(covar, degenerate_ratio=1.0e-6)
        self.assertTrue(np.allclose(sorted(receipt.eigenvalues), [1.0e-9, 1.0, 4.0]))
        self.assertAlmostEqual(receipt.condition_number, 4.0 / 1.0e-9, delta=4.0 / 1.0e-9 * 1e-6)
        self.assertTrue(receipt.near_degenerate)


# --------------------------------------------------------------------------------------------------
# 2. Planted catastrophic cancellation: bound ordering + validity against a high-precision reference
# --------------------------------------------------------------------------------------------------
class CatastrophicCancellationTest(unittest.TestCase):
    @staticmethod
    def _planted_values(n=200000, big=1.0e8, seed=0):
        # A large offset followed by many small increments that mostly cancel: the classic
        # catastrophic-cancellation setup for naive left-to-right float64 summation. Values:
        # [+big, -big, +tiny_1, -tiny_1, +tiny_2, -tiny_2, ...] with a tiny residual net sum so the
        # true answer is well-defined and far smaller than the naive rounding noise.
        rng = np.random.RandomState(seed)
        small = rng.uniform(1.0, 10.0, size=n // 2)
        values = [big, -big]
        for s in small:
            values.append(s)
            values.append(-s + 1e-3)  # tiny net residual per pair, not exact cancellation
        return values

    def test_naive_vs_kahan_bound_ordering_and_validity(self):
        values = self._planted_values()

        naive_acc = CompensatedAccumulator(compensated=False)
        kahan_acc = CompensatedAccumulator(compensated=True)
        for v in values:
            naive_acc.add(v)
            kahan_acc.add(v)

        reference = math.fsum(values)  # near-exact (Shewchuk's algorithm): ground truth
        naive_error = abs(naive_acc.total - reference)
        kahan_error = abs(kahan_acc.total - reference)
        naive_bound = naive_acc.bound()
        kahan_bound = kahan_acc.bound()

        # Sanity: the planted scenario actually exercises meaningfully different float64 results
        # under naive vs compensated summation (otherwise the ordering claim would be vacuous).
        self.assertNotEqual(naive_acc.total, kahan_acc.total)

        # (a) the reported bound is a genuinely valid, non-violated upper bound on the real error,
        #     for BOTH accumulation modes, verified against the high-precision reference.
        self.assertLessEqual(naive_error, naive_bound, "naive bound must not be violated")
        self.assertLessEqual(kahan_error, kahan_bound, "kahan bound must not be violated")

        # (b) bound ORDERING: the naive path's reported bound is larger than the compensated path's.
        self.assertGreater(naive_bound, kahan_bound)

        # (c) the ordering matches reality, not just the formula: Kahan's actual error is also
        #     smaller (or equal) than naive's actual error on this cancellation-heavy workload.
        self.assertLessEqual(kahan_error, naive_error)

        self._naive_bound, self._kahan_bound = naive_bound, kahan_bound
        self._naive_error, self._kahan_error = naive_error, kahan_error

    def test_error_bound_function_matches_accumulator_bound(self):
        values = self._planted_values(n=2000, seed=3)
        acc = CompensatedAccumulator(compensated=True)
        for v in values:
            acc.add(v)
        self.assertAlmostEqual(acc.bound(), error_bound(acc.n, acc.abs_total, compensated=True))

    def test_gaussian_accumulator_compensated_bound_ordering(self):
        # The same catastrophic-cancellation scenario through the real GaussianAccumulator wiring
        # (not just the standalone CompensatedAccumulator primitive), on the 'sum' sufficient
        # statistic (large mu offset + many small-magnitude observations).
        rng = np.random.RandomState(4)
        n = 100000
        offset = 1.0e7
        x = np.concatenate([[offset], rng.uniform(-1.0, 1.0, size=n - 1)])
        weights = np.ones(n)

        naive = GaussianAccumulator(compensated=False)
        kahan = GaussianAccumulator(compensated=True)
        naive.seq_update(x, weights, None)
        kahan.seq_update(x, weights, None)

        reference = math.fsum(x.tolist())
        naive_error = abs(naive.sum - reference)
        kahan_error = abs(kahan.sum - reference)
        naive_bound = error_bound(n, float(np.sum(np.abs(x))), compensated=False)
        kahan_bound = kahan.error_bound()["sum"]

        self.assertLessEqual(naive_error, naive_bound)
        self.assertLessEqual(kahan_error, kahan_bound)
        self.assertGreater(naive_bound, kahan_bound)

    def test_bounds_compose_through_combine(self):
        # split the planted workload into two partitions, accumulate each separately (as two
        # workers would), then combine() -- the resulting bound must equal what a single
        # accumulator over the whole stream would report (bounds compose exactly through the
        # additive (abs_total, n) receipt, like the roadmap requires).
        values = self._planted_values(n=20000, seed=5)
        half = len(values) // 2

        whole = CompensatedAccumulator(compensated=True)
        for v in values:
            whole.add(v)

        part_a = CompensatedAccumulator(compensated=True)
        for v in values[:half]:
            part_a.add(v)
        part_b = CompensatedAccumulator(compensated=True)
        for v in values[half:]:
            part_b.add(v)
        part_a.combine(part_b)

        self.assertEqual(whole.n, part_a.n)
        self.assertAlmostEqual(whole.abs_total, part_a.abs_total, delta=1e-6 * whole.abs_total)
        self.assertAlmostEqual(whole.bound(), part_a.bound(), delta=1e-9 * whole.bound())

    def test_gaussian_accumulator_combine_preserves_bound_composition(self):
        # Same composition property, but through the real GaussianAccumulator.combine() /
        # .value() plumbing (GaussianSuffStat receipt round-trip), matching how partitions are
        # actually merged in this codebase (acc.combine(other.value())).
        rng = np.random.RandomState(6)
        x = np.concatenate([[5.0e6], rng.uniform(-2.0, 2.0, size=4999)])

        whole = GaussianAccumulator(compensated=True)
        whole.seq_update(x, np.ones(len(x)), None)

        a = GaussianAccumulator(compensated=True)
        b = GaussianAccumulator(compensated=True)
        a.seq_update(x[:2500], np.ones(2500), None)
        b.seq_update(x[2500:], np.ones(len(x) - 2500), None)
        a.combine(b.value())

        self.assertAlmostEqual(whole.sum, a.sum, places=6)
        self.assertAlmostEqual(whole.error_bound()["sum"], a.error_bound()["sum"], delta=1e-9)


# --------------------------------------------------------------------------------------------------
# 3. Disabled path: identical behavior + no measurable overhead
# --------------------------------------------------------------------------------------------------
class DisabledPathOverheadTest(unittest.TestCase):
    def test_disabled_accumulator_carries_no_receipt_state(self):
        acc = GaussianAccumulator()  # compensated defaults to False
        self.assertIsNone(acc._sum_acc)
        self.assertIsNone(acc._sum2_acc)
        self.assertIsNone(acc.error_bound())

    def test_disabled_path_matches_naive_float64_reference(self):
        # behavioral identity: the disabled path's sum/sum2 must equal plain np.dot accumulation
        # exactly (bit-for-bit) -- i.e. it takes the SAME code path as before this change.
        rng = np.random.RandomState(7)
        x = rng.normal(size=5000)
        w = np.ones(5000)
        acc = GaussianAccumulator(compensated=False)
        acc.seq_update(x, w, None)
        self.assertEqual(acc.sum, float(np.dot(x, w)))
        self.assertEqual(acc.sum2, float(np.dot(x * x, w)))

    def test_disabled_seq_update_overhead_under_threshold(self):
        # Large workload + repeated trials so the timing comparison is not flaky: the disabled path
        # (vectorized np.dot, untouched) must not run measurably slower than a plain baseline
        # reimplementation of the pre-existing formula. Stated threshold: <50% overhead (see the
        # module-level comment on _DISABLED_OVERHEAD_THRESHOLD for why this is generous rather than
        # tight -- the tight guarantee is the structural/behavioral-identity test below).
        rng = np.random.RandomState(8)
        n = 2_000_000
        x = rng.normal(size=n)
        w = np.ones(n)
        repeats = 15

        def baseline():
            s = 0.0
            s2 = 0.0
            s += np.dot(x, w)
            s2 += np.dot(x * x, w)
            return s, s2

        def disabled_path():
            acc = GaussianAccumulator(compensated=False)
            acc.seq_update(x, w, None)
            return acc.sum, acc.sum2

        # warm-up (avoid first-call cache effects dominating the measurement)
        baseline()
        disabled_path()

        # Take the MIN over independently-timed repeats (standard robust microbenchmark
        # methodology -- the min is the least noise-contaminated estimate, since noise from other
        # processes/GC/scheduling can only ever slow a single trial down, never speed it up) rather
        # than a single summed loop, so this is not flaky under parallel test execution (pytest-xdist).
        baseline_times = []
        disabled_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            baseline()
            baseline_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            disabled_path()
            disabled_times.append(time.perf_counter() - t0)
        t_baseline = min(baseline_times)
        t_disabled = min(disabled_times)

        overhead = (t_disabled - t_baseline) / t_baseline if t_baseline > 0 else 0.0
        self.assertLess(
            overhead,
            _DISABLED_OVERHEAD_THRESHOLD,
            "disabled GaussianAccumulator.seq_update overhead %.1f%% exceeds the %.0f%% threshold"
            % (overhead * 100.0, _DISABLED_OVERHEAD_THRESHOLD * 100.0),
        )

    def test_compensated_path_is_much_slower_than_disabled(self):
        # Corroborating evidence the disabled path is genuinely the fast/untouched one: the
        # compensated path (Python-level per-element Kahan loop) should be dramatically slower on
        # a large workload, confirming the two paths are not accidentally sharing the slow branch.
        rng = np.random.RandomState(9)
        n = 200000
        x = rng.normal(size=n)
        w = np.ones(n)

        t0 = time.perf_counter()
        acc = GaussianAccumulator(compensated=False)
        acc.seq_update(x, w, None)
        t_disabled = time.perf_counter() - t0

        t0 = time.perf_counter()
        acc2 = GaussianAccumulator(compensated=True)
        acc2.seq_update(x, w, None)
        t_compensated = time.perf_counter() - t0

        self.assertGreater(t_compensated, t_disabled * 2.0)

    def test_estimator_compensated_flag_defaults_false(self):
        est = GaussianEstimator()
        self.assertFalse(est.compensated)
        acc = est.accumulator_factory().make()
        self.assertFalse(acc.compensated)


if __name__ == "__main__":
    unittest.main()
