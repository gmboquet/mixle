"""Array-level conformal prediction (mixle.inference.conformal)."""

import unittest

import numpy as np

from mixle.inference import (
    conformal_label_threshold,
    cv_plus,
    jackknife_plus,
    mondrian_conformal,
    split_conformal,
    weighted_conformal,
)
from mixle.inference.conformal import _conformal_quantile


def _fit_predict(x_tr, y_tr, x_eval):
    X = np.column_stack([np.ones(len(x_tr)), np.atleast_2d(x_tr).reshape(len(x_tr), -1)])
    beta = np.linalg.lstsq(X, y_tr, rcond=None)[0]
    Xe = np.column_stack([np.ones(len(x_eval)), np.atleast_2d(x_eval).reshape(len(x_eval), -1)])
    return Xe @ beta


class SplitConformalTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.uniform(-3, 3, (1500, 1))
        self.y = 2 + 1.5 * self.X.ravel() + rng.normal(0, 1, 1500)
        # a large calibration set keeps the single-split coverage variance small
        self.cal_pred = _fit_predict(self.X[:500], self.y[:500], self.X[500:])
        self.cal_y = self.y[500:]
        self.Xt = rng.uniform(-3, 3, (8000, 1))
        self.yt = 2 + 1.5 * self.Xt.ravel() + rng.normal(0, 1, 8000)
        self.tp = _fit_predict(self.X[:500], self.y[:500], self.Xt)

    def test_two_sided_coverage(self):
        lo, hi = split_conformal(self.cal_pred, self.cal_y, self.tp, alpha=0.1)
        cov = np.mean((self.yt >= lo) & (self.yt <= hi))
        self.assertAlmostEqual(cov, 0.9, delta=0.03)

    def test_one_sided_upper(self):
        lo, hi = split_conformal(self.cal_pred, self.cal_y, self.tp, alpha=0.1, side="upper")
        self.assertTrue(np.all(np.isneginf(lo)))
        self.assertAlmostEqual(np.mean(self.yt <= hi), 0.9, delta=0.04)

    def test_one_sided_lower(self):
        lo, hi = split_conformal(self.cal_pred, self.cal_y, self.tp, alpha=0.1, side="lower")
        self.assertTrue(np.all(np.isposinf(hi)))
        self.assertAlmostEqual(np.mean(self.yt >= lo), 0.9, delta=0.04)


class JackknifeCVTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.X = rng.uniform(-3, 3, (120, 1))
        self.y = 2 + 1.5 * self.X.ravel() + rng.normal(0, 1, 120)
        self.Xt = rng.uniform(-3, 3, (800, 1))
        self.yt = 2 + 1.5 * self.Xt.ravel() + rng.normal(0, 1, 800)

    def test_jackknife_plus_coverage(self):
        lo, hi = jackknife_plus(self.X, self.y, _fit_predict, self.Xt, alpha=0.1)
        cov = np.mean((self.yt >= lo) & (self.yt <= hi))
        self.assertGreaterEqual(cov, 0.88)

    def test_cv_plus_coverage(self):
        lo, hi = cv_plus(self.X, self.y, _fit_predict, self.Xt, alpha=0.1, n_folds=10)
        cov = np.mean((self.yt >= lo) & (self.yt <= hi))
        self.assertGreaterEqual(cov, 0.88)


class MondrianTest(unittest.TestCase):
    def test_group_conditional_coverage(self):
        rng = np.random.RandomState(2)
        n = 800
        X = rng.uniform(-3, 3, (n, 1))
        g = rng.randint(0, 2, n)
        # group 1 has 4x the noise scale -> a marginal interval under-covers it
        y = 2 + 1.5 * X.ravel() + rng.normal(0, 1, n) * (1 + 3 * g)
        cal = slice(500, 800)
        cal_pred = _fit_predict(X[:500], y[:500], X[cal])
        Xt = rng.uniform(-3, 3, (4000, 1))
        gt = rng.randint(0, 2, 4000)
        yt = 2 + 1.5 * Xt.ravel() + rng.normal(0, 1, 4000) * (1 + 3 * gt)
        tp = _fit_predict(X[:500], y[:500], Xt)
        lo, hi = mondrian_conformal(cal_pred, y[cal], g[cal], tp, gt, alpha=0.1)
        cov0 = np.mean((yt[gt == 0] >= lo[gt == 0]) & (yt[gt == 0] <= hi[gt == 0]))
        cov1 = np.mean((yt[gt == 1] >= lo[gt == 1]) & (yt[gt == 1] <= hi[gt == 1]))
        self.assertAlmostEqual(cov0, 0.9, delta=0.04)
        self.assertAlmostEqual(cov1, 0.9, delta=0.04)
        # the high-variance group gets a wider interval
        self.assertGreater((hi - lo)[gt == 1].mean(), 2 * (hi - lo)[gt == 0].mean())


class WeightedTest(unittest.TestCase):
    def test_uniform_weights_match_split(self):
        rng = np.random.RandomState(3)
        cal_pred = rng.normal(0, 1, 300)
        cal_y = cal_pred + rng.normal(0, 1, 300)
        test_pred = rng.normal(0, 1, 50)
        s_lo, s_hi = split_conformal(cal_pred, cal_y, test_pred, alpha=0.1)
        w_lo, w_hi = weighted_conformal(cal_pred, cal_y, test_pred, np.ones(300), alpha=0.1)
        # within one calibration score of each other (quantile-index convention differs by <=1)
        self.assertLess(abs((w_hi - w_lo)[0] - (s_hi - s_lo)[0]), 0.5)

    def test_coverage_under_covariate_shift(self):
        rng = np.random.RandomState(4)
        # train inputs ~ N(0,1); error scale grows with x, so shifting the test toward large x
        # needs reweighting to keep coverage
        x_cal = rng.normal(0, 1, 600)
        cal_pred = x_cal
        cal_y = x_cal + rng.normal(0, 1, 600) * (1 + 0.5 * np.abs(x_cal))
        x_test = rng.normal(1.5, 1, 4000)  # shifted
        test_pred = x_test
        test_y = x_test + rng.normal(0, 1, 4000) * (1 + 0.5 * np.abs(x_test))
        # likelihood ratio N(1.5,1)/N(0,1) at the calibration points
        w = np.exp(-((x_cal - 1.5) ** 2) / 2 + (x_cal**2) / 2)
        lo, hi = weighted_conformal(cal_pred, cal_y, test_pred, w, alpha=0.1)
        cov = np.mean((test_y >= lo) & (test_y <= hi))
        # weighted conformal keeps coverage close to nominal under the shift
        self.assertGreater(cov, 0.85)

    def test_negative_weight_raises(self):
        # weights are likelihood ratios p_test/p_train, inherently non-negative; a negative entry
        # corrupts the weighted CDF (a cumulative sum of weights should be non-decreasing) instead
        # of being caught.
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        with self.assertRaises(ValueError):
            weighted_conformal(cal_pred, cal_y, np.zeros(1), np.array([1.0, 1.0, -5.0]), alpha=0.3)

    def test_negative_test_weight_raises(self):
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        with self.assertRaises(ValueError):
            weighted_conformal(cal_pred, cal_y, np.zeros(1), np.ones(3), alpha=0.3, test_weight=-1.0)

    def test_empty_calibration_raises_value_error_not_indexerror(self):
        # weighted_conformal has its own inline quantile logic rather than routing through
        # _conformal_quantile, so it needs -- and previously lacked -- its own n==0 guard: it
        # crashed with a raw IndexError from cdf[-1] instead of a clear ValueError.
        with self.assertRaises(ValueError):
            weighted_conformal(np.array([]), np.array([]), np.zeros(1), np.array([]), alpha=0.1)

    def test_nan_weight_raises_instead_of_silently_returning_the_trivial_interval(self):
        # `np.any(w < 0.0)` does not catch NaN (a NaN comparison is always False), so a NaN weight
        # used to silently corrupt the cumulative weighted CDF and fall through to the "insufficient
        # mass" branch, indistinguishable from a legitimate empty-mass case.
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        with self.assertRaises(ValueError):
            weighted_conformal(cal_pred, cal_y, np.zeros(1), np.array([1.0, np.nan, 1.0]), alpha=0.3)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            weighted_conformal(np.zeros(3), np.zeros(3), np.zeros(1), np.ones(5), alpha=0.1)

    def test_alpha_below_zero_raises(self):
        # weighted_conformal has its own inline quantile logic (see the guard-duplication comment
        # above) and, unlike _conformal_quantile, previously had no alpha validation at all: an
        # out-of-range alpha silently fell through to a plausible-looking interval instead of
        # raising.
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        with self.assertRaises(ValueError):
            weighted_conformal(cal_pred, cal_y, np.zeros(1), np.ones(3), alpha=-0.1)

    def test_alpha_above_one_raises(self):
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        with self.assertRaises(ValueError):
            weighted_conformal(cal_pred, cal_y, np.zeros(1), np.ones(3), alpha=1.1)

    def test_alpha_above_one_does_not_silently_return_the_alpha_one_answer(self):
        # concretely: alpha=1.1 (and 1.5, 5.0, ...) used to collapse to the exact same finite,
        # plausible-looking interval as the legitimate alpha=1.0 boundary -- a unit/sign bug in a
        # caller (e.g. passing a percentage or a confidence level by mistake) would silently get a
        # confident-looking but statistically meaningless interval instead of an error.
        rng = np.random.RandomState(4)
        x_cal = rng.normal(0, 1, 600)
        cal_pred = x_cal
        cal_y = x_cal + rng.normal(0, 1, 600) * (1 + 0.5 * np.abs(x_cal))
        test_pred = np.array([1.5])
        w = np.exp(-((x_cal - 1.5) ** 2) / 2 + (x_cal**2) / 2)
        for bad_alpha in (1.1, 1.5, 5.0):
            with self.assertRaises(ValueError):
                weighted_conformal(cal_pred, cal_y, test_pred, w, alpha=bad_alpha)

    def test_nan_alpha_raises(self):
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        with self.assertRaises(ValueError):
            weighted_conformal(cal_pred, cal_y, np.zeros(1), np.ones(3), alpha=np.nan)

    def test_alpha_boundaries_zero_and_one_remain_valid(self):
        # weighted_conformal's own alpha convention mirrors _conformal_quantile's: 0.0 and 1.0 are
        # valid boundaries (0% / 100% requested miscoverage), not errors -- only strictly out-of-
        # range or non-finite alpha should raise.
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        w = np.ones(3)
        lo0, hi0 = weighted_conformal(cal_pred, cal_y, np.zeros(1), w, alpha=0.0)
        self.assertTrue(np.isneginf(lo0[0]) and np.isposinf(hi0[0]))
        lo1, hi1 = weighted_conformal(cal_pred, cal_y, np.zeros(1), w, alpha=1.0)
        self.assertTrue(np.isfinite(lo1[0]) and np.isfinite(hi1[0]))

    def test_zero_total_weight_raises_instead_of_dividing_by_zero(self):
        # all calibration weights AND test_weight zero -> total weight is exactly 0, so the CDF
        # computation `cumsum(w_sorted) / total` used to silently divide 0/0 into NaN (with a
        # RuntimeWarning easy to miss) and, because a NaN comparison is always False, fall through
        # to the "insufficient mass" branch and return an unbounded interval without ever raising.
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        with self.assertRaises(ValueError):
            weighted_conformal(cal_pred, cal_y, np.zeros(1), np.zeros(3), alpha=0.1, test_weight=0.0)

    def test_all_zero_calibration_weights_with_positive_test_weight_still_valid(self):
        # distinct from the zero-*total*-weight case above: the default test_weight=1.0 keeps the
        # total positive even when every calibration weight is zero, which is a legitimate (if
        # maximally conservative) "none of my calibration data resembles the test point" answer.
        cal_pred = np.zeros(3)
        cal_y = np.array([1.0, -1.0, 2.0])
        lo, hi = weighted_conformal(cal_pred, cal_y, np.zeros(1), np.zeros(3), alpha=0.1)
        self.assertTrue(np.isneginf(lo[0]) and np.isposinf(hi[0]))


class ConformalQuantileBoundaryTest(unittest.TestCase):
    """Regression: _conformal_quantile's ``s[k - 1]`` used to wrap around via Python negative indexing
    to ``s[-1]`` (the MAXIMUM score) whenever ``k <= 0`` -- exactly at ``alpha == 1.0`` (0% coverage
    requested) -- returning the loosest threshold instead of the tightest, and either an arbitrary
    interior score or an uncaught IndexError for alpha > 1."""

    def test_alpha_one_returns_the_minimum_score_not_the_maximum(self):
        s = np.sort(np.random.RandomState(1).rand(50))
        self.assertAlmostEqual(_conformal_quantile(s, 1.0), float(s.min()))

    def test_quantile_is_monotonically_non_increasing_in_alpha_through_the_boundary(self):
        s = np.random.RandomState(2).rand(50)
        alphas = [0.5, 0.9, 0.99, 0.999999, 1.0]
        qhats = [_conformal_quantile(s, a) for a in alphas]
        for a, b in zip(qhats, qhats[1:]):
            self.assertGreaterEqual(a, b)  # non-increasing as alpha climbs toward 1, no jump back up

    def test_alpha_above_one_raises_instead_of_wrapping_or_crashing(self):
        s = np.random.RandomState(3).rand(20)
        with self.assertRaises(ValueError):
            _conformal_quantile(s, 1.2)

    def test_alpha_below_zero_raises(self):
        s = np.random.RandomState(3).rand(20)
        with self.assertRaises(ValueError):
            _conformal_quantile(s, -0.1)

    def test_empty_scores_raises_instead_of_indexerror(self):
        # k < 1 used to return s[0] unconditionally; with n == 0 (no calibration data at all) that
        # indexed an empty array. alpha=1.0 hits the k < 1 branch (see the module docstring), so it
        # reproduces the exact old crash rather than any other ValueError already raised earlier.
        with self.assertRaises(ValueError):
            _conformal_quantile(np.array([]), 1.0)

    def test_nan_score_raises_instead_of_silently_becoming_the_quantile(self):
        # a NaN calibration score sorts to the array's end; depending on alpha the returned quantile
        # could land exactly on it and silently poison every downstream interval with NaN.
        with self.assertRaises(ValueError):
            _conformal_quantile(np.array([0.1, 0.2, 0.3, np.nan]), 0.3)

    def test_label_threshold_does_not_flip_from_escalate_to_confident_at_alpha_one(self):
        # A near-uniform-looking calibration set with a confident test point: as alpha climbs, the
        # LAC threshold should only get tighter (harder to satisfy), never loosen back up at alpha=1.
        cal_prob_true = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.55, 0.5, 0.4, 0.2])
        qhats = [conformal_label_threshold(cal_prob_true, alpha=a) for a in (0.8, 0.95, 0.999, 1.0)]
        for a, b in zip(qhats, qhats[1:]):
            self.assertLessEqual(b, a)  # threshold only tightens (shrinks) as alpha -> 1, never jumps up


if __name__ == "__main__":
    unittest.main()
