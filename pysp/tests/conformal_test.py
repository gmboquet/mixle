"""Tests for pysp.ppl split-conformal prediction."""

import unittest

import numpy as np

from pysp.ppl import ConformalClassifier, ConformalRegressor, Field, Normal, conformal, free
from pysp.ppl.conformal import conformal_quantile


class ConformalRegressorTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n = 6000
        x = rng.uniform(-3, 3, n)
        y = np.sin(x) + 0.3 * x + rng.normal(0, 0.4, n)
        tr, cal, te = np.split(rng.permutation(n), [3000, 4500])
        self.m = Normal(free * Field("x") + free, free).fit(list(y[tr]), given={"x": list(x[tr])})
        self.cal = ({"x": list(x[cal])}, y[cal])
        self.te = ({"x": list(x[te])}, y[te])

    def test_marginal_coverage_holds(self):
        cp = conformal(self.m.result, self.cal[0], self.cal[1], alpha=0.1)
        cov = cp.covers(self.te[0], self.te[1]).mean()
        self.assertGreater(cov, 0.86)  # finite-sample valid: at least ~0.90 up to sampling noise
        self.assertLess(cov, 0.94)
        self.assertGreater(cp.qhat, 0.0)

    def test_interval_is_symmetric_about_prediction(self):
        cp = ConformalRegressor(self.m.result, self.cal[0], self.cal[1], alpha=0.2)
        lo, hi = cp.interval(self.te[0])
        center = np.asarray(self.m.result.predict(self.te[0]))
        np.testing.assert_allclose((lo + hi) / 2.0, center, atol=1e-9)
        np.testing.assert_allclose(hi - lo, 2.0 * cp.qhat, atol=1e-9)

    def test_misspecified_model_keeps_coverage(self):
        # a constant predictor still attains coverage; only the interval widens
        class _ConstMean:
            def __init__(self, c):
                self.c = float(c)

            def predict(self, given):
                return np.full(len(next(iter(given.values()))), self.c)

        const = _ConstMean(np.mean(self.cal[1]))
        cp = conformal(const, self.cal[0], self.cal[1], alpha=0.1)
        self.assertGreater(cp.covers(self.te[0], self.te[1]).mean(), 0.86)


class ConformalQuantileRegressorTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        n = 6000
        self.x = rng.uniform(0, 5, n)
        self.y = 2.0 + 1.5 * self.x + rng.normal(0, 0.3 + 0.7 * self.x, n)  # heteroskedastic
        self.tr, self.cal, self.te = np.split(rng.permutation(n), [3000, 4500])

    def _qfit(self, tau):
        return Normal(free * Field("x") + free, free).fit(
            list(self.y[self.tr]), given={"x": list(self.x[self.tr])}, quantile=tau
        )

    def test_marginal_coverage_and_adaptive_width(self):
        from pysp.ppl import ConformalQuantileRegressor

        lo, hi = self._qfit(0.05), self._qfit(0.95)
        cqr = ConformalQuantileRegressor(lo.result, hi.result, {"x": list(self.x[self.cal])}, self.y[self.cal], alpha=0.1)
        cov = cqr.covers({"x": list(self.x[self.te])}, self.y[self.te]).mean()
        self.assertGreater(cov, 0.86)
        self.assertLess(cov, 0.95)
        a, b = cqr.interval({"x": list(self.x[self.te])})
        width = b - a
        xt = np.asarray(self.x[self.te])
        self.assertGreater(width[xt > 4].mean(), width[xt < 1].mean())  # band widens with the noise


class ConformalQuantileTestCase(unittest.TestCase):
    def test_finite_sample_correction(self):
        scores = np.arange(1.0, 101.0)  # 1..100
        # ceil((100+1)*0.9) = 91 -> the 91st smallest score
        self.assertEqual(conformal_quantile(scores, 0.1), 91.0)

    def test_alpha_too_small_is_infinite(self):
        # (n+1)(1-alpha) > n  =>  no finite threshold can guarantee coverage
        self.assertEqual(conformal_quantile(np.arange(10.0), 0.01), float("inf"))


class ConformalClassifierTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n, K = 3000, 4
        y = rng.randint(0, K, n)
        # a noisy-but-informative probability matrix: extra mass on the true class
        logits = rng.normal(0, 1, (n, K))
        logits[np.arange(n), y] += 1.5
        P = np.exp(logits)
        self.P = P / P.sum(1, keepdims=True)
        self.y = y
        self.cal = slice(0, 1500)
        self.te = slice(1500, n)

    def test_set_coverage_and_sizes(self):
        cc = ConformalClassifier(self.P[self.cal], self.y[self.cal], alpha=0.1)
        cov = cc.covers(self.P[self.te], self.y[self.te]).mean()
        self.assertGreater(cov, 0.86)
        self.assertLess(cov, 0.94)
        sizes = cc.set_sizes(self.P[self.te])
        self.assertGreaterEqual(sizes.min(), 1)  # never empty in practice for a decent model
        self.assertLessEqual(sizes.max(), 4)


if __name__ == "__main__":
    unittest.main()
