"""Ordinal regression + concordance measures (pysp.inference.ordinal)."""

import unittest

import numpy as np
from scipy import stats

from pysp.inference import (
    concordance_summary,
    goodman_kruskal_gamma,
    kendall_tau,
    ordinal_regression,
    somers_d,
)


class OrdinalRegressionTest(unittest.TestCase):
    def _sim(self, seed, link="logit", n=4000):
        rng = np.random.RandomState(seed)
        X = rng.normal(0, 1, (n, 2))
        beta = np.array([1.0, -0.7])
        eta = X @ beta
        noise = rng.logistic(0, 1, n) if link == "logit" else rng.normal(0, 1, n)
        thresh = np.array([-1.0, 0.5, 2.0])
        y = np.digitize(eta + noise, thresh)
        return X, y, beta, thresh

    def test_logit_recovers_coefficients_and_ordered_thresholds(self):
        X, y, beta, thresh = self._sim(0, "logit")
        r = ordinal_regression(X, y, link="logit")
        np.testing.assert_allclose(r.coef, beta, atol=0.2)
        self.assertTrue(np.all(np.diff(r.thresholds) > 0))
        np.testing.assert_allclose(r.thresholds, thresh, atol=0.3)
        self.assertEqual(r.n_categories, 4)
        self.assertTrue(np.all(r.se > 0))

    def test_probit_runs_and_orders(self):
        X, y, _, _ = self._sim(1, "probit")
        r = ordinal_regression(X, y, link="probit")
        self.assertTrue(np.all(np.diff(r.thresholds) > 0))
        # probit and logit slopes share sign
        self.assertGreater(r.coef[0], 0)
        self.assertLess(r.coef[1], 0)

    def test_predict_proba_rows_sum_to_one(self):
        X, y, _, _ = self._sim(2)
        r = ordinal_regression(X, y)
        pp = r.predict_proba(X[:10])
        self.assertEqual(pp.shape, (10, 4))
        np.testing.assert_allclose(pp.sum(axis=1), 1.0, atol=1e-9)

    def test_predict_labels_in_range(self):
        X, y, _, _ = self._sim(3)
        r = ordinal_regression(X, y)
        pred = r.predict(X[:20])
        self.assertTrue(np.all((pred >= 0) & (pred < 4)))


class ConcordanceTest(unittest.TestCase):
    def test_kendall_tau_matches_scipy(self):
        rng = np.random.RandomState(0)
        a = rng.normal(0, 1, 300)
        b = a + rng.normal(0, 1, 300)
        self.assertAlmostEqual(kendall_tau(a, b), stats.kendalltau(a, b)[0], places=9)

    def test_no_tie_measures_coincide(self):
        rng = np.random.RandomState(1)
        a = rng.normal(0, 1, 200)
        b = a + rng.normal(0, 0.5, 200)
        tau = kendall_tau(a, b)
        self.assertAlmostEqual(goodman_kruskal_gamma(a, b), tau, places=9)
        self.assertAlmostEqual(somers_d(a, b), tau, places=9)

    def test_perfect_monotone(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(kendall_tau(a, 2 * a), 1.0)
        self.assertAlmostEqual(kendall_tau(a, -a), -1.0)

    def test_summary_counts(self):
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 2, 2, 4])  # one tie on y
        s = concordance_summary(x, y)
        self.assertEqual(s["ty"], 1)  # pair (2,3) tied on y, not x
        self.assertGreater(s["concordant"], s["discordant"])


if __name__ == "__main__":
    unittest.main()
