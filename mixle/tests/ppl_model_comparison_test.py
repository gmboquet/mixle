"""mixle.ppl: Bayesian predictive model comparison via WAIC and PSIS-LOO.

Covers the diagnostics math directly and the RandomVariable.waic / .loo / compare(by=...) surface that
integrates over the posterior draws of a Bayesian fit.
"""

import unittest

import numpy as np
from scipy.stats import norm

from mixle.ppl import Normal, compare
from mixle.ppl.diagnostics import loo, psis_loo, waic


def _loglik(mu, y, sd=1.0):
    return np.array([norm(m, sd).logpdf(y) for m in mu])  # (n_draws, n_obs)


class DiagnosticsMathTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.y = rng.normal(0.0, 1.0, 60)
        n = len(self.y)
        self.good = _loglik(rng.normal(self.y.mean(), 1.0 / np.sqrt(n), 2000), self.y)
        self.bad = _loglik(rng.normal(3.0, 1.0 / np.sqrt(n), 2000), self.y)

    def test_waic_and_loo_prefer_the_better_model(self):
        self.assertLess(waic(self.good)["waic"], waic(self.bad)["waic"])
        self.assertLess(psis_loo(self.good)["loo"], psis_loo(self.bad)["loo"])

    def test_waic_close_to_loo_for_well_behaved_model(self):
        w, lo = waic(self.good), psis_loo(self.good)
        self.assertAlmostEqual(w["waic"], lo["loo"], delta=1.0)
        self.assertAlmostEqual(w["p_waic"], lo["p_loo"], delta=0.3)
        self.assertLess(lo["khat_max"], 0.7)  # reliable

    def test_effective_parameters_near_one(self):
        # one free parameter (mu) -> p_waic ~ 1
        self.assertAlmostEqual(waic(self.good)["p_waic"], 1.0, delta=0.6)

    def test_keys_and_standard_error(self):
        w = waic(self.good)
        for key in ("elpd_waic", "p_waic", "waic", "se", "n_draws", "pointwise"):
            self.assertIn(key, w)
        self.assertTrue(np.isfinite(w["se"]))
        self.assertEqual(len(w["pointwise"]), len(self.y))

    def test_single_draw_waic_is_defined_but_loo_is_rejected(self):
        single = self.good[:1]
        self.assertTrue(np.isfinite(waic(single)["waic"]))
        self.assertEqual(waic(single)["p_waic"], 0.0)
        with self.assertRaises(ValueError):
            psis_loo(single)

    def test_waic_rejects_empty_loglik(self):
        # An empty matrix used to sum to 0.0 and return elpd_waic=0.0/waic=0.0 -- indistinguishable
        # from "a model with a perfect fit" instead of "no data was given". Every axis-degenerate
        # shape (fully empty, zero draws, zero observations) must now raise instead.
        for shape in ((0, 0), (5, 0), (0, 5)):
            with self.assertRaises(ValueError):
                waic(np.empty(shape))
        with self.assertRaises(ValueError):
            waic(np.array([]))

    def test_psis_loo_rejects_empty_loglik(self):
        # Mirrors test_waic_rejects_empty_loglik: an empty matrix silently returned elpd_loo=0.0, and
        # a zero-draws matrix (s == 0) crashed with an opaque IndexError from `loglik[0]` deep inside
        # the s < 2 branch instead of a clear, up-front ValueError.
        for shape in ((0, 0), (5, 0), (0, 5)):
            with self.assertRaises(ValueError):
                psis_loo(np.empty(shape))
        with self.assertRaises(ValueError):
            psis_loo(np.array([]))

    def test_waic_rejects_nonfinite_loglik(self):
        nan_ll = self.good.copy()
        nan_ll[0, 0] = np.nan
        with self.assertRaises(ValueError):
            waic(nan_ll)
        inf_ll = self.good.copy()
        inf_ll[0, 0] = np.inf
        with self.assertRaises(ValueError):
            waic(inf_ll)

    def test_psis_loo_rejects_nonfinite_loglik(self):
        nan_ll = self.good.copy()
        nan_ll[0, 0] = np.nan
        with self.assertRaises(ValueError):
            psis_loo(nan_ll)
        inf_ll = self.good.copy()
        inf_ll[0, 0] = -np.inf
        with self.assertRaises(ValueError):
            psis_loo(inf_ll)

    def test_loo_alias_rejects_empty_and_nonfinite(self):
        # `loo` is psis_loo's conventional short name (a thin wrapper) -- confirm the guard is visible
        # through the alias too, not just the underlying function.
        with self.assertRaises(ValueError):
            loo(np.empty((0, 0)))
        nan_ll = self.good.copy()
        nan_ll[0, 0] = np.nan
        with self.assertRaises(ValueError):
            loo(nan_ll)


class RandomVariableComparisonTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.data = list(rng.normal(2.0, 1.0, 300))

    def _fit(self, sd, seed):
        return Normal(Normal(0, 5, name="mu"), sd).fit(
            self.data, how="mcmc", draws=1200, burn=400, rng=np.random.RandomState(seed)
        )

    def test_waic_loo_match_on_a_bayesian_fit(self):
        m = self._fit(1.0, 2)
        w, lo = m.waic(self.data), m.loo(self.data)
        self.assertGreater(w["n_draws"], 1)  # used posterior draws, not a point estimate
        self.assertAlmostEqual(w["waic"], lo["loo"], delta=2.0)
        self.assertLess(lo["khat_max"], 0.7)

    def test_compare_by_waic_and_loo_ranks_correct_model_first(self):
        right = self._fit(1.0, 2)  # correct noise scale
        wide = self._fit(3.0, 3)  # too-wide noise scale
        for crit in ("waic", "loo"):
            table = compare([wide, right], self.data, by=crit)
            self.assertLessEqual(table[0][crit], table[1][crit])
            self.assertEqual(table[0]["d_elpd"], 0.0)
            self.assertLessEqual(table[1]["d_elpd"], 0.0)
            self.assertAlmostEqual(
                right.waic(self.data)["waic"] if crit == "waic" else right.loo(self.data)["loo"],
                table[0][crit],
                delta=3.0,
            )

    def test_compare_by_aic_still_works(self):
        right = self._fit(1.0, 2)
        wide = self._fit(3.0, 3)
        rows = compare([right, wide], self.data, by="aic")
        self.assertEqual(len(rows), 2)
        self.assertIn("aic", rows[0])

    def test_point_estimate_fit_falls_back_to_single_draw(self):
        m = Normal(Normal(0, 5, name="mu"), 1.0).fit(self.data, how="map")
        w = m.waic(self.data)
        self.assertEqual(w["n_draws"], 1)
        self.assertTrue(np.isfinite(w["waic"]))

    def test_waic_and_loo_on_empty_data_raise(self):
        # RandomVariable.waic/.loo route through pointwise_log_likelihood into waic()/psis_loo(); an
        # empty `data` produces a (n_draws, 0) matrix that used to silently score as a "perfect fit"
        # (elpd_waic == 0.0 / elpd_loo == 0.0) instead of surfacing that no data was evaluated.
        m = self._fit(1.0, 2)
        with self.assertRaises(ValueError):
            m.waic([])
        with self.assertRaises(ValueError):
            m.loo([])


class SummaryTestCase(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.data = list(rng.normal(2.0, 1.0, 300))

    def test_mcmc_summary_has_credible_interval(self):
        m = Normal(Normal(0, 5, name="mu"), 1.0).fit(
            self.data, how="mcmc", draws=1500, burn=500, rng=np.random.RandomState(2)
        )
        s = m.summary()
        self.assertIn("mu", s)
        for key in ("mean", "std", "q2.5", "q97.5"):
            self.assertIn(key, s["mu"])
        self.assertLessEqual(s["mu"]["q2.5"], 2.0)
        self.assertGreaterEqual(s["mu"]["q97.5"], 2.0)  # 95% CI brackets the truth

    def test_multichain_summary_reports_rhat_and_ess(self):
        m = Normal(Normal(0, 5, name="mu"), 1.0).fit(
            self.data, how="mcmc", draws=800, burn=300, chains=2, rng=np.random.RandomState(3)
        )
        s = m.summary()
        self.assertIn("_rhat", s)
        self.assertIn("_ess", s)
        self.assertLess(abs(list(s["_rhat"].values())[0] - 1.0), 0.1)  # converged

    def test_point_estimate_summary_returns_params(self):
        m = Normal(Normal(0, 5, name="mu"), 1.0).fit(self.data, how="map")
        self.assertEqual(m.summary(), m.params)


if __name__ == "__main__":
    unittest.main()
