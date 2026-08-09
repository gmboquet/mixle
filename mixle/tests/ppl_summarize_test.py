"""Highest-density intervals and the posterior-summary table (mixle.ppl.summarize)."""

import unittest
import warnings

import numpy as np

import mixle.ppl as P
from mixle.ppl.summarize import hdi, posterior_summary
from mixle.stats.univariate.continuous.gaussian import GaussianDistribution


class HdiTest(unittest.TestCase):
    def test_symmetric_for_normal(self):
        z = np.random.RandomState(0).standard_normal(50000)
        lo, hi = hdi(z, 0.94)
        self.assertLess(abs(lo + hi), 0.1)  # symmetric about 0
        self.assertAlmostEqual(lo, -1.88, delta=0.1)

    def test_narrower_than_equal_tailed_for_skewed(self):
        e = np.random.RandomState(0).exponential(1.0, 50000)
        lo, hi = hdi(e, 0.9)
        ql, qh = np.quantile(e, [0.05, 0.95])
        self.assertLess(lo, 0.1)  # HDI starts near the mode at 0
        self.assertLess(hi - lo, qh - ql)  # and is narrower than the equal-tailed interval

    def test_rejects_bad_prob(self):
        with self.assertRaises(ValueError):
            hdi([1.0, 2.0, 3.0], prob=1.5)

    def test_rejects_empty_nonfinite_and_multivariate_draws(self):
        for draws in ([], [0.0, np.nan], [0.0, np.inf]):
            with self.subTest(draws=repr(draws)), self.assertRaises(ValueError):
                hdi(draws)
        with self.assertRaisesRegex(ValueError, "select one coordinate"):
            hdi(np.zeros((2, 3)))

    def test_one_draw_has_a_degenerate_finite_interval(self):
        self.assertEqual(hdi([2.5]), (2.5, 2.5))


class PosteriorSummaryTest(unittest.TestCase):
    def test_table_has_mean_sd_hdi(self):
        data = GaussianDistribution(1.0, 4.0).sampler(seed=0).sample(300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = P.Normal(P.Normal(0, 10), P.HalfNormal(5)).fit(data, how="mcmc", draws=400)
        ps = posterior_summary(m)
        self.assertGreaterEqual(len(ps), 2)
        for row in ps.values():
            for key in ("mean", "sd", "hdi_low", "hdi_high"):
                self.assertIn(key, row)
            self.assertLessEqual(row["hdi_low"], row["mean"])
            self.assertLessEqual(row["mean"], row["hdi_high"])

    def test_parameter_ess_is_computed_from_each_parameters_draws(self):
        rng = np.random.RandomState(10)
        iid = rng.normal(size=400)
        sticky = np.repeat(rng.normal(size=20), 20)

        class Fitted:
            _result = None

            def summary(self):
                return {
                    "iid": {"mean": float(iid.mean()), "std": float(iid.std())},
                    "sticky": {"mean": float(sticky.mean()), "std": float(sticky.std())},
                }

            def posterior(self, name):
                return {"iid": iid, "sticky": sticky}[name]

        table = posterior_summary(Fitted())
        self.assertGreater(table["iid"]["ess"], table["sticky"]["ess"])
        # STAT-RR17-11: a flat draw vector is ONE chain -- mixing is unassessable without R-hat,
        # so this is never "ok" (the old semantics published parameter numbers under "ok" here);
        # the mean's Monte Carlo noise floor now travels with the number
        for name in ("iid", "sticky"):
            self.assertEqual(table[name]["diagnostic_status"], "single-chain-mixing-unassessable")
            self.assertIsNotNone(table[name]["mcse"])
            self.assertGreater(table[name]["mcse"], 0.0)
        self.assertGreater(table["sticky"]["mcse"], table["iid"]["mcse"])  # fewer effective draws

    def test_failed_diagnostics_are_explicit_in_fixed_schema(self):
        class Fitted:
            _result = None

            def summary(self):
                return {"theta": {"mean": 1.0, "sd": 0.2}}

            def posterior(self, _name):
                raise RuntimeError("draw store unavailable")

        row = posterior_summary(Fitted())["theta"]
        self.assertEqual(
            set(row),
            {
                "mean",
                "sd",
                "hdi_low",
                "hdi_high",
                "ess",
                "ess_tail",
                "r_hat",
                "mcse",
                "diagnostic_status",
                "diagnostic_error",
            },
        )
        self.assertEqual(row["diagnostic_status"], "failed")
        self.assertIn("draw store unavailable", row["diagnostic_error"])
        self.assertIsNone(row["ess"])


if __name__ == "__main__":
    unittest.main()


class UnusableDiagnosticsAreNeverOkTest(unittest.TestCase):
    def test_nan_diagnostics_are_labeled_unusable(self):
        # STAT-RR17-11: NaN split-R-hat / ESS published parameter numbers under "ok"
        import numpy as np

        class Result:
            split_rhat = {"theta": float("nan")}
            bulk_ess = {"theta": 100.0}
            tail_ess = {"theta": 90.0}

        class Fitted:
            _result = Result()

            def summary(self):
                return {"theta": {"mean": 1.0, "std": 0.2}}

            def posterior(self, _name):
                return np.random.RandomState(0).normal(size=(2, 200))

        row = posterior_summary(Fitted())["theta"]
        self.assertEqual(row["diagnostic_status"], "unusable")
        self.assertNotEqual(row["diagnostic_status"], "ok")

    def test_multi_chain_finite_diagnostics_are_ok_with_mcse(self):
        import numpy as np

        class Result:
            split_rhat = {"theta": 1.01}
            bulk_ess = {"theta": 350.0}
            tail_ess = {"theta": 300.0}

        class Fitted:
            _result = Result()

            def summary(self):
                return {"theta": {"mean": 1.0, "std": 0.2}}

            def posterior(self, _name):
                return np.random.RandomState(0).normal(size=(4, 200))

        row = posterior_summary(Fitted())["theta"]
        self.assertEqual(row["diagnostic_status"], "ok")
        self.assertAlmostEqual(row["mcse"], 0.2 / np.sqrt(350.0), places=10)
