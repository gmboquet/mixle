"""Calibration as a post-condition of fit (B2): the PIT test tells if the model's UQ is honest."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import mixle.stats as st
from mixle import Model
from mixle.inference import CalibrationReport, calibration_report, optimize
from mixle.utils.optional_deps import HAS_PANDAS
from mixle.utils.optional_deps import pandas as pd


class PITCalibrationTest(unittest.TestCase):
    def test_well_specified_model_is_calibrated(self):
        train = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 800)]
        hold = [float(x) for x in np.random.RandomState(1).normal(5.0, 2.0, 400)]
        rep = calibration_report(optimize(train, st.GaussianEstimator(), out=None), hold)
        self.assertEqual(rep.method, "PIT")
        self.assertTrue(rep.is_calibrated())  # PIT error within the finite-sample noise floor
        self.assertIn("calibrated", rep.note)

    def test_misspecified_model_is_flagged(self):
        r = np.random.RandomState(2)
        bim = np.concatenate([r.normal(-6, 1, 400), r.normal(6, 1, 400)]).tolist()
        r2 = np.random.RandomState(3)
        bim_h = np.concatenate([r2.normal(-6, 1, 200), r2.normal(6, 1, 200)]).tolist()
        rep = calibration_report(optimize(bim, st.GaussianEstimator(), out=None), bim_h)
        self.assertFalse(rep.is_calibrated())  # a Gaussian on bimodal data is not calibrated
        self.assertGreater(rep.pit_error, 2.5 * rep.noise_floor())
        self.assertIn("deviates", rep.note)

    def test_noise_floor_scales_with_n(self):
        rep_small = CalibrationReport(n=100, mean_log_density=0.0, pit_error=0.0)
        rep_big = CalibrationReport(n=10000, mean_log_density=0.0, pit_error=0.0)
        self.assertGreater(rep_small.noise_floor(), rep_big.noise_floor())  # smaller n -> higher floor


class NoCDFTest(unittest.TestCase):
    def test_composite_reports_na_honestly(self):
        tr = [(float(a), float(b)) for a, b in zip(np.random.randn(200), np.random.randn(200))]
        te = [(float(a), float(b)) for a, b in zip(np.random.randn(100), np.random.randn(100))]
        model = optimize(tr, st.CompositeEstimator((st.GaussianEstimator(), st.GaussianEstimator())), out=None)
        rep = calibration_report(model, te)
        self.assertIsNone(rep.pit_error)  # no scalar CDF -> PIT not applicable
        self.assertFalse(rep.is_calibrated())  # unknown -> conservatively not calibrated
        self.assertTrue(np.isfinite(rep.mean_log_density))  # but the proper score is always reported


@unittest.skipUnless(HAS_PANDAS, "pandas not installed; pip install mixle[pandas]")
class CalibrationReportToDataFrameTest(unittest.TestCase):
    def test_pit_histogram_becomes_one_row_per_bin(self):
        # Hand-specified pit_histogram (as pit_histogram() itself would shape it for bins=4): edges has
        # bins+1 entries, counts/density/uniform have bins entries. to_dataframe() must split edges into
        # bin_left/bin_right and pass every other field straight through unrounded.
        rep = CalibrationReport(
            n=20,
            mean_log_density=-1.5,
            pit_error=0.1,
            pit_histogram={
                "counts": np.array([2, 3, 5, 10]),
                "density": np.array([0.4, 0.6, 1.0, 2.0]),
                "edges": np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                "uniform": np.array([1.0, 1.0, 1.0, 1.0]),
            },
            bins=4,
            method="PIT",
        )
        df = rep.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["bin_left", "bin_right", "count", "density", "uniform"])
        self.assertEqual(df.shape, (4, 5))
        np.testing.assert_array_equal(df["bin_left"].to_numpy(), [0.0, 0.25, 0.5, 0.75])
        np.testing.assert_array_equal(df["bin_right"].to_numpy(), [0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_equal(df["count"].to_numpy(), [2, 3, 5, 10])
        np.testing.assert_array_equal(df["density"].to_numpy(), [0.4, 0.6, 1.0, 2.0])
        np.testing.assert_array_equal(df["uniform"].to_numpy(), [1.0, 1.0, 1.0, 1.0])
        # a real PIT report's bin table integrates to 1 over [0, 1] by construction (density * width)
        self.assertAlmostEqual(float(((df["bin_right"] - df["bin_left"]) * df["density"]).sum()), 1.0, places=9)

    def test_real_pit_report_dataframe_matches_its_own_histogram(self):
        train = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 800)]
        hold = [float(x) for x in np.random.RandomState(1).normal(5.0, 2.0, 400)]
        rep = calibration_report(optimize(train, st.GaussianEstimator(), out=None), hold)
        df = rep.to_dataframe()
        self.assertEqual(df.shape, (rep.bins, 5))
        np.testing.assert_array_equal(df["count"].to_numpy(), rep.pit_histogram["counts"])
        np.testing.assert_allclose(df["density"].to_numpy(), rep.pit_histogram["density"])
        self.assertEqual(int(df["count"].sum()), rep.n)  # every held-out point lands in exactly one bin

    def test_no_histogram_becomes_one_row_summary_matching_fields(self):
        rep = CalibrationReport(n=100, mean_log_density=-2.3, pit_error=None, method="log-density", note="no CDF")
        df = rep.to_dataframe()
        self.assertEqual(list(df.columns), ["n", "mean_log_density", "pit_error", "n_invalid_scores", "method", "note"])
        self.assertEqual(df.shape, (1, 6))
        row = df.iloc[0]
        self.assertEqual(int(row["n"]), 100)
        self.assertEqual(float(row["mean_log_density"]), -2.3)
        self.assertIsNone(row["pit_error"])
        self.assertEqual(int(row["n_invalid_scores"]), 0)
        self.assertEqual(row["method"], "log-density")
        self.assertEqual(row["note"], "no CDF")

    def test_to_parquet_roundtrips(self):
        rep = CalibrationReport(
            n=8,
            mean_log_density=-0.5,
            pit_error=0.02,
            pit_histogram={
                "counts": np.array([4, 4]),
                "density": np.array([1.0, 1.0]),
                "edges": np.array([0.0, 0.5, 1.0]),
                "uniform": np.array([1.0, 1.0]),
            },
            bins=2,
            method="PIT",
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "calibration_report.parquet"
            rep.to_parquet(path)
            roundtrip = pd.read_parquet(path)
            pd.testing.assert_frame_equal(roundtrip, rep.to_dataframe())


class ModelFacadeTest(unittest.TestCase):
    def test_calibrate_opt_in_attaches_report_and_reserves_holdout(self):
        train = [float(x) for x in np.random.RandomState(0).normal(5.0, 2.0, 800)]
        m = Model(st.GaussianEstimator()).fit(train, calibrate=0.25)
        self.assertIsInstance(m.calibration, CalibrationReport)
        self.assertIsNotNone(m.calibration.pit_error)
        self.assertEqual(m._fit_info["n"], 600)  # 25% reserved -> fit on 600, not 800

    def test_calibrate_true_uses_quarter(self):
        train = [float(x) for x in np.random.RandomState(0).normal(0.0, 1.0, 400)]
        m = Model(st.GaussianEstimator()).fit(train, calibrate=True)
        self.assertIsNotNone(m.calibration)
        self.assertEqual(m._fit_info["n"], 300)

    def test_default_fit_has_no_calibration_cost(self):
        train = [float(x) for x in np.random.RandomState(0).normal(0.0, 1.0, 400)]
        m = Model(st.GaussianEstimator()).fit(train)
        self.assertIsNone(m.calibration)  # opt-in: off by default, no held-out data spent
        self.assertEqual(m._fit_info["n"], 400)


class _ExactCdfModel:
    """A model whose predictive CDF is the exact uniform(0,1) CDF; per-row scoring is configurable."""

    def __init__(self, mode):
        self.mode = mode  # "ok" | "nan" | "single" | "impossible"

    def cdf(self, y):
        return min(max(float(y), 0.0), 1.0)

    def dist_to_encoder(self):
        return self

    def seq_encode(self, rows):
        return list(rows)

    def seq_log_density(self, enc):
        if self.mode == "nan":
            return np.full(len(enc), np.nan, dtype=float)
        if self.mode == "single":
            return np.zeros(1, dtype=float)
        if self.mode == "impossible":
            return np.full(len(enc), -np.inf, dtype=float)
        return np.zeros(len(enc), dtype=float)

    def log_density(self, x):
        return 0.0


class ScoringValidityGateTest(unittest.TestCase):
    """MXR-080-1657: a calibration verdict requires one usable predictive score per held-out row."""

    HOLD = [i / 1000.0 for i in range(1000)]  # evenly spaced uniform observations

    def test_all_nan_scoring_is_not_calibrated_despite_a_perfect_pit(self):
        rep = calibration_report(_ExactCdfModel("nan"), self.HOLD)
        self.assertLess(rep.pit_error, 0.05)  # the CDF really is exact
        self.assertEqual(rep.n_invalid_scores, len(self.HOLD))
        self.assertFalse(rep.scoring_is_valid())
        self.assertFalse(rep.is_calibrated())
        self.assertIn("unavailable", rep.note)

    def test_one_score_for_every_row_is_rejected_as_a_cardinality_violation(self):
        with self.assertRaisesRegex(ValueError, "exactly one predictive score per row"):
            calibration_report(_ExactCdfModel("single"), self.HOLD)

    def test_exact_scoring_still_passes(self):
        rep = calibration_report(_ExactCdfModel("ok"), self.HOLD)
        self.assertEqual(rep.n_invalid_scores, 0)
        self.assertTrue(rep.is_calibrated())
        self.assertIn("calibrated", rep.note)

    def test_explicitly_impossible_rows_are_valid_scores_not_invalid_ones(self):
        # -inf is the model's legitimate "this row cannot happen"; only NaN means "no score".
        rep = calibration_report(_ExactCdfModel("impossible"), self.HOLD)
        self.assertEqual(rep.n_invalid_scores, 0)
        self.assertTrue(rep.scoring_is_valid())
        self.assertTrue(rep.is_calibrated())


class DiscretePITTest(unittest.TestCase):
    """MXR-080-1617: F(Y) is uniform only for a CONTINUOUS law; atoms need the randomized PIT."""

    def test_exact_poisson_is_not_declared_uncalibrated(self):
        model = st.PoissonDistribution(1.0)
        draws = model.sampler(7).sample(5000)
        rep = calibration_report(model, draws)
        # Routed through the continuous PIT this reported error ~1.4 and "uncalibrated" for the
        # model the data was drawn from -- the three occupied bins are the discrete CDF's support.
        self.assertEqual(rep.method, "randomized-PIT")
        self.assertTrue(rep.is_calibrated())
        self.assertLess(rep.pit_error, 0.3)

    def test_a_genuinely_wrong_discrete_model_still_fails(self):
        draws = st.PoissonDistribution(1.0).sampler(7).sample(5000)
        rep = calibration_report(st.PoissonDistribution(6.0), draws)
        self.assertEqual(rep.method, "randomized-PIT")
        self.assertFalse(rep.is_calibrated())

    def test_continuous_models_keep_the_ordinary_pit(self):
        model = st.GaussianDistribution(0.0, 1.0)
        rep = calibration_report(model, model.sampler(7).sample(5000))
        self.assertEqual(rep.method, "PIT")
        self.assertTrue(rep.is_calibrated())

    def test_randomization_is_seeded_and_reproducible(self):
        model = st.PoissonDistribution(1.0)
        draws = model.sampler(7).sample(2000)
        a = calibration_report(model, draws, seed=3)
        b = calibration_report(model, draws, seed=3)
        self.assertEqual(a.pit_error, b.pit_error)
        self.assertIn("seed=3", a.note)


class CalibrationGateIsNotVacuousTest(unittest.TestCase):
    """MXR-080-1618: the default gate must not accept every possible histogram."""

    def _report(self, n):
        model = st.GaussianDistribution(0.0, 1.0)
        return calibration_report(model, model.sampler(7).sample(n))

    def test_tiny_samples_are_not_declared_calibrated(self):
        for n in (1, 10):
            rep = self._report(n)
            # threshold was 7.91 (n=1) and 2.5 (n=10) against a statistic that cannot exceed 1.8
            self.assertFalse(rep.is_calibrated())
            self.assertFalse(rep.has_enough_data())

    def test_max_attainable_error_bounds_the_statistic(self):
        rep = self._report(500)
        self.assertAlmostEqual(rep.max_pit_error(), 1.8)
        self.assertLessEqual(rep.pit_error, rep.max_pit_error())

    def test_default_threshold_is_inside_the_statistics_range(self):
        rep = self._report(500)
        self.assertTrue(rep.has_enough_data())
        self.assertLess(2.5 * rep.noise_floor(), rep.max_pit_error())

    def test_vacuous_and_non_finite_tolerances_are_rejected(self):
        rep = self._report(500)
        for bad in (float("inf"), float("nan"), -1.0, 1.8, 5.0):
            with self.assertRaises(ValueError):
                rep.is_calibrated(tol=bad)

    def test_adequate_samples_still_pass(self):
        rep = self._report(500)
        self.assertTrue(rep.is_calibrated())
        self.assertTrue(rep.is_calibrated(tol=1.0))


if __name__ == "__main__":
    unittest.main()
