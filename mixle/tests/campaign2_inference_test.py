"""Inference-core repairs from the second campaign wave: receipts that describe the returned
model, tables that get the same automatic structure either way they are spelled, and loud caps.

Campaign findings covered here:

* T4-8 -- ``FitProvenance.final_objective`` was documented as "the objective value of the returned
  model" but under ``monotone=False`` (and under validation-based best-model selection) it recorded
  the trajectory's last accepted value, which belongs to a model that was NOT returned;
* T2-09b -- a pandas DataFrame passed to ``optimize()`` silently bypassed ``structure='auto'``
  (iterating a DataFrame yields its column names), so the same table fit to a different model than
  its list-of-records spelling;
* T4-3 -- a latent-variable fit truncated at the ``max_its`` cap with the objective still improving
  presented exactly like a finished fit; the ``converged=False`` flag was computed and never spoken;
* T2-01 (surviving residual) -- ``num_chunks`` / ``init_p`` are initialization-sensitive knobs that
  can change the fitted parameters for approximate-update families, which the docstrings hid, and
  the Returns section claimed convergence the receipt disproves;
* T2-04 -- the supported model-comparison path was undocumented at the core entry point.
"""

import unittest
import warnings

import numpy as np
import pytest

import mixle.inference
import mixle.ppl
from mixle.inference import optimize
from mixle.stats import GaussianEstimator, MixtureEstimator, WeibullEstimator
from mixle.stats.compute.pdist import FitProvenance
from mixle.stats.compute.sequence import seq_estimate


def _total_ll(model, data):
    enc = model.dist_to_encoder().seq_encode(data)
    return float(np.sum(model.seq_log_density(enc)))


def _overlapping_mixture_data():
    rng = np.random.default_rng(11)
    return np.r_[rng.normal(0, 1, 400), rng.normal(2.5, 1, 400), rng.normal(5, 1, 400)].tolist()


def _weibull_data():
    rng = np.random.default_rng(1)
    return (rng.weibull(0.7, 500) * 40.0).tolist()


class FinalObjectiveDescribesTheReturnedModelTest(unittest.TestCase):
    """T4-8: the receipt's final_objective is the returned model's value, never a ghost's."""

    def _non_monotone_fit(self):
        rng = np.random.default_rng(0)
        data = rng.normal(3.0, 1.0, 500).tolist()
        calls = {"n": 0}

        def strategy(enc, estimator, model):
            # First call: the real M-step (the good model). Later calls: a deliberately worse
            # proposal, the shape of a surrogate objective that walks downhill in reporting LL.
            calls["n"] += 1
            if calls["n"] == 1:
                return seq_estimate(enc, estimator, model)
            return type(model)(model.mu + 5.0, model.sigma2)

        model = optimize(data, GaussianEstimator(), strategy=strategy, monotone=False, delta=None, max_its=3)
        return model, data

    def test_final_objective_matches_the_returned_model_under_non_monotone_selection(self):
        model, data = self._non_monotone_fit()
        provenance = model.fit_provenance()
        # Before the fix this reported the LAST accepted (much worse) trajectory model's value
        # (about -25050 here) while returning the best-seen model (about -716).
        self.assertAlmostEqual(provenance.final_objective, _total_ll(model, data), places=6)

    def test_the_non_returned_trajectory_value_is_kept_under_its_own_name(self):
        model, data = self._non_monotone_fit()
        provenance = model.fit_provenance()
        self.assertIsNotNone(provenance.last_accepted_objective)
        # The trajectory ended far below the returned model: the two fields must disagree, in the
        # direction that shows best-seen selection did its job.
        self.assertLess(provenance.last_accepted_objective, provenance.final_objective - 1000.0)

    def test_a_capped_fused_fit_reports_the_value_of_the_model_it_returns(self):
        # The fused loop's likelihood lags one step; at the cap the final fold-in used to leave the
        # receipt describing the model one step BEHIND the returned one.
        data = _overlapping_mixture_data()
        estimator = MixtureEstimator([GaussianEstimator() for _ in range(3)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the T4-3 cap note, tested separately below
            model = optimize(data, estimator)
        provenance = model.fit_provenance()
        self.assertFalse(provenance.converged)
        self.assertAlmostEqual(provenance.final_objective, _total_ll(model, data), places=6)

    def _vdata_fit(self, **kwargs):
        # The second verification pass's headline case: ALL defaults plus the documented ``vdata=``
        # argument. Best-by-validation selection returns an earlier iterate than the trajectory's
        # last accepted step (mismatched receipts in 40/40 seeds before the fix), so the receipt
        # must describe the returned iterate, not the trajectory's end.
        rng = np.random.default_rng(3)
        train = rng.normal(0.0, 1.0, 80).tolist()
        valid = rng.normal(0.0, 1.0, 40).tolist()
        estimator = MixtureEstimator([GaussianEstimator() for _ in range(8)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # cap note + small-component notes are not under test
            model = optimize(train, estimator, vdata=valid, max_its=60, delta=1e-12, **kwargs)
        return model, train

    def test_validation_selection_receipt_describes_the_returned_model_fused_loop(self):
        model, train = self._vdata_fit()
        provenance = model.fit_provenance()
        self.assertAlmostEqual(provenance.final_objective, _total_ll(model, train), places=6)
        # The trajectory ran past the selected iterate; monotone acceptance makes its last accepted
        # training value at least the selected iterate's, and on this fixture strictly above it --
        # the two fields must genuinely diverge for the fix to be exercised.
        self.assertIsNotNone(provenance.last_accepted_objective)
        self.assertGreater(provenance.last_accepted_objective, provenance.final_objective + 0.5)

    def test_validation_selection_receipt_describes_the_returned_model_standard_loop(self):
        model, train = self._vdata_fit(reuse_estep_ll=False)
        provenance = model.fit_provenance()
        self.assertEqual(provenance.algorithm, "em")
        self.assertAlmostEqual(provenance.final_objective, _total_ll(model, train), places=6)
        self.assertIsNotNone(provenance.last_accepted_objective)
        self.assertGreater(provenance.last_accepted_objective, provenance.final_objective + 0.5)

    def test_an_ordinary_converged_fit_still_reports_its_own_value(self):
        data = np.random.default_rng(0).normal(3.0, 1.0, 300).tolist()
        model = optimize(data, GaussianEstimator())
        provenance = model.fit_provenance()
        self.assertTrue(provenance.converged)
        self.assertAlmostEqual(provenance.final_objective, _total_ll(model, data), places=6)

    def test_run_em_stays_self_consistent(self):
        # Anchor from the verification pass: run_em returns ``last_good`` and records that same
        # iterate's value, so it was never affected -- pin that it stays that way.
        from numpy.random import RandomState

        from mixle.inference.em import StandardEM, run_em
        from mixle.stats.compute.sequence import seq_encode, seq_initialize, seq_log_density_sum

        data = np.random.default_rng(7).normal(1.0, 2.0, 300).tolist()
        estimator = GaussianEstimator()
        enc = seq_encode(data, estimator=estimator)
        initial = seq_initialize(enc_data=enc, estimator=estimator, rng=RandomState(1), p=1.0)
        fitted = run_em(enc, estimator, initial, strategy=StandardEM())
        provenance = fitted.fit_provenance()
        self.assertAlmostEqual(provenance.final_objective, seq_log_density_sum(enc, fitted)[1], places=6)

    def test_converged_is_documented_as_describing_the_run_not_the_returned_model(self):
        # converged=True with an earlier returned iterate is truthful only because the docstring
        # says converged describes the trajectory; pin that sentence (wrap-independent).
        doc = " ".join(FitProvenance.__doc__.split())
        self.assertIn("describes the RUN, not the returned model", doc)

    def test_the_new_field_round_trips_and_sanitizes_like_the_other_measurements(self):
        base = dict(
            algorithm="em",
            estimator="GaussianEstimator",
            objective="mle",
            iterations=3,
            max_iterations=10,
            converged=False,
        )
        record = FitProvenance(**base, final_objective=-716.2, last_accepted_objective=-25049.9)
        rebuilt = FitProvenance.from_dict(record.as_dict())
        self.assertEqual(rebuilt.last_accepted_objective, -25049.9)
        # Non-finite measurements become None (strict-JSON receipts), exactly as final_objective does.
        self.assertIsNone(FitProvenance(**base, last_accepted_objective=float("-inf")).last_accepted_objective)
        with self.assertRaises(TypeError):
            FitProvenance(**base, last_accepted_objective="broken")


class DataFrameStructureParityTest(unittest.TestCase):
    """T2-09b: a table gets the same structure='auto' inference as a DataFrame or as records."""

    @classmethod
    def setUpClass(cls):
        try:
            pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy
        except ImportError:  # pragma: no cover - pandas is a test-env dependency
            raise unittest.SkipTest("pandas not installed")
        rng = np.random.default_rng(5)
        n = 400
        cls.x = rng.choice(["a", "b"], size=n)
        cls.y = np.where(cls.x == "a", rng.normal(0.0, 1.0, n), rng.normal(8.0, 1.0, n))
        cls.records = [(str(a), float(b)) for a, b in zip(cls.x, cls.y)]

    def _frame(self):
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy

        return pd.DataFrame({"x": self.x, "y": self.y})

    def test_dataframe_and_records_take_the_same_automatic_path(self):
        from_records = optimize(self.records)
        from_frame = optimize(self._frame())
        # Before the fix the records won a dependence-aware model while the identical DataFrame
        # silently fit an independent composite about 276 nats worse.
        self.assertEqual(type(from_frame).__name__, type(from_records).__name__)
        self.assertAlmostEqual(_total_ll(from_frame, self.records), _total_ll(from_records, self.records), places=6)

    def test_the_dependence_model_actually_wins_on_this_table(self):
        # Guards the fixture: parity above must not be two composites agreeing by accident.
        model = optimize(self._frame())
        self.assertNotEqual(type(model).__name__, "CompositeDistribution")

    def test_structure_off_is_still_respected_for_a_dataframe(self):
        model = optimize(self._frame(), structure="off")
        self.assertEqual(type(model).__name__, "CompositeDistribution")

    def test_fit_front_door_gets_the_same_parity(self):
        from mixle.inference import fit

        self.assertEqual(type(fit(self._frame())).__name__, type(fit(self.records)).__name__)

    def test_fields_selection_works_with_inferred_estimators(self):
        # The documented ``fields=`` remedy was unusable in the headline estimator=None mode:
        # inference sized the model to every column while encoding emitted only the selection, and
        # the fit died with a row-shape ContractError blaming the user's rows (T2-09's
        # documented-workaround defect). Inference must see the same records encoding will fit.
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy

        frame = pd.DataFrame({"drop_me": [f"ID-{i}" for i in range(len(self.records))], "x": self.x, "y": self.y})
        model = optimize(frame, out=None, fields=["x", "y"], structure="off")
        self.assertEqual(type(model).__name__, "CompositeDistribution")
        # A scalar field selects a single column and must infer a univariate family.
        scalar_model = optimize(frame, out=None, fields="y")
        self.assertTrue(np.isfinite(scalar_model.log_density(float(self.y[0]))))

    def test_fields_selection_works_from_the_fit_front_door_too(self):
        pd = pytest.importorskip("pandas")  # optional dep: base CI envs ship only numpy+scipy

        from mixle.inference import fit

        frame = pd.DataFrame({"drop_me": [f"ID-{i}" for i in range(len(self.records))], "x": self.x, "y": self.y})
        model = fit(frame, fields=["x", "y"], structure="off")
        self.assertEqual(type(model).__name__, "CompositeDistribution")


class CapTruncationDisclosureTest(unittest.TestCase):
    """T4-3: a convergence-seeking run stopped by the cap says so; everything else stays quiet."""

    def _fit(self, **kwargs):
        data = _overlapping_mixture_data()
        estimator = MixtureEstimator([GaussianEstimator() for _ in range(3)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(data, estimator, **kwargs)
        notes = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        return model, notes

    def test_a_latent_fit_truncated_at_the_default_budget_warns_with_a_remedy(self):
        model, notes = self._fit()
        provenance = model.fit_provenance()
        self.assertFalse(provenance.converged)
        self.assertEqual(provenance.iterations, provenance.max_iterations)
        self.assertEqual(len(notes), 1, notes)
        self.assertIn("max_its cap", notes[0])
        self.assertIn("converged=False", notes[0])
        self.assertIn("Raise max_its", notes[0])

    def test_a_fixed_iteration_request_is_not_second_guessed(self):
        # delta=None is the documented "fixed iteration count" spelling; stopping at the cap is
        # then the caller's intent, not a truncation.
        _, notes = self._fit(delta=None)
        self.assertEqual(notes, [])

    def test_a_run_that_reaches_convergence_stays_quiet(self):
        # Well-separated clusters: the default kmeans++ start converges inside the default budget
        # (pinned by campaign_em_test), so the note must not fire on an ordinary successful fit.
        rng = np.random.default_rng(1000)
        data = np.r_[rng.normal(0.0, 1.0, 300), rng.normal(6.0, 1.0, 300)].tolist()
        estimator = MixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(data, estimator)
        self.assertTrue(model.fit_provenance().converged)
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])

    def test_a_below_cap_rejected_update_stop_is_not_misreported_as_truncation(self):
        # The Weibull default fit stops when its first non-improving update is rejected -- more
        # iterations would not help, so the cap note must not fire (fail-closed-overreach check).
        data = _weibull_data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = optimize(data, WeibullEstimator())
        provenance = model.fit_provenance()
        self.assertFalse(provenance.converged)
        self.assertLess(provenance.iterations, provenance.max_iterations)
        self.assertEqual([str(w.message) for w in caught if issubclass(w.category, UserWarning)], [])


class InitializationSensitiveKnobDisclosureTest(unittest.TestCase):
    """T2-01 residual: what num_chunks / init_p do to the answer is stated, and the receipt is true."""

    def test_the_default_weibull_fit_is_not_the_inferior_answer(self):
        # Pins the verifier's refutation of the original framing: the default fit's log-likelihood
        # is at least the chunked run's, so num_chunks=2 is not "more correct".
        data = _weibull_data()
        default_fit = optimize(data, WeibullEstimator())
        chunked_fit = optimize(data, WeibullEstimator(), num_chunks=2)
        self.assertGreaterEqual(_total_ll(default_fit, data), _total_ll(chunked_fit, data))

    def test_raising_the_budget_does_not_change_a_rejected_update_stop(self):
        # Pins the confirmed diagnosis: the stop is the monotone gate, not the iteration budget.
        data = _weibull_data()
        short = optimize(data, WeibullEstimator())
        long = optimize(data, WeibullEstimator(), max_its=50)
        self.assertEqual(str(short), str(long))
        self.assertFalse(long.fit_provenance().converged)

    def test_the_docstrings_now_state_what_the_knobs_do(self):
        doc = optimize.__doc__
        # num_chunks: answer-changing through the initialization subsample, not execution-only.
        self.assertIn("can change the fitted parameters", doc)
        # init_p: a statistical knob whose start the trajectory depends on.
        self.assertIn("statistical knob", doc)
        # Returns: no unconditional convergence claim; the receipt is the arbiter.
        self.assertNotIn("when stopping criteria of EM algorithm\n            is met", doc)
        self.assertIn("fit_provenance()", doc)

    def test_converged_false_below_the_cap_is_a_documented_verdict(self):
        # Collapse the docstring's line wrapping so the pinned phrases are wrap-independent.
        doc = " ".join(FitProvenance.__doc__.split())
        self.assertIn("monotone acceptance gate", doc)
        self.assertIn("iterations < max_iterations", doc)


class ModelComparisonReachabilityTest(unittest.TestCase):
    """T2-04: the supported comparison path is reachable and documented at the core entry point."""

    def test_the_core_surface_exports_the_paired_comparison_tests(self):
        for name in ("vuong_test", "clarke_test", "paired_score_difference", "compare_elpd"):
            self.assertTrue(callable(getattr(mixle.inference, name)), name)

    def test_the_ic_path_exists_on_the_ppl_surface(self):
        self.assertTrue(callable(mixle.ppl.compare))
        fitted = mixle.ppl.Normal(mixle.ppl.free, mixle.ppl.free).fit(
            np.random.default_rng(0).normal(3.0, 1.0, 120).tolist()
        )
        data = np.random.default_rng(1).normal(3.0, 1.0, 40).tolist()
        self.assertTrue(np.isfinite(fitted.aic(data)))
        self.assertTrue(np.isfinite(fitted.bic(data)))

    def test_optimize_documents_the_comparison_path(self):
        doc = optimize.__doc__
        self.assertIn("Model comparison", doc)
        for pointer in ("vuong_test", "mixle.ppl", "aic"):
            self.assertIn(pointer, doc)


if __name__ == "__main__":
    unittest.main()
