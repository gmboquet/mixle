"""Fast, network-free tests for ``examples/flagship_heterogeneous_adult.py``'s building blocks
(worklist F10.1): calibration split, dual fit-path, baseline comparison, ``explain_fit``, and save/reload.

Exercised against a small synthetic in-memory dataset shaped like the real Adult records (mixed
int/categorical fields, a genuinely missing categorical, a planted dependency), so this suite runs in the
default fast gate with no ``datasets`` package or network fetch. The real-data end-to-end run is
``flagship_heterogeneous_adult_smoke_test.py`` (network-gated, tagged slow/integration).
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from flagship_heterogeneous_adult import (  # noqa: E402
    _FIELDS,
    explain_fit,
    fit_automatic,
    fit_baseline,
    fit_explicit,
    income_accuracy,
    load_model,
    predict_income,
    save_model,
    split_records,
)


def _synthetic_records(n: int = 1200, seed: int = 0) -> list[tuple]:
    """A small in-memory dataset with the same shape/semantics as the real Adult records this example
    fits: mixed int/categorical fields, a genuinely missing categorical (``workclass``, ~10% ``None``),
    and a planted ``workclass -> hours.per.week -> income`` dependence chain a dependency-aware model can
    exploit and an independent one cannot -- exactly the property F10.1's dual-fit-path and baseline
    comparison need to be meaningful rather than vacuous."""
    rng = np.random.RandomState(seed)
    cats = ["Private", "Self-emp", "Gov"]
    edu = ["HS-grad", "Bachelors", "Masters"]
    records = []
    for _ in range(n):
        age = int(rng.randint(20, 70))
        workclass = None if rng.rand() < 0.1 else cats[rng.randint(0, 3)]
        base = 48.0 if workclass == "Self-emp" else 38.0
        hours = float(rng.randn() * 3 + base)
        education = edu[rng.randint(0, 3)]
        sex = "Male" if rng.rand() < 0.5 else "Female"
        income = ">50K" if hours > 43 else "<=50K"
        records.append((age, workclass, education, hours, sex, income))
    return records


class SplitRecordsTest(unittest.TestCase):
    def test_splits_are_disjoint_and_cover_exactly_the_requested_total(self):
        records = [(i,) for i in range(600)]  # index-tagged: value equality coincides with identity here
        train, calibration, test = split_records(records, n_train=300, n_calibration=100, n_test=100, seed=1)
        self.assertEqual((len(train), len(calibration), len(test)), (300, 100, 100))
        train_s, cal_s, test_s = set(train), set(calibration), set(test)
        self.assertEqual(train_s & cal_s, set())
        self.assertEqual(train_s & test_s, set())
        self.assertEqual(cal_s & test_s, set())
        self.assertEqual(len(train_s) + len(cal_s) + len(test_s), 500)

    def test_raises_when_total_exceeds_available_records(self):
        with self.assertRaises(ValueError):
            split_records([(i,) for i in range(10)], n_train=5, n_calibration=4, n_test=4, seed=0)

    def test_deterministic_for_a_fixed_seed(self):
        records = [(i,) for i in range(200)]
        a = split_records(records, n_train=100, n_calibration=50, n_test=50, seed=7)
        b = split_records(records, n_train=100, n_calibration=50, n_test=50, seed=7)
        self.assertEqual(a, b)


class DualFitPathTest(unittest.TestCase):
    """Path A (automatic optimize()) and Path B (explicit learn_bayesian_network + calibration selection)
    must be genuinely different fitting procedures, not the same call under two names."""

    def test_explicit_path_actually_consults_calibration_data(self):
        records = _synthetic_records(1200)
        train, calibration, _ = split_records(records, n_train=800, n_calibration=200, n_test=200, seed=0)
        model_explicit, selection = fit_explicit(train, calibration, max_parents_candidates=(1, 2))
        self.assertGreater(len(model_explicit.edges()), 0)  # the planted dependence is found
        self.assertIn("chosen_max_parents", selection)
        self.assertEqual(len(selection["candidates"]), 2)
        self.assertTrue(all("calibration_mean_log_density" in c for c in selection["candidates"]))
        self.assertTrue(all(np.isfinite(c["calibration_mean_log_density"]) for c in selection["candidates"]))

    def test_explicit_selection_picks_the_best_calibration_score(self):
        records = _synthetic_records(1200)
        train, calibration, _ = split_records(records, n_train=800, n_calibration=200, n_test=200, seed=0)
        _, selection = fit_explicit(train, calibration, max_parents_candidates=(1, 2, 3))
        scores = {c["max_parents"]: c["calibration_mean_log_density"] for c in selection["candidates"]}
        self.assertEqual(selection["chosen_max_parents"], max(scores, key=scores.get))

    def test_automatic_path_never_sees_the_calibration_split(self):
        # fit_automatic's signature has no calibration argument to pass by mistake, unlike fit_explicit --
        # this pins that the two procedures are structurally distinct, not the same call under two names.
        import inspect

        self.assertNotIn("calibration", inspect.signature(fit_automatic).parameters)
        self.assertIn("calibration", inspect.signature(fit_explicit).parameters)


class BaselineComparisonTest(unittest.TestCase):
    def test_dependency_aware_model_beats_the_independent_baseline(self):
        records = _synthetic_records(1500)
        train, calibration, test = split_records(records, n_train=1000, n_calibration=200, n_test=300, seed=0)
        model_explicit, _ = fit_explicit(train, calibration)
        baseline = fit_baseline(train)
        self.assertEqual(baseline.edges() if hasattr(baseline, "edges") else [], [])  # transparent: no structure
        acc_model = income_accuracy(model_explicit, test)
        acc_baseline = income_accuracy(baseline, test)
        self.assertGreater(acc_model, acc_baseline)  # real, sensible numbers: structure beats independence here
        self.assertGreater(acc_model, 0.65)

    def test_baseline_income_prediction_is_the_constant_majority_class(self):
        # A mathematical consequence of field independence (see fit_baseline's docstring), not a
        # hand-tuned result: every test record gets the SAME predicted income under an independent model.
        records = _synthetic_records(600)
        train, _, test = split_records(records, n_train=400, n_calibration=100, n_test=100, seed=0)
        baseline = fit_baseline(train)
        predictions = {predict_income(baseline, r) for r in test}
        self.assertEqual(len(predictions), 1)


class ExplainFitTest(unittest.TestCase):
    def test_explain_fit_reports_the_planted_effect_and_is_json_safe(self):
        records = _synthetic_records(1200)
        train, calibration, _ = split_records(records, n_train=800, n_calibration=200, n_test=200, seed=0)
        model_explicit, _ = fit_explicit(train, calibration)
        report = explain_fit(model_explicit)
        self.assertGreater(len(report["edges"]) + len(report["roots"]), 0)
        json.dumps(report)  # raises TypeError if anything leaked a numpy scalar or other non-plain-JSON value
        hours_edge = next(f for f in report["edges"] if f["field"] == "hours.per.week")
        # not a placeholder: the fitted coefficient must actually recover the planted workclass effect,
        # correctly signed and well above noise. It need not recover the full planted +10 gap exactly --
        # `income` (hours' own downstream, threshold-derived label) is a stronger partial predictor and
        # typically joins as a second parent, which legitimately soaks up some of workclass's marginal
        # coefficient (a real multi-parent-regression effect, confirmed against this exact fit, not a bug).
        self.assertIn("workclass='Self-emp'", hours_edge["coefficients"])
        self.assertGreater(hours_edge["coefficients"]["workclass='Self-emp'"], 2.0)

    def test_explain_fit_handles_the_independent_baseline(self):
        records = _synthetic_records(600)
        train, _, _ = split_records(records, n_train=400, n_calibration=100, n_test=100, seed=0)
        baseline = fit_baseline(train)
        report = explain_fit(baseline)
        self.assertEqual(report["edges"], [])
        self.assertEqual(len(report["roots"]), len(_FIELDS))
        json.dumps(report)


class SaveReloadTest(unittest.TestCase):
    def test_reload_is_bit_identical_in_process(self):
        records = _synthetic_records(1200)
        train, calibration, test = split_records(records, n_train=800, n_calibration=200, n_test=200, seed=0)
        model, _ = fit_explicit(train, calibration)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.json")
            save_model(model, path)
            reloaded = load_model(path)
            self.assertIsNot(reloaded, model)
            for r in test[:50]:
                self.assertEqual(reloaded.log_density(r), model.log_density(r))

    def test_reload_in_a_fresh_os_process_is_numerically_identical(self):
        """The strong form of F10.1's save/reload requirement: a SEPARATE Python process (not just a
        fresh object in this process) loads the artifact and reproduces the identical score."""
        records = _synthetic_records(1200)
        train, calibration, test = split_records(records, n_train=800, n_calibration=200, n_test=200, seed=0)
        model, _ = fit_explicit(train, calibration)
        probe = test[0]
        original_ll = model.log_density(probe)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.json")
            save_model(model, path)
            examples_dir = str(Path(__file__).resolve().parents[2] / "examples")
            script = (
                f"import sys; sys.path.insert(0, {examples_dir!r})\n"
                "from flagship_heterogeneous_adult import load_model\n"
                f"m = load_model({path!r})\n"
                f"print(repr(m.log_density({probe!r})))\n"
            )
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr)
            reloaded_ll = float(result.stdout.strip())
            self.assertEqual(reloaded_ll, original_ll)


if __name__ == "__main__":
    unittest.main()
