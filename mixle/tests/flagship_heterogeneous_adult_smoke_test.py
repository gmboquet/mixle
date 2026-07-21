"""Heterogeneous real-data flagship smoke gate (worklist F10.1 / F10.4).

Flagship A (``examples/flagship_heterogeneous_adult.py``) fits real UCI Adult records -- integer and
categorical fields mixed in one tuple, no manual schema, a genuine missing category -- through a
calibration-gated dual fit-path, compares both against a transparent independent-fields baseline, reports
held-out log score plus a task-relevant income-prediction accuracy, explains the selected fit, and
verifies save/reload. This is its bounded gate: a small run that exercises the whole workflow end to end
and asserts a well-formed, sensible receipt (not just "doesn't crash").

Needs ``datasets`` (``mixle[scientist]``) and network to fetch Adult; skips cleanly when either is
missing, so it never fails a base-install run. It runs for real in the optional/slow CI lane
(``mixle/tests/conftest.py`` tags this file ``integration``/``slow``, matching
``real_receipt_banking77_smoke_test.py``'s sibling gate).

The piece-level correctness of every new mechanism (calibration split, dual fit-path, baseline
comparison, ``explain_fit``, save/reload including a genuinely fresh OS process) is covered without
network by ``flagship_heterogeneous_adult_test.py`` against synthetic data; this file's job is only to
confirm the whole thing actually works on the real dataset within a bounded budget.
"""

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


@unittest.skipUnless(_HAS_DATASETS, "datasets (mixle[scientist]) not installed")
class HeterogeneousAdultFlagshipSmokeTest(unittest.TestCase):
    def test_bounded_run_fits_generalizes_and_produces_a_well_formed_receipt(self):
        try:
            from datasets import load_dataset
            from flagship_heterogeneous_adult import _DATASET_REPO, _DATASET_REVISION

            load_dataset(_DATASET_REPO, split="train", revision=_DATASET_REVISION)
        except Exception as exc:  # noqa: BLE001 -- offline / dataset-host down is a skip, not a failure
            self.skipTest(f"Adult dataset unavailable (offline?): {type(exc).__name__}: {exc}")

        from flagship_heterogeneous_adult import run

        with tempfile.TemporaryDirectory() as tmp:
            save_path = str(Path(tmp) / "model.json")
            r = run(n_train=600, n_calibration=200, n_test=200, save_path=save_path, verbose=False)

        self.assertEqual(r["fields"], ["age", "workclass", "education", "hours.per.week", "sex", "income"])
        self.assertEqual((r["n_train"], r["n_calibration"], r["n_test"]), (600, 200, 200))

        # dual fit-path: both an automatic and an explicit-selected model were actually produced
        self.assertIn("automatic", r["dual_fit_path"])
        selection = r["dual_fit_path"]["explicit_selection"]
        self.assertIn("chosen_max_parents", selection)
        self.assertGreaterEqual(len(selection["candidates"]), 2)

        # held-out log score + task-relevant metric for all three models (automatic, explicit, baseline)
        self.assertEqual(set(r["held_out"]), {"automatic", "explicit", "baseline"})
        for name, entry in r["held_out"].items():
            with self.subTest(model=name):
                self.assertTrue(math.isfinite(entry["train_mean_log_density"]))
                self.assertTrue(math.isfinite(entry["test_mean_log_density"]))
                # held-out density must not collapse relative to train -- the memorization/overfit tripwire
                self.assertGreater(entry["test_mean_log_density"], entry["train_mean_log_density"] - 2.0)
                self.assertGreaterEqual(entry["income_prediction_accuracy"], 0.0)
                self.assertLessEqual(entry["income_prediction_accuracy"], 1.0)

        # baseline comparison is a real, sensible number: majority-class floor <= baseline's own accuracy
        # (they coincide exactly -- see fit_baseline's docstring -- an independent-fields model cannot beat
        # the training majority class on this task)
        self.assertAlmostEqual(
            r["held_out"]["baseline"]["income_prediction_accuracy"], r["majority_class_floor"]["accuracy"], places=6
        )

        # explain_fit: real, substantive content, not a placeholder
        explanation = r["explain_fit"]
        self.assertGreater(len(explanation.get("edges", [])) + len(explanation.get("roots", [])), 0)

        # save/reload: bit-identical held-out score from the on-disk artifact
        self.assertTrue(r["save_reload"]["identical_to_original"])


if __name__ == "__main__":
    unittest.main()
