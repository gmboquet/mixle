"""Banking77 flagship smoke gate (worklist F10.4).

The Banking77 example (``examples/real_receipt_banking77.py``) is the one flagship that runs the
teacher/student cascade against a REAL public dataset. This is its fast, bounded gate: a small run
(torch-free generative student, ~1.2k seed / 60 test) that exercises the whole pipeline end to end --
solve -> conformal-gated cascade -> scorecard -> one improve round -- and asserts the receipt is
well-formed.

Needs the ``datasets`` package (``mixle[scientist]``) and network access to fetch Banking77; it skips
cleanly when either is missing, so it never fails a base-install run. It runs for real in the optional
CI lane, which installs ``datasets``.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


@unittest.skipUnless(_HAS_DATASETS, "datasets (mixle[scientist]) not installed")
class Banking77FlagshipSmokeTest(unittest.TestCase):
    def test_bounded_run_produces_a_wellformed_receipt(self):
        # Fetching the dataset is the only network step; skip (don't fail) when it is unavailable.
        try:
            from datasets import load_dataset

            load_dataset("banking77")
        except Exception as exc:  # noqa: BLE001 -- offline / dataset-host down is a skip, not a failure
            self.skipTest(f"Banking77 unavailable (offline?): {type(exc).__name__}: {exc}")

        from real_receipt_banking77 import run

        result = run(n_seed=1155, n_round=40, n_rounds=1, n_test=60, student="generative", verbose=False)
        metrics = result["metrics"]
        self.assertEqual(metrics["task"], "banking77 intents (77 classes)")
        self.assertEqual(metrics["n_test"], 60)
        for key in ("end_to_end_accuracy", "local_agreement", "escalation_rate"):
            self.assertTrue(0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}")
        self.assertGreater(metrics["escalation_rate"], 0.0)  # a small student on 77 classes must escalate
        self.assertEqual(len(result["rounds"]), 1)
        self.assertTrue(0.0 <= result["rounds"][0]["accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
