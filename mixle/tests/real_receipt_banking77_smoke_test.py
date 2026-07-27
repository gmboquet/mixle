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
        from real_receipt_banking77 import Banking77UnavailableError, load_banking77, run

        try:
            dataset = load_banking77()
        except Banking77UnavailableError as exc:
            self.skipTest(str(exc))
        result = run(
            n_seed=1155,
            n_round=40,
            n_rounds=1,
            n_test=60,
            student="generative",
            verbose=False,
            dataset=dataset,
        )
        metrics = result["metrics"]
        self.assertEqual(metrics["task"], "banking77 intents (77 classes)")
        self.assertEqual(metrics["n_test"], 60)
        for key in ("end_to_end_accuracy", "local_agreement", "escalation_rate"):
            self.assertTrue(0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}")
        self.assertGreater(metrics["escalation_rate"], 0.0)  # a small student on 77 classes must escalate
        self.assertEqual(len(result["rounds"]), 1)
        self.assertTrue(0.0 <= result["rounds"][0]["accuracy"] <= 1.0)
        self.assertEqual(result["dataset"]["source_commit"], "9d081458ff52e53cf7e848f414e6e9344e4e6696")


if __name__ == "__main__":
    unittest.main()
