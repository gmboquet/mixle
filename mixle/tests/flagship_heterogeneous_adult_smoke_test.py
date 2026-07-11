"""Heterogeneous real-data flagship smoke gate (worklist F10.1 / F10.4).

Flagship A (``examples/flagship_heterogeneous_adult.py``) fits a model over real UCI Adult records --
integer and categorical fields mixed in one tuple, no manual schema -- in a single ``optimize`` call.
This is its fast, bounded gate: a small run that exercises load -> heterogeneous fit -> held-out scoring
and asserts a well-formed generalization receipt.

Needs ``datasets`` (``mixle[scientist]``) and network to fetch Adult; skips cleanly when either is
missing, so it never fails a base-install run. It runs for real in the optional CI lane.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))


@unittest.skipUnless(_HAS_DATASETS, "datasets (mixle[scientist]) not installed")
class HeterogeneousAdultFlagshipSmokeTest(unittest.TestCase):
    def test_bounded_run_fits_and_generalizes(self):
        try:
            from datasets import load_dataset

            load_dataset("scikit-learn/adult-census-income", split="train")
        except Exception as exc:  # noqa: BLE001 -- offline / dataset-host down is a skip, not a failure
            self.skipTest(f"Adult dataset unavailable (offline?): {type(exc).__name__}: {exc}")

        from flagship_heterogeneous_adult import run

        r = run(n_train=800, n_test=200, verbose=False)
        self.assertEqual(r["n_fields"], 6)
        self.assertEqual(r["model_type"], "HeterogeneousBayesianNetwork")
        self.assertTrue(math.isfinite(r["train_mean_log_density"]))
        self.assertTrue(math.isfinite(r["test_mean_log_density"]))
        # held-out density must not collapse relative to train -- the memorization / overfit tripwire
        self.assertGreater(r["test_mean_log_density"], r["train_mean_log_density"] - 2.0)


if __name__ == "__main__":
    unittest.main()
