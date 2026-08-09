"""Smoke test for ``examples/label_economics_demo.py``: the PAIRED receipt still holds.

Reuses the example's own ``run_demo`` rather than re-deriving the budgeted labeling loop, so this
pins the actual example against regressions. Small/fast scale; the pinned claim is the example's
own paired one (STAT-RR19-01): over paired seeds EIG is never worse more often than better, and
the receipt carries the win/tie/loss record with an exact sign test -- no pooled ratio exists to
pin anymore, by design.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from label_economics_demo import run_demo  # noqa: E402


class LabelEconomicsDemoSmokeTest(unittest.TestCase):
    def test_paired_receipt_reports_wins_ties_losses_and_sign_test(self) -> None:
        result = run_demo(
            pool_size=150,
            ho_size=300,
            seed_size=6,
            budgets=list(range(6, 31, 2)),
            target=-0.35,
            n_seeds=4,
            n_members=12,
        )

        self.assertEqual(result["n_seeds"], 4)
        self.assertGreaterEqual(result["n_joint"], 1, "no seed reached the target under both strategies")
        self.assertEqual(result["wins"] + result["ties"] + result["losses"], result["n_joint"])
        self.assertGreaterEqual(result["wins"], result["losses"])  # qualitative: EIG not worse here
        self.assertTrue(0.0 <= result["p_sign"] <= 1.0)
        self.assertNotIn("ratio", result)  # the pooled multiple is gone by design


if __name__ == "__main__":
    unittest.main()
