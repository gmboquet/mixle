"""Smoke test for ``examples/label_economics_demo.py``: the label-count-ratio receipt still holds.

Reuses the example's own ``run_demo`` rather than re-deriving the budgeted labeling loop, so this pins
the actual example against regressions instead of a parallel hand-rolled copy. Runs at a small/fast
scale (small pool, few random seeds, a narrow target well within budget) so it is fast enough for a
test suite -- no full-scale run needed, only the qualitative receipt: EIG-ranked selection reaches the
target held-out likelihood using strictly fewer labels than random selection.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from label_economics_demo import run_demo  # noqa: E402


class LabelEconomicsDemoSmokeTest(unittest.TestCase):
    def test_eig_reaches_target_with_fewer_labels_than_random(self) -> None:
        result = run_demo(
            pool_size=150,
            ho_size=300,
            seed_size=6,
            budgets=list(range(6, 31, 2)),
            target=-0.35,
            n_random_seeds=3,
            n_members=12,
        )

        self.assertIsNotNone(result["n_eig"], "EIG-ranked selection never reached the target within budget")
        self.assertIsNotNone(result["n_random"], "random selection never reached the target within budget")
        self.assertLess(result["n_eig"], result["n_random"])
        self.assertGreater(result["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
