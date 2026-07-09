"""Smoke test for ``examples/vlm_trust_receipts_demo.py`` (B4): the three composed receipts still hold.

Reuses the example's own importable ``run_demo`` rather than re-deriving the pipeline -- same pattern
as ``multimodal_stage1_demo_smoke_test.py`` for B1. Runs on a small/fast synthetic case (few embeddings
per class) and checks each of the three receipts the demo composes: the hvis health diagnostic runs and
returns a result, the disagreement gate flags at least one guaranteed (not probabilistic) disagreement
case, and the epistemic journal's ``.verify()`` confirms the audit trail after the run.
"""

import sys
import unittest
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from vlm_trust_receipts_demo import run_demo  # noqa: E402


class VlmTrustReceiptsDemoSmokeTest(unittest.TestCase):
    def test_health_gate_and_journal_receipts_all_hold_end_to_end(self):
        result = run_demo(n_per_class=10, seed=0)

        # 1. the hvis health diagnostic ran and returned a real result
        fit_health = result["fit_health"]
        self.assertIn("diagnosis", fit_health)
        self.assertIn("components", fit_health)
        self.assertIsInstance(fit_health["diagnosis"], list)

        # 2. the disagreement gate correctly flags at least one disagreement case -- guaranteed by the
        # demo's hand-built adversarial embedding, not left to chance
        gate_result = result["gate_result"]
        self.assertGreaterEqual(gate_result["n_flagged"], 1)
        self.assertTrue(bool(gate_result["mask"][-1]))  # the adversarial case is always the last row

        # 3. the epistemic journal's audit trail is intact after the run
        journal = result["journal"]
        self.assertGreater(len(journal), 0)
        self.assertTrue(result["journal_verified"])
        self.assertEqual(len(result["journal_replay"]), len(journal))


if __name__ == "__main__":
    unittest.main()
