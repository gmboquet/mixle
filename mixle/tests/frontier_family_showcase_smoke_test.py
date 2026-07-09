"""Smoke test for ``examples/frontier_family_showcase.py`` (roadmap B6): the lifecycle receipt still
holds at a reduced, fast scale.

Reuses the example's own building blocks (``train_headline``, ``build_edge_cascade_receipt``) rather
than re-deriving them, so this pins the actual example against regressions instead of a parallel
hand-rolled copy. Runs a single rung and a tiny edge-cascade fixture -- real machinery throughout, just
far less of it than the example's own laptop-scale run -- to stay well under 60s.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from frontier_family_showcase import build_edge_cascade_receipt, train_headline  # noqa: E402

from mixle.task.checkpoint_family_ladder import RungSpec, build_checkpoint_family  # noqa: E402
from mixle.task.deploy_family import deploy_family  # noqa: E402
from mixle.task.economics import CostModel  # noqa: E402

pytestmark = pytest.mark.fast


class FrontierFamilyShowcaseSmokeTest(unittest.TestCase):
    def test_lifecycle_receipt_at_reduced_scale(self):
        vocab, d_model, n_layer, n_head, block = 23, 16, 4, 2, 12
        headline = train_headline(
            seed=7, vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block, steps=200
        )

        calib_rng = np.random.RandomState(7)
        calibration_data = torch.as_tensor(calib_rng.randint(0, vocab, size=(60, block)), dtype=torch.long)
        rung_specs = [
            RungSpec(name="rung_edge", real_target="edge-equivalent stand-in", budget=1.5, trust_region=1.5, seed=0)
        ]
        family = build_checkpoint_family(headline, rung_specs, calibration_data=calibration_data, eval_n_examples=64)

        edge_receipt = build_edge_cascade_receipt(n_train_frontier=200, n_train_student=80, n_cal=40, n_test=60)
        self.assertTrue(edge_receipt.earns_its_complexity())

        serve_receipt = deploy_family(
            family, headline, edge_cascade_receipt=edge_receipt, cost=CostModel(c_frontier=1.0), seed=0
        )
        print("\n" + serve_receipt.summary())

        self.assertEqual(len(serve_receipt.points), 1 + len(family.rungs))
        for p in serve_receipt.points:
            self.assertGreater(p.cost_per_request, 0.0)
            self.assertLess(p.artifact.quantized_bytes, p.artifact.dense_bytes)
            self.assertGreaterEqual(p.quality, 0.0)
            self.assertLessEqual(p.quality, 1.0)
        self.assertIs(serve_receipt.edge_cascade, edge_receipt)


if __name__ == "__main__":
    unittest.main()
