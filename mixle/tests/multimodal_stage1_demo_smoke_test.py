"""Smoke test for ``examples/multimodal_stage1_demo.py``: the stage-1 receipt still holds.

Reuses the example's own building blocks (synthetic volumes, the frozen-encoder/trainable-projection/
frozen-toy-LM module) rather than re-deriving them, so this pins the actual example against
regressions instead of a parallel hand-rolled copy -- same pattern as
``peft_lora_grad_leaf_smoke_test.py`` for B5.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from multimodal_stage1_demo import backbone_param_names, build_dataset, build_module  # noqa: E402

from mixle.inference.estimation import optimize  # noqa: E402
from mixle.models import GradLeaf  # noqa: E402


class MultimodalStage1DemoSmokeTest(unittest.TestCase):
    def test_projection_trains_while_both_backbones_stay_frozen_and_likelihood_improves(self):
        data = build_dataset(n_per_class=20, seed=0)
        module = build_module(seed=0)

        before = {n: p.detach().clone() for n, p in module.named_parameters()}

        leaf = GradLeaf(module, m_steps=120, lr=5e-2)
        stacked = np.stack(data)
        before_ll = float(np.mean(leaf.seq_log_density(stacked)))

        fitted = optimize(data, leaf, max_its=4, out=None)

        after_ll = float(np.mean(fitted.seq_log_density(stacked)))
        after = dict(fitted.module.named_parameters())

        backbone_names = backbone_param_names(module)
        proj_names = [n for n in before if n.startswith("projection.")]
        self.assertTrue(backbone_names)
        self.assertTrue(proj_names)

        for name in backbone_names:  # both frozen backbones (encoder + toy LM) never moved
            self.assertTrue(torch.equal(before[name], after[name]), f"frozen parameter {name!r} changed")

        moved = [name for name in proj_names if not torch.equal(before[name], after[name])]
        self.assertTrue(moved, "expected at least one projection parameter to change during fit")

        self.assertGreater(after_ll, before_ll)  # caption likelihood improved


if __name__ == "__main__":
    unittest.main()
