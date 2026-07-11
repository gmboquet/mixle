"""Frontier-training surfaces are labeled as experimental prototypes (worklist A1.4).

mixle carries sharding mathematics and small-scale training simulations (muP, 2:4 sparsity, scaling-law
fits, tensor/pipeline/context parallelism, fault-tolerant training, sharded checkpointing) whose mechanics
are exact but which are NOT production frontier trainers -- the real multi-GPU / multi-node execution needs
hardware this project does not gate on. A1.4's bar is that no user can mistake these for production. This
test enforces that each such module's docstring carries an explicit experimental/prototype/simulation
signal, so a new frontier surface cannot land in a stable namespace without one.
"""

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Frontier-training prototype modules that live in stable namespaces (not under mixle.experimental).
_FRONTIER_MODULES = [
    "mixle/models/mup.py",
    "mixle/models/sparsity_2_4.py",
    "mixle/ppl/scaling_laws.py",
    "mixle/utils/parallel/context_parallel_spine.py",
    "mixle/utils/parallel/fault_tolerant_training.py",
    "mixle/utils/parallel/tensor_pipeline_context_parallel.py",
    "mixle/utils/parallel/dcp_checkpoint.py",
]

# Any one of these phrases in the module docstring signals "prototype, not production".
_SIGNALS = ("experimental", "prototype", "simulat", "not a production", "not production", "research prototype")


class FrontierPrototypeLabelTest(unittest.TestCase):
    def test_every_frontier_module_declares_prototype_status(self):
        unlabeled = []
        for rel in _FRONTIER_MODULES:
            path = REPO_ROOT / rel
            self.assertTrue(path.exists(), f"frontier module missing: {rel}")
            doc = (ast.get_docstring(ast.parse(path.read_text())) or "").lower()
            if not any(sig in doc for sig in _SIGNALS):
                unlabeled.append(rel)
        self.assertEqual(
            unlabeled,
            [],
            "these frontier-training modules do not label themselves experimental/prototype in their module "
            "docstring (A1.4 -- a user could mistake them for a production trainer):\n" + "\n".join(unlabeled),
        )


if __name__ == "__main__":
    unittest.main()
