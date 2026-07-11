"""Engine-routed CKY inside scoring parity for HeterogeneousPCFG.

The grammar inside dynamic program now runs in ComputeEngine ops, so the PCFG scores on numpy and
torch with parity to the legacy NumPy ``seq_log_density`` - reversing its former NumPy-only status.
"""

import unittest

import numpy as np

import mixle.stats as s
from mixle.engines import NUMPY_ENGINE
from mixle.stats import (
    GaussianDistribution,
    HeterogeneousPCFGDistribution,
    StudentTDistribution,
)
from mixle.stats.compute.backend import backend_seq_log_density

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


def _numeric_grammar():
    return HeterogeneousPCFGDistribution(
        binary_rules={
            "S": [("A", "B", 0.5), ("B", "A", 0.5)],
            "A": [("A", "A", 0.3), ("A", "B", 0.7)],
            "B": [("B", "B", 0.4), ("A", "B", 0.6)],
        },
        terminal_rules={
            "A": [(GaussianDistribution(-1.0, 0.8), 1.0)],
            "B": [(StudentTDistribution(5.0, 1.0, 1.2), 1.0)],
        },
        start="S",
        nonterminals=["S", "A", "B"],
    )


class PcfgEngineParityTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = _numeric_grammar()
        rng = np.random.RandomState(0)
        self.data = [list(rng.randn(rng.randint(2, 6))) for _ in range(12)]
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_no_longer_numpy_only(self):
        self.assertNotIn(type(self.dist), set(s.numpy_only_distribution_types()))
        self.assertIn("torch", self.dist.compute_capabilities().engine_ready)

    def test_engine_inside_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        ref = np.asarray(self.dist.seq_log_density(enc))
        for name, engine in self.engines:
            with self.subTest(engine=name):
                self.assertTrue(self.dist.supports_engine(engine))
                got = np.asarray(engine.to_numpy(backend_seq_log_density(self.dist, enc, engine)))
                self.assertTrue(
                    np.allclose(got, ref, atol=1.0e-9), "%s inside score disagrees with seq_log_density" % name
                )


if __name__ == "__main__":
    unittest.main()
