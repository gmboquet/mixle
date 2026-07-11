"""Engine-resident E-step parity for the dirac-length mixture (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import CategoricalDistribution, DiracLengthMixtureDistribution

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class DiracLengthEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = DiracLengthMixtureDistribution(CategoricalDistribution({2: 0.3, 3: 0.3, 4: 0.4}), p=0.6, v=0)
        self.data = self.dist.sampler(seed=1).sample(40)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.est = self.dist.estimator()
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "DiracLengthMixtureKernel")
                value = kernel.accumulate(enc, self.weights)
                self.assertTrue(np.allclose(np.asarray(hv[0]), np.asarray(value[0]), atol=1.0e-8))


if __name__ == "__main__":
    unittest.main()
