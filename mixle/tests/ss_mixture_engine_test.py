"""Engine-resident E-step parity for the semi-supervised mixture (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import GaussianDistribution, GaussianEstimator, SemiSupervisedMixtureDistribution
from mixle.stats.latent.semi_supervised_mixture import SemiSupervisedMixtureEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class SemiSupervisedMixtureEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = SemiSupervisedMixtureDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5]
        )
        self.est = SemiSupervisedMixtureEstimator([GaussianEstimator(), GaussianEstimator()])
        rng = np.random.RandomState(0)
        self.data = []
        for _ in range(30):
            v = float(rng.randn() * 2)
            r = rng.rand()
            if r < 0.4:
                self.data.append((v, None))
            elif r < 0.7:
                self.data.append((v, [(0, 1.0)]))
            else:
                self.data.append((v, [(0, 0.3), (1, 0.7)]))
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "SemiSupervisedMixtureKernel")
                value = kernel.accumulate(enc, self.weights)
                self.assertTrue(np.allclose(np.asarray(hv[0]), np.asarray(value[0]), atol=1.0e-8))
                for ha, ea in zip(hv[1], value[1]):
                    self.assertTrue(np.allclose(np.asarray(ha), np.asarray(ea), atol=1.0e-7))


if __name__ == "__main__":
    unittest.main()
