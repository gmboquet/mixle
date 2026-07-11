"""Engine-resident E-step parity for the joint mixture (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import GaussianDistribution, GaussianEstimator, JointMixtureDistribution
from mixle.stats.latent.joint_mixture import JointMixtureEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class JointMixtureEngineTestCase(unittest.TestCase):
    def setUp(self):
        taus12 = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
        taus21 = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
        self.dist = JointMixtureDistribution(
            [GaussianDistribution(-2.0, 1.0), GaussianDistribution(2.0, 1.0)],
            [GaussianDistribution(0.0, 1.0), GaussianDistribution(5.0, 1.0), GaussianDistribution(-5.0, 1.0)],
            [0.5, 0.5],
            [0.4, 0.3, 0.3],
            taus12,
            taus21,
        )
        self.data = self.dist.sampler(seed=1).sample(30)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.est = JointMixtureEstimator(
            [GaussianEstimator(), GaussianEstimator()], [GaussianEstimator(), GaussianEstimator(), GaussianEstimator()]
        )
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "JointMixtureKernel")
                value = kernel.accumulate(enc, self.weights)
                for j in range(3):
                    self.assertTrue(
                        np.allclose(np.asarray(hv[j]), np.asarray(value[j]), atol=1.0e-8),
                        "%s counts block %d differ" % (name, j),
                    )


if __name__ == "__main__":
    unittest.main()
