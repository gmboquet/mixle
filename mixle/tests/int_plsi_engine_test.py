"""Engine-resident E-step parity for IntegerProbabilisticLatentSemanticIndexing (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import CategoricalDistribution, IntegerProbabilisticLatentSemanticIndexingDistribution

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class IntegerProbabilisticLatentSemanticIndexingEngineTestCase(unittest.TestCase):
    def setUp(self):
        sw = np.random.RandomState(0).dirichlet(np.ones(5), size=3).T  # (5 words, 3 states)
        ds = np.random.RandomState(1).dirichlet(np.ones(3), size=4)  # (4 docs, 3 states)
        dv = np.ones(4) / 4.0
        self.dist = IntegerProbabilisticLatentSemanticIndexingDistribution(
            sw, ds, dv, len_dist=CategoricalDistribution({3: 1.0})
        )
        self.data = self.dist.sampler(seed=2).sample(15)
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
                self.assertEqual(type(kernel).__name__, "IntegerProbabilisticLatentSemanticIndexingKernel")
                value = kernel.accumulate(enc, self.weights)
                for j in range(3):
                    self.assertTrue(
                        np.allclose(np.asarray(hv[j]), np.asarray(value[j]), atol=1.0e-8),
                        "%s counts block %d differ" % (name, j),
                    )


if __name__ == "__main__":
    unittest.main()
