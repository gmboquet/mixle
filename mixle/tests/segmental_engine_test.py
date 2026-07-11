"""Engine-resident E-step parity for the segmental HMM (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import CategoricalDistribution, GaussianDistribution, SegmentalHiddenMarkovModelDistribution

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class SegmentalEngineEStepTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = SegmentalHiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(2.0, 1.0)],
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            len_dist=CategoricalDistribution({3: 0.5, 4: 0.5}),
        )
        self.data = self.dist.sampler(seed=1).sample(25)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.est = self.dist.estimator(pseudo_count=1.0e-6)
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "SegmentalHiddenMarkovModelKernel")
                value = kernel.accumulate(enc, self.weights)
                for k in (1, 2, 3):
                    self.assertTrue(
                        np.allclose(np.asarray(hv[k]), np.asarray(value[k]), atol=1.0e-8),
                        "%s suff-stat block %d differs" % (name, k),
                    )


if __name__ == "__main__":
    unittest.main()
