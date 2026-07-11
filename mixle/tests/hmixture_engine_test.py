"""Engine-resident E-step parity for the hierarchical mixture (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import (
    CategoricalDistribution,
    GaussianDistribution,
    GaussianEstimator,
    HierarchicalMixtureDistribution,
)
from mixle.stats.latent.hierarchical_mixture import HierarchicalMixtureEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class HierarchicalMixtureEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = HierarchicalMixtureDistribution(
            [GaussianDistribution(-2.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(3.0, 1.0)],
            [0.5, 0.5],
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]],
            len_dist=CategoricalDistribution({2: 0.5, 3: 0.5}),
        )
        self.data = self.dist.sampler(seed=1).sample(30)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.est = HierarchicalMixtureEstimator(
            [GaussianEstimator(), GaussianEstimator(), GaussianEstimator()],
            num_mixtures=2,
            len_estimator=CategoricalDistribution({2: 0.5}).estimator(),
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
                self.assertEqual(type(kernel).__name__, "HierarchicalMixtureKernel")
                value = kernel.accumulate(enc, self.weights)
                self.assertTrue(
                    np.allclose(np.asarray(hv[0]), np.asarray(value[0]), atol=1.0e-8),
                    "%s component counts differ" % name,
                )
                for ha, ea in zip(hv[1], value[1]):
                    self.assertTrue(
                        np.allclose(np.asarray(ha), np.asarray(ea), atol=1.0e-7), "%s topic suff-stats differ" % name
                    )


if __name__ == "__main__":
    unittest.main()
