"""Engine-resident E-step parity for LDA (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import CategoricalDistribution, CategoricalEstimator, LDADistribution
from mixle.stats.latent.lda import LDAEstimator
from mixle.utils.optsutil import count_by_value

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class LDAEngineTestCase(unittest.TestCase):
    def setUp(self):
        topics = [
            CategoricalDistribution({0: 0.6, 1: 0.2, 2: 0.1, 3: 0.1}),
            CategoricalDistribution({0: 0.1, 1: 0.1, 2: 0.4, 3: 0.4}),
            CategoricalDistribution({0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}),
        ]
        self.dist = LDADistribution(
            topics, alpha=[1.0, 1.0, 1.0], len_dist=CategoricalDistribution({4: 0.3, 5: 0.4, 6: 0.3})
        )
        raw = self.dist.sampler(seed=3).sample(40)
        self.data = [sorted(count_by_value(u).items()) for u in raw]
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.est = LDAEstimator([CategoricalEstimator() for _ in range(3)])
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "LDAKernel")
                value = kernel.accumulate(enc, self.weights)
                # sum_of_logs (index 1), doc_counts (2), topic_counts (3)
                self.assertTrue(
                    np.allclose(np.asarray(hv[1]), np.asarray(value[1]), atol=1.0e-7), "%s sum_of_logs differ" % name
                )
                self.assertTrue(
                    np.allclose(np.asarray(hv[2]), np.asarray(value[2]), atol=1.0e-8), "%s doc_counts differ" % name
                )
                self.assertTrue(
                    np.allclose(np.asarray(hv[3]), np.asarray(value[3]), atol=1.0e-7), "%s topic_counts differ" % name
                )
                for ha, ea in zip(hv[4], value[4]):
                    self.assertTrue(
                        np.allclose(np.asarray(ha[0]), np.asarray(ea[0]), atol=1.0e-7),
                        "%s topic suff-stats differ" % name,
                    )


if __name__ == "__main__":
    unittest.main()
