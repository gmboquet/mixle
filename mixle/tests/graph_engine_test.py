"""Engine-backend scoring parity for the graph distributions.

ErdosRenyi and StochasticBlock graphs now route their Bernoulli edge log-likelihood through the
ComputeEngine backend layer, so they score on numpy and torch with parity to seq_log_density.
"""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import ErdosRenyiGraphDistribution, StochasticBlockGraphDistribution
from mixle.stats.compute.backend import backend_seq_log_density

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


class GraphEngineParityTestCase(unittest.TestCase):
    def _dists(self):
        return [
            ErdosRenyiGraphDistribution(0.3, num_nodes=6),
            ErdosRenyiGraphDistribution(0.45, num_nodes=7, directed=True),
            StochasticBlockGraphDistribution([[0.8, 0.2], [0.2, 0.7]], [0, 0, 1, 1, 0, 1]),
            StochasticBlockGraphDistribution(
                [[0.9, 0.1], [0.1, 0.6]], [0, 1, 0, 1, 1, 0], include_assignment_prior=True
            ),
        ]

    def test_backend_scoring_parity(self):
        engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])
        for dist in self._dists():
            data = dist.sampler(seed=2).sample(15)
            enc = dist.dist_to_encoder().seq_encode(data)
            ref = np.asarray(dist.seq_log_density(enc))
            for name, engine in engines:
                with self.subTest(family=type(dist).__name__, engine=name):
                    self.assertTrue(dist.supports_engine(engine))
                    got = np.asarray(engine.to_numpy(backend_seq_log_density(dist, enc, engine)))
                    self.assertTrue(
                        np.allclose(got, ref, atol=1.0e-7),
                        "%s/%s backend score disagrees with seq_log_density" % (type(dist).__name__, name),
                    )


if __name__ == "__main__":
    unittest.main()
