"""Engine-backend scoring parity for SparseMarkovAssociation.

The large word-by-word transition matrix is gathered host-side with SciPy sparse ops, but the
smoothing, logs, and segment-sum reductions now run through the ComputeEngine layer, so the model
scores on numpy and torch with parity to the legacy seq_log_density - reversing its NumPy-only status.
"""

import unittest

import numpy as np

import mixle.stats as s
from mixle.engines import NUMPY_ENGINE
from mixle.stats import CategoricalDistribution, CompositeDistribution, SparseMarkovAssociationDistribution
from mixle.stats.compute.backend import backend_seq_log_density

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


def _model():
    nw = 3
    init_prob = np.asarray([0.5, 0.3, 0.2])
    rng = np.random.RandomState(11)
    cond_prob = rng.rand(nw, nw) + 0.1
    cond_prob /= cond_prob.sum(axis=1, keepdims=True)
    len_dist = CompositeDistribution(
        (CategoricalDistribution({2: 0.5, 3: 0.5}), CategoricalDistribution({3: 0.6, 4: 0.4}))
    )
    return SparseMarkovAssociationDistribution(init_prob, cond_prob, alpha=0.1, len_dist=len_dist)


class SparseMarkovEngineParityTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = _model()
        self.data = self.dist.sampler(seed=3).sample(25)
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_no_longer_numpy_only(self):
        self.assertNotIn(type(self.dist), set(s.numpy_only_distribution_types()))
        self.assertIn("torch", self.dist.compute_capabilities().engine_ready)

    def test_engine_scoring_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        ref = np.asarray(self.dist.seq_log_density(enc))
        for name, engine in self.engines:
            with self.subTest(engine=name):
                self.assertTrue(self.dist.supports_engine(engine))
                got = np.asarray(engine.to_numpy(backend_seq_log_density(self.dist, enc, engine)))
                self.assertTrue(
                    np.allclose(got, ref, atol=1.0e-9),
                    "%s sparse-association score disagrees with seq_log_density" % name,
                )


if __name__ == "__main__":
    unittest.main()
