"""Engine-resident E-step parity for the integer hidden association model (numpy + torch)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.latent.integer_hidden_association import (
    IntegerHiddenAssociationDistribution,
    IntegerHiddenAssociationEstimator,
)

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


COND_WEIGHTS = np.asarray([[0.80, 0.20], [0.25, 0.75], [0.55, 0.45]], dtype=float)
STATE_PROB = np.asarray([[0.70, 0.20, 0.10], [0.15, 0.25, 0.60]], dtype=float)


class IntHiddenAssociationEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = IntegerHiddenAssociationDistribution(
            state_prob_mat=STATE_PROB, cond_weights=COND_WEIGHTS, alpha=0.30, use_numba=False
        )
        self.data = [
            ([(0, 2.0), (1, 1.0)], [(0, 3.0), (1, 2.0)]),
            ([(1, 1.0), (2, 2.0)], [(2, 1.0)]),
            ([(0, 1.0)], [(0, 1.0), (1, 1.0), (2, 2.0)]),
            ([(2, 3.0), (0, 1.0)], [(1, 2.0), (2, 1.0)]),
        ]
        self.weights = np.array([1.0, 0.7, 1.3, 0.9])
        self.est = IntegerHiddenAssociationEstimator(num_vals=[3, 3], num_states=2, alpha=0.30, use_numba=False)
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        # value(): (init_count, weight_count, state_count, prev_ss, size_ss)
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "IntegerHiddenAssociationKernel")
                value = kernel.accumulate(enc, self.weights)
                self.assertTrue(
                    np.allclose(np.asarray(hv[0]), np.asarray(value[0]), atol=1.0e-9), "%s init_count differ" % name
                )
                self.assertTrue(
                    np.allclose(np.asarray(hv[1]), np.asarray(value[1]), atol=1.0e-9), "%s weight_count differ" % name
                )
                self.assertTrue(
                    np.allclose(np.asarray(hv[2]), np.asarray(value[2]), atol=1.0e-9), "%s state_count differ" % name
                )


if __name__ == "__main__":
    unittest.main()
