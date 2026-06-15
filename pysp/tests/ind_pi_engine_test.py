"""Engine-resident E-step parity for the per-sequence-init HMM (IndPi).

IndPi reuses the generalized forward-backward with a per-sequence initial vector. This also exercises
the fix to the blocked (use_numba=False) host seq_update, whose per-sequence initial-state counts
were previously broken.
"""

import unittest

import numpy as np

from pysp.engines import NUMPY_ENGINE
from pysp.stats import CategoricalDistribution, CategoricalEstimator
from pysp.stats.latent.hidden_markov_ind_pi import IndPiHiddenMarkovEstimator, IndPiHiddenMarkovModelDistribution

try:
    from pysp.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:
    _TORCH = None


def _model():
    topics = [CategoricalDistribution({"a": 0.7, "b": 0.3}), CategoricalDistribution({"a": 0.2, "b": 0.8})]
    return IndPiHiddenMarkovModelDistribution(
        topics,
        [[0.6, 0.4], [0.3, 0.7]],
        [[0.7, 0.3], [0.4, 0.6]],
        None,
        len_dist=CategoricalDistribution({3: 0.5, 4: 0.5}),
        use_numba=False,
    )


class IndPiEngineEStepTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = _model()
        self.data = self.dist.sampler(seed=1).sample(25)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.est = IndPiHiddenMarkovEstimator(
            [CategoricalEstimator(), CategoricalEstimator()],
            len_estimator=CategoricalEstimator(),
            pseudo_count=(1.0, 1.0),
        )
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_blocked_host_init_counts_are_per_sequence(self):
        # Regression: the blocked seq_update previously failed/ pooled the per-sequence init counts.
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        acc = self.est.accumulator_factory().make()
        acc.seq_update(enc, self.weights, self.dist)
        self.assertEqual(np.asarray(acc.value()[1]).shape, (len(self.data), 2))

    def test_engine_estep_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                acc = self.est.accumulator_factory().make()
                acc.seq_update_engine(enc, self.weights, self.dist, engine)
                v = acc.value()
                for k in (1, 2, 3):
                    self.assertTrue(
                        np.allclose(np.asarray(hv[k]), np.asarray(v[k]), atol=1.0e-8),
                        "%s suff-stat block %d differs" % (name, k),
                    )
                for ha, ea in zip(hv[4], v[4]):
                    for key in set(ha) | set(ea):
                        self.assertAlmostEqual(ha.get(key, 0.0), ea.get(key, 0.0), places=7)

    def test_kernel_routes_engine_estep(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                self.assertEqual(type(kernel).__name__, "IndPiHiddenMarkovModelKernel")
                value = kernel.accumulate(enc, self.weights)
                self.assertEqual(np.asarray(value[1]).shape, (len(self.data), 2))


if __name__ == "__main__":
    unittest.main()
