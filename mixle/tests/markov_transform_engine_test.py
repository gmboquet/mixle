"""Engine scoring + E-step parity for MarkovTransform (numpy + torch).

The sufficient statistic is a sparse matrix, so the sparse gather/scatter stays on the host; this
checks that the engine-routed per-observation responsibility arithmetic matches the host E-step and
that engine scoring matches the host log-density.
"""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats.combinator.composite import CompositeDistribution, CompositeEstimator
from mixle.stats.compute.backend import backend_seq_log_density
from mixle.stats.sequences.markov_transform import MarkovTransformDistribution, MarkovTransformEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


def _make_dist(alpha=0.05):
    nw = 3
    init_prob = np.asarray([0.5, 0.3, 0.2])
    rng = np.random.RandomState(7)
    cond_prob = rng.rand(nw * nw, nw) + 0.1
    cond_prob /= cond_prob.sum(axis=1, keepdims=True)
    len_dist = CompositeDistribution(
        (
            CategoricalDistribution({2: 0.5, 3: 0.5}),
            CategoricalDistribution({2: 0.5, 3: 0.5}),
            CategoricalDistribution({3: 0.6, 4: 0.4}),
        )
    )
    return MarkovTransformDistribution(init_prob, cond_prob, alpha=alpha, len_dist=len_dist)


def _make_est():
    len_est = CompositeEstimator((CategoricalEstimator(), CategoricalEstimator(), CategoricalEstimator()))
    return MarkovTransformEstimator(3, alpha=0.05, len_estimator=len_est)


class MarkovTransformEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = _make_dist()
        self.data = self.dist.sampler(seed=11).sample(size=25)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.enc = self.dist.dist_to_encoder().seq_encode(self.data)
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_scoring_parity(self):
        host = self.dist.seq_log_density(self.enc)
        for name, engine in self.engines:
            with self.subTest(engine=name):
                v = np.asarray(engine.to_numpy(backend_seq_log_density(self.dist, self.enc, engine)))
                self.assertTrue(np.allclose(host, v, atol=1.0e-6), "%s scoring differs" % name)

    def test_estep_parity(self):
        self.assertIn("torch", self.dist.supported_engines())
        est = _make_est()
        host = est.accumulator_factory().make()
        host.seq_update(self.enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=est)
                value = kernel.accumulate(self.enc, self.weights)
                # value: (init_count, trans_count (sparse), size_value)
                self.assertTrue(
                    np.allclose(np.asarray(hv[0]), np.asarray(value[0]), atol=1.0e-7), "%s init_count differs" % name
                )
                diff = (
                    np.abs((hv[1] - value[1]).toarray()).max()
                    if hasattr(hv[1], "toarray")
                    else np.abs(np.asarray(hv[1]) - np.asarray(value[1])).max()
                )
                self.assertLess(diff, 1.0e-7, "%s trans_count differs" % name)


if __name__ == "__main__":
    unittest.main()
