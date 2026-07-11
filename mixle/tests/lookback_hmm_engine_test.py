"""Engine scoring + E-step parity for the lookback HMM (numpy + torch)."""

import unittest

import numpy as np

import mixle.stats.latent.lookback_hidden_markov_model as new_mod
from mixle.engines import NUMPY_ENGINE
from mixle.stats.combinator.null_dist import NullEstimator
from mixle.stats.combinator.sequence import SequenceDistribution, SequenceEstimator
from mixle.stats.compute.backend import backend_seq_log_density
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator
from mixle.stats.univariate.discrete.integer_categorical import (
    IntegerCategoricalDistribution,
    IntegerCategoricalEstimator,
)

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None

W = [0.6, 0.4]
TRANSITIONS = [[0.8, 0.2], [0.3, 0.7]]
EMISSION_PROBS = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]


def _make_dist(mod, len_dist):
    topics = [
        SequenceDistribution(IntegerCategoricalDistribution(0, p), len_dist=CategoricalDistribution({1: 1.0}))
        for p in EMISSION_PROBS
    ]
    init_dist = None
    return mod.LookbackHiddenMarkovModelDistribution(
        topics, w=W, transitions=TRANSITIONS, lag=0, init_dist=init_dist, len_dist=len_dist
    )


def _make_estimator(mod):
    topic_est = SequenceEstimator(
        IntegerCategoricalEstimator(min_val=0, max_val=2, pseudo_count=0.1),
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
    )
    return mod.LookbackHiddenMarkovModelEstimator(
        [topic_est] * 2,
        lag=0,
        init_estimators=[NullEstimator()] * 2,
        len_estimator=CategoricalEstimator(pseudo_count=0.1),
        pseudo_count=(1.0, 1.0),
    )


def _flatten(v):
    out = []

    def rec(u):
        if u is None:
            return
        if isinstance(u, dict):
            for _, val in sorted(u.items(), key=lambda kv: str(kv[0])):
                rec(val)
        elif isinstance(u, (tuple, list)):
            for el in u:
                rec(el)
        else:
            a = np.asarray(u, dtype=np.float64).ravel()
            if a.size:
                out.append(a)

    rec(v)
    return np.concatenate(out) if out else np.zeros(0)


class LookbackHmmEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.len_dist = CategoricalDistribution({2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25})
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def _run(self, mod):
        dist = _make_dist(mod, self.len_dist)
        rng = np.random.RandomState(5)
        data = [list(rng.randint(0, 3, size=int(rng.randint(2, 6)))) for _ in range(30)]
        weights = np.linspace(0.5, 1.5, len(data))
        enc = dist.dist_to_encoder().seq_encode(data)
        self.assertIn("torch", dist.supported_engines())
        # scoring parity
        host_ll = dist.seq_log_density(enc)
        for name, engine in self.engines:
            v = np.asarray(engine.to_numpy(backend_seq_log_density(dist, enc, engine)))
            self.assertTrue(np.allclose(host_ll, v, atol=1.0e-6), "%s/%s scoring differs" % (mod.__name__, name))
        # E-step parity
        est = _make_estimator(mod)
        host = est.accumulator_factory().make()
        host.seq_update(enc, weights, dist)
        hv = _flatten(host.value())
        for name, engine in self.engines:
            kernel = dist.kernel(engine=engine, estimator=est)
            val = _flatten(kernel.accumulate(enc, weights))
            self.assertEqual(hv.shape, val.shape, "%s/%s shape" % (mod.__name__, name))
            self.assertTrue(np.allclose(hv, val, atol=1.0e-6), "%s/%s E-step differs" % (mod.__name__, name))

    def test_new_module(self):
        self._run(new_mod)


if __name__ == "__main__":
    unittest.main()
