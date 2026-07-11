"""Engine scoring + E-step parity for LabeledLDA (numpy + torch)."""

import unittest

import numpy as np
from numpy.random import RandomState

from mixle.engines import NUMPY_ENGINE
from mixle.stats.compute.backend import backend_seq_log_density
from mixle.stats.latent.labeled_lda import LabeledLDADistribution, LabeledLDAEstimator
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None

VOCAB = ["w0", "w1", "w2", "w3"]
PMATS = [[0.4, 0.4, 0.1, 0.1], [0.1, 0.1, 0.4, 0.4]]


def _data(label_sets, n, seed):
    rng = RandomState(seed)
    out = []
    for i in range(n):
        labels = list(label_sets[i % len(label_sets)])
        p = np.asarray(PMATS[labels[0] % 2]) * 0.7 + np.asarray(PMATS[(labels[-1] + 1) % 2]) * 0.3
        words = rng.choice(4, size=9, p=p / p.sum())
        cnts = {}
        for wd in words:
            cnts[VOCAB[wd]] = cnts.get(VOCAB[wd], 0) + 1
        out.append((sorted(cnts.items()), labels))
    return out


class LabeledLDAEngineTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = LabeledLDADistribution(
            [
                CategoricalDistribution({"w0": 0.4, "w1": 0.4, "w2": 0.1, "w3": 0.1}),
                CategoricalDistribution({"w0": 0.1, "w1": 0.1, "w2": 0.4, "w3": 0.4}),
            ],
            np.asarray([[1.0, 1.0], [1.5, 0.5], [0.5, 1.5]], dtype=float),
            gamma_threshold=1.0e-10,
        )
        self.data = _data([[0], [1], [0, 2], [1, 2]], 24, seed=7)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.est = LabeledLDAEstimator(
            [CategoricalEstimator(), CategoricalEstimator()], num_alphas=3, gamma_threshold=1.0e-10
        )
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def test_scoring_parity(self):
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.dist.seq_log_density(enc)
        for name, engine in self.engines:
            with self.subTest(engine=name):
                v = np.asarray(engine.to_numpy(backend_seq_log_density(self.dist, enc, engine)))
                self.assertTrue(np.allclose(host, v, atol=1.0e-6), "%s ELBO differs" % name)

    def test_estep_parity(self):
        self.assertIn("torch", self.dist.supported_engines())
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        host = self.est.accumulator_factory().make()
        host.seq_update(enc, self.weights, self.dist)
        hv = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = self.dist.kernel(engine=engine, estimator=self.est)
                value = kernel.accumulate(enc, self.weights)
                # value: (prev_alpha, set_stats, doc_counts, topic_counts, topic_accs)
                self.assertTrue(
                    np.allclose(np.asarray(hv[2]), np.asarray(value[2]), atol=1.0e-6), "%s doc_counts differ" % name
                )
                self.assertTrue(
                    np.allclose(np.asarray(hv[3]), np.asarray(value[3]), atol=1.0e-6), "%s topic_counts differ" % name
                )
                for ha, ea in zip(hv[4], value[4]):
                    hd = ha[0] if isinstance(ha, tuple) else ha
                    ed = ea[0] if isinstance(ea, tuple) else ea
                    keys = sorted(set(hd) | set(ed), key=str)
                    hvec = np.asarray([hd.get(k, 0.0) for k in keys], dtype=float)
                    evec = np.asarray([ed.get(k, 0.0) for k in keys], dtype=float)
                    self.assertTrue(np.allclose(hvec, evec, atol=1.0e-6), "%s topic suff-stats differ" % name)


if __name__ == "__main__":
    unittest.main()
