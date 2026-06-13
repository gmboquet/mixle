"""Parity tests for the engine-routed HMM forward-backward.

``hmm_engine_forward_backward`` is a single log-space implementation in ComputeEngine ops. It must
reproduce, on every engine (numpy and torch), both the HMM log-likelihood (``seq_log_density``) and
the Baum-Welch sufficient statistics that the legacy accumulator collects (initial/state/transition
expected counts), including per-sequence weights.
"""
import unittest

import numpy as np

from pysp.stats import HiddenMarkovModelDistribution, CategoricalDistribution
from pysp.stats.hidden_markov import hmm_pad_log_emissions, hmm_engine_forward_backward
from pysp.engines import NUMPY_ENGINE

try:
    from pysp.engines import TorchEngine
    _TORCH = TorchEngine(device='cpu', dtype='float64')
except Exception:
    _TORCH = None


def _model():
    topics = [CategoricalDistribution({'a': 0.7, 'b': 0.2, 'c': 0.1}),
              CategoricalDistribution({'a': 0.1, 'b': 0.3, 'c': 0.6})]
    return HiddenMarkovModelDistribution(
        topics, [0.6, 0.4], [[0.7, 0.3], [0.4, 0.6]],
        len_dist=CategoricalDistribution({3: 0.4, 4: 0.3, 5: 0.3}), use_numba=True)


def _fb_inputs(dist, data):
    """Build (log_emit padded, mask, sz) and the flat per-state log emissions for `data`."""
    _, (numba_enc, _) = dist.dist_to_encoder().seq_encode(data)
    idx, sz, xs = numba_enc
    tot = int(np.sum(sz))
    pr = np.empty((tot, dist.n_states), dtype=np.float64)
    for i in range(dist.n_states):
        pr[:, i] = dist.topics[i].seq_log_density(xs)
    padded, mask, _ = hmm_pad_log_emissions(pr, sz)
    return padded, mask, np.asarray(sz)


class HmmEngineForwardBackwardTestCase(unittest.TestCase):

    def setUp(self):
        self.dist = _model()
        self.data = self.dist.sampler(seed=3).sample(40)
        self.engines = [('numpy', NUMPY_ENGINE)]
        if _TORCH is not None:
            self.engines.append(('torch', _TORCH))

    def test_log_likelihood_parity(self):
        padded, mask, _ = _fb_inputs(self.dist, self.data)
        log_w = np.log(self.dist.w)
        log_a = np.log(self.dist.transitions)
        ref = np.asarray(self.dist.seq_log_density(self.dist.dist_to_encoder().seq_encode(self.data)))
        len_ll = np.asarray([self.dist.len_dist.log_density(len(x)) for x in self.data])
        for name, engine in self.engines:
            with self.subTest(engine=name):
                ll, _, _, _ = hmm_engine_forward_backward(engine, padded, log_w, log_a, mask)
                ll = np.asarray(engine.to_numpy(ll))
                self.assertTrue(np.allclose(ll + len_ll, ref, atol=1.0e-9),
                                '%s log-likelihood disagrees with seq_log_density' % name)

    def test_estep_counts_parity(self):
        # reference: the legacy accumulator's Baum-Welch sufficient statistics
        est = self.dist.estimator()
        acc = est.accumulator_factory().make()
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        weights = np.linspace(0.5, 1.5, len(self.data))
        acc.seq_update(enc, weights, self.dist)

        padded, mask, _ = _fb_inputs(self.dist, self.data)
        log_w = np.log(self.dist.w)
        log_a = np.log(self.dist.transitions)
        for name, engine in self.engines:
            with self.subTest(engine=name):
                _, gamma, xi_sum, pi = hmm_engine_forward_backward(
                    engine, padded, log_w, log_a, mask, weights=weights)
                gamma = np.asarray(engine.to_numpy(gamma))
                xi_sum = np.asarray(engine.to_numpy(xi_sum))
                pi = np.asarray(engine.to_numpy(pi))
                init_counts = pi.sum(axis=0)
                state_counts = gamma.sum(axis=(0, 1))
                self.assertTrue(np.allclose(init_counts, acc.init_counts, atol=1.0e-8),
                                '%s init counts differ' % name)
                self.assertTrue(np.allclose(state_counts, acc.state_counts, atol=1.0e-8),
                                '%s state counts differ' % name)
                self.assertTrue(np.allclose(xi_sum, acc.trans_counts, atol=1.0e-8),
                                '%s transition counts differ' % name)


if __name__ == '__main__':
    unittest.main()
