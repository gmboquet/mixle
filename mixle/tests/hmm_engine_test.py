"""Parity tests for the engine-routed HMM forward-backward.

``hmm_engine_forward_backward`` is a single log-space implementation in ComputeEngine ops. It must
reproduce, on every engine (numpy and torch), both the HMM log-likelihood (``seq_log_density``) and
the Baum-Welch sufficient statistics that the legacy accumulator collects (initial/state/transition
expected counts), including per-sequence weights.
"""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE
from mixle.stats import CategoricalDistribution, HiddenMarkovModelDistribution
from mixle.stats.latent.hidden_markov import hmm_engine_forward_backward, hmm_pad_log_emissions

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:  # noqa: BLE001
    _TORCH = None


def _model():
    topics = [
        CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
        CategoricalDistribution({"a": 0.1, "b": 0.3, "c": 0.6}),
    ]
    return HiddenMarkovModelDistribution(
        topics,
        [0.6, 0.4],
        [[0.7, 0.3], [0.4, 0.6]],
        len_dist=CategoricalDistribution({3: 0.4, 4: 0.3, 5: 0.3}),
        use_numba=True,
    )


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
        self.engines = [("numpy", NUMPY_ENGINE)]
        if _TORCH is not None:
            self.engines.append(("torch", _TORCH))

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
                self.assertTrue(
                    np.allclose(ll + len_ll, ref, atol=1.0e-9),
                    "%s log-likelihood disagrees with seq_log_density" % name,
                )

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
                _, gamma, xi_sum, pi = hmm_engine_forward_backward(engine, padded, log_w, log_a, mask, weights=weights)
                gamma = np.asarray(engine.to_numpy(gamma))
                xi_sum = np.asarray(engine.to_numpy(xi_sum))
                pi = np.asarray(engine.to_numpy(pi))
                init_counts = pi.sum(axis=0)
                state_counts = gamma.sum(axis=(0, 1))
                self.assertTrue(np.allclose(init_counts, acc.init_counts, atol=1.0e-8), "%s init counts differ" % name)
                self.assertTrue(
                    np.allclose(state_counts, acc.state_counts, atol=1.0e-8), "%s state counts differ" % name
                )
                self.assertTrue(
                    np.allclose(xi_sum, acc.trans_counts, atol=1.0e-8), "%s transition counts differ" % name
                )


class HmmEngineForwardBackwardInitTestCase(unittest.TestCase):
    """hmm_engine_forward_backward accepts a per-sequence (N, S) initial vector (SemiSupervised HMM)."""

    def test_per_sequence_initial_vector(self):
        rng = np.random.RandomState(0)
        n_states, n_seq, tmax = 3, 4, 5
        log_emit = np.log(rng.rand(n_seq, tmax, n_states) + 0.1)
        mask = np.ones((n_seq, tmax))
        mask[0, 4] = 0.0
        mask[1, 3:] = 0.0
        log_a = np.log(rng.dirichlet(np.ones(n_states), size=n_states))
        log_w_seq = np.log(rng.dirichlet(np.ones(n_states), size=n_seq))
        engines = [NUMPY_ENGINE] + ([_TORCH] if _TORCH is not None else [])
        for engine in engines:
            with self.subTest(engine=engine.name):
                ll, _, _, _ = hmm_engine_forward_backward(engine, log_emit, log_w_seq, log_a, mask)
                ll = np.asarray(engine.to_numpy(ll))
                # each sequence's per-row init result equals running it alone with that shared init
                for n in range(n_seq):
                    one, _, _, _ = hmm_engine_forward_backward(
                        engine, log_emit[n : n + 1], log_w_seq[n], log_a, mask[n : n + 1]
                    )
                    self.assertAlmostEqual(float(np.asarray(engine.to_numpy(one))[0]), float(ll[n]), places=10)


class HmmEngineEStepTestCase(unittest.TestCase):
    """The engine-resident E-step (accumulator + kernel) matches the host Baum-Welch."""

    def setUp(self):
        topics = [
            CategoricalDistribution({"a": 0.7, "b": 0.2, "c": 0.1}),
            CategoricalDistribution({"a": 0.1, "b": 0.3, "c": 0.6}),
        ]
        self.numba_dist = HiddenMarkovModelDistribution(
            topics,
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            len_dist=CategoricalDistribution({3: 0.4, 4: 0.6}),
            use_numba=True,
        )
        self.blocked_dist = HiddenMarkovModelDistribution(
            topics,
            [0.6, 0.4],
            [[0.7, 0.3], [0.4, 0.6]],
            len_dist=CategoricalDistribution({3: 0.4, 4: 0.6}),
            use_numba=False,
        )
        self.data = self.numba_dist.sampler(seed=3).sample(40)
        self.weights = np.linspace(0.5, 1.5, len(self.data))
        self.engines = [("numpy", NUMPY_ENGINE)] + ([("torch", _TORCH)] if _TORCH is not None else [])

    def _assert_value_parity(self, host_value, eng_value, label):
        for k in (1, 2, 3):  # init / state / transition counts
            self.assertTrue(
                np.allclose(np.asarray(host_value[k]), np.asarray(eng_value[k]), atol=1.0e-8),
                "%s suff-stat block %d differs" % (label, k),
            )
        for host_acc, eng_acc in zip(host_value[4], eng_value[4]):
            for key in set(host_acc) | set(eng_acc):
                self.assertAlmostEqual(
                    host_acc.get(key, 0.0), eng_acc.get(key, 0.0), places=7, msg="%s emission counts differ" % label
                )

    def test_accumulator_seq_update_engine_parity(self):
        for dist in (self.numba_dist, self.blocked_dist):
            est = dist.estimator()
            enc = dist.dist_to_encoder().seq_encode(self.data)
            host = est.accumulator_factory().make()
            host.seq_update(enc, self.weights, dist)
            host_value = host.value()
            for name, engine in self.engines:
                with self.subTest(encoding=("numba" if dist.use_numba else "blocked"), engine=name):
                    acc = est.accumulator_factory().make()
                    acc.seq_update_engine(enc, self.weights, dist, engine)
                    self._assert_value_parity(host_value, acc.value(), name)

    def test_engine_kernel_accumulate_parity(self):
        dist = self.blocked_dist  # torch-capable encoding
        est = dist.estimator()
        enc = dist.dist_to_encoder().seq_encode(self.data)
        host = est.accumulator_factory().make()
        host.seq_update(enc, self.weights, dist)
        host_value = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                kernel = dist.kernel(engine=engine, estimator=est)
                self.assertEqual(type(kernel).__name__, "HiddenMarkovModelKernel")
                self._assert_value_parity(host_value, kernel.accumulate(enc, self.weights), name)


class HmmTorchGpuEStepTestCase(unittest.TestCase):
    """An HMM (use_numba=False) runs its full forward-backward EM on the torch engine — and thus a GPU —
    converging identically to numpy. Guards the device-portability fix (no np.asarray on a device tensor)."""

    def test_hmm_em_on_torch_matches_numpy(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianDistribution, GaussianEstimator, HiddenMarkovEstimator

        rng = np.random.RandomState(0)
        seqs = []
        for _ in range(120):
            s, st = [], rng.randint(0, 2)
            for _t in range(6):
                st = st if rng.rand() < 0.85 else 1 - st
                s.append(float(rng.normal(-4 if st == 0 else 4, 1.0)))
            seqs.append(s)
        init = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)],
            [0.5, 0.5],
            [[0.8, 0.2], [0.2, 0.8]],
            use_numba=False,
        )
        if _TORCH is not None:
            self.assertTrue(init.supports_engine(_TORCH))

        est = HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()], use_numba=False)
        ref = optimize(seqs, est, max_its=25, engine=NUMPY_ENGINE, prev_estimate=init)
        ref_mu = sorted(c.mu for c in ref.topics)
        self.assertAlmostEqual(ref_mu[0], -4.0, delta=0.6)
        self.assertAlmostEqual(ref_mu[1], 4.0, delta=0.6)
        if _TORCH is not None:
            got = optimize(seqs, est, max_its=25, engine=_TORCH, prev_estimate=init)
            np.testing.assert_allclose(sorted(c.mu for c in got.topics), ref_mu, atol=5e-3)


class HmmDefaultNumbaTorchTestCase(unittest.TestCase):
    """The DEFAULT HMM (use_numba=True) is torch-eligible: the engine consumes the numba encoding for
    both scoring (_backend_numba_encoding_ll -> hmm_engine_forward_ll) and the E-step, so the seamless
    default runs its full EM on torch/GPU while the numpy engine keeps the tuned numba host path."""

    def test_default_numba_hmm_scores_and_fits_on_torch(self):
        from mixle.inference import optimize
        from mixle.stats import GaussianDistribution, GaussianEstimator, HiddenMarkovEstimator
        from mixle.stats.compute.backend import backend_seq_log_density

        rng = np.random.RandomState(0)
        seqs = [
            [float(rng.normal(-4 if rng.rand() < 0.5 else 4, 1.0)) for _ in range(rng.randint(2, 7))] for _ in range(60)
        ]
        seqs.append([])  # the empty-sequence edge must contribute only the length term
        dist = HiddenMarkovModelDistribution(
            [GaussianDistribution(-4.0, 1.0), GaussianDistribution(4.0, 1.0)],
            [0.5, 0.5],
            [[0.8, 0.2], [0.3, 0.7]],
            len_dist=CategoricalDistribution({0: 0.05, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2, 6: 0.15}),
            use_numba=True,
        )
        if _TORCH is None:
            self.skipTest("torch not installed")
        self.assertTrue(dist.supports_engine(_TORCH))
        enc = dist.dist_to_encoder().seq_encode(seqs)
        ref = dist.seq_log_density(enc)
        got = _TORCH.to_numpy(backend_seq_log_density(dist, enc, _TORCH))
        np.testing.assert_allclose(got, ref, atol=1e-10)

        init = HiddenMarkovModelDistribution(
            [GaussianDistribution(-1.0, 1.0), GaussianDistribution(1.0, 1.0)], [0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]]
        )
        est = HiddenMarkovEstimator([GaussianEstimator(), GaussianEstimator()])  # default (numba)
        train = [s for s in seqs if s]
        ref_fit = optimize(train, est, max_its=20, engine=NUMPY_ENGINE, prev_estimate=init)
        got_fit = optimize(train, est, max_its=20, engine=_TORCH, prev_estimate=init)
        np.testing.assert_allclose(
            sorted(c.mu for c in got_fit.topics), sorted(c.mu for c in ref_fit.topics), atol=5e-3
        )


if __name__ == "__main__":
    unittest.main()
