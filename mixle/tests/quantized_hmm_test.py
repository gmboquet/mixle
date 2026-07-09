"""Tests for mixle.stats.latent.quantized_hidden_markov_model.

Covers the theta^k parameterization (row normalization, log-prob structure, structural zeros,
stationary init mode), agreement with the dense HiddenMarkovModelDistribution, sampling, and EM
estimation smoke runs (free theta, fixed theta, k_max caps) on tiny synthetic data with fixed
seeds.
"""

import itertools
import unittest

import numpy as np
from numpy.random import RandomState

from mixle.engines import NUMPY_ENGINE
from mixle.inference import seq_estimate, seq_initialize
from mixle.stats.combinator.null_dist import NullDistribution
from mixle.stats.latent.hidden_markov import HiddenMarkovModelDistribution
from mixle.stats.univariate.discrete.categorical import CategoricalDistribution, CategoricalEstimator

try:
    from mixle.engines import TorchEngine

    _TORCH = TorchEngine(device="cpu", dtype="float64")
except Exception:
    _TORCH = None
from mixle.enumeration.algorithms import freeze
from mixle.stats.latent.quantized_hidden_markov_model import (
    QuantizedHiddenMarkovEstimator,
    QuantizedHiddenMarkovModelDistribution,
    QuantizedHiddenMarkovModelEnumerator,
    _split_collapsed_states,
)


def make_quantized_dist(init_mode="quantized", use_numba=False, theta=0.5):
    levels = ["a", "b", "c"]
    trans_exp = [[0, 1], [2, 0]]
    emis_exp = [[0, 1, 2], [2, 1, 0]]
    init_exp = [0, 1] if init_mode == "quantized" else None
    len_dist = CategoricalDistribution({3: 0.5, 4: 0.5})
    return QuantizedHiddenMarkovModelDistribution(
        theta,
        levels,
        trans_exp,
        emis_exp,
        initial_exponents=init_exp,
        init_mode=init_mode,
        len_dist=len_dist,
        use_numba=use_numba,
    )


class QuantizedHmmParameterizationTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = make_quantized_dist()

    def test_rows_normalized(self):
        self.assertTrue(np.allclose(self.dist.transitions.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(self.dist.w.sum(), 1.0))
        for topic in self.dist.topics:
            self.assertAlmostEqual(sum(topic.pmap.values()), 1.0)

    def test_log_probs_have_quantized_form(self):
        # log p = k * log(theta) - log Z_row, with one shared log Z per row
        log_theta = self.dist.log_theta
        k = self.dist.transition_exponents
        for i in range(2):
            log_z = np.log(np.sum(self.dist.theta ** k[i, :]))
            expected = k[i, :] * log_theta - log_z
            self.assertTrue(np.allclose(self.dist.log_transitions[i, :], expected))

    def test_structural_zero_probability(self):
        dist = QuantizedHiddenMarkovModelDistribution(
            0.5,
            ["a", "b"],
            [[0, -1], [1, 0]],
            [[0, 1], [-1, 0]],
            initial_exponents=[0, 0],
            len_dist=CategoricalDistribution({2: 1.0}),
        )
        self.assertEqual(dist.transitions[0, 1], 0.0)
        self.assertEqual(dist.transitions[0, 0], 1.0)
        self.assertEqual(dist.topics[1].pmap["a"], 0.0)

    def test_invalid_args_raise(self):
        with self.assertRaises(ValueError):
            make_quantized_dist(theta=1.5)
        with self.assertRaises(ValueError):
            QuantizedHiddenMarkovModelDistribution(0.5, ["a"], [[-1]], [[0]], initial_exponents=[0])
        with self.assertRaises(ValueError):
            QuantizedHiddenMarkovModelDistribution(
                0.5, ["a", "b"], [[0, 0], [0, 0]], [[0, 0], [0, 0]], init_mode="quantized"
            )

    def test_stationary_init_mode(self):
        dist = make_quantized_dist(init_mode="stationary")
        self.assertIsNone(dist.initial_exponents)
        self.assertTrue(np.allclose(dist.w @ dist.transitions, dist.w))

    def test_matches_dense_hmm(self):
        data = self.dist.sampler(seed=3).sample(20)
        dense = HiddenMarkovModelDistribution(
            topics=list(self.dist.topics),
            w=self.dist.w.copy(),
            transitions=self.dist.transitions.copy(),
            len_dist=self.dist.len_dist,
        )
        for seq in data:
            self.assertAlmostEqual(self.dist.log_density(seq), dense.log_density(seq), places=10)

    def test_seq_log_density_matches_scalar(self):
        data = self.dist.sampler(seed=4).sample(25)
        enc = self.dist.dist_to_encoder().seq_encode(data)
        vec_ll = self.dist.seq_log_density(enc)
        scalar_ll = np.asarray([self.dist.log_density(seq) for seq in data])
        self.assertTrue(np.allclose(vec_ll, scalar_ll))

    def test_str_eval_round_trip(self):
        namespace = {
            "QuantizedHiddenMarkovModelDistribution": QuantizedHiddenMarkovModelDistribution,
            "CategoricalDistribution": CategoricalDistribution,
            "NullDistribution": NullDistribution,
        }
        dist2 = eval(str(self.dist), namespace)
        self.assertEqual(dist2.theta, self.dist.theta)
        self.assertTrue(np.array_equal(dist2.transition_exponents, self.dist.transition_exponents))
        self.assertTrue(np.array_equal(dist2.emission_exponents, self.dist.emission_exponents))
        self.assertTrue(np.allclose(dist2.w, self.dist.w))

    def test_sampler_output(self):
        data = self.dist.sampler(seed=1).sample(30)
        self.assertEqual(len(data), 30)
        for seq in data:
            self.assertIn(len(seq), (3, 4))
            for v in seq:
                self.assertIn(v, self.dist.levels)


class QuantizedHmmEnumeratorTestCase(unittest.TestCase):
    def setUp(self):
        self.dist = make_quantized_dist()

    def test_specialized_enumerator_matches_brute_force(self):
        self.assertIsInstance(self.dist.enumerator(), QuantizedHiddenMarkovModelEnumerator)

        support = [list(t) for n in (3, 4) for t in itertools.product(self.dist.levels, repeat=n)]
        brute = [(v, self.dist.log_density(v)) for v in support]
        brute = [(v, lp) for v, lp in brute if lp > -np.inf]
        brute.sort(key=lambda u: -u[1])

        items = list(self.dist.enumerator())
        self.assertEqual(len(items), len(brute))

        lps = [lp for _, lp in items]
        for i in range(len(lps) - 1):
            self.assertGreaterEqual(lps[i], lps[i + 1] - 1.0e-9)
        self.assertEqual(len({freeze(v) for v, _ in items}), len(items))

        np.testing.assert_allclose(lps, [lp for _, lp in brute], atol=1.0e-9)
        for v, lp in items:
            self.assertAlmostEqual(lp, self.dist.log_density(v), delta=1.0e-9)

        def tiers(pairs):
            out = {}
            for v, lp in pairs:
                out.setdefault(round(lp, 8), set()).add(freeze(v))
            return out

        self.assertEqual(tiers(items), tiers(brute))


class QuantizedHmmEstimationTestCase(unittest.TestCase):
    def setUp(self):
        self.gen = make_quantized_dist(theta=0.5)
        self.data = self.gen.sampler(seed=11).sample(150)

    def _fit(self, est, its=10):
        encoder = est.accumulator_factory().make().acc_to_encoder()
        enc_data = [(len(self.data), encoder.seq_encode(self.data))]
        model = seq_initialize(enc_data, est, RandomState(7), p=1.0)
        ll_first = None
        for _ in range(its):
            model = seq_estimate(enc_data, est, model)
            if ll_first is None:
                ll_first = np.sum(model.seq_log_density(encoder.seq_encode(self.data)))
        ll_last = np.sum(model.seq_log_density(encoder.seq_encode(self.data)))
        return model, ll_first, ll_last

    def test_seq_estimate_smoke(self):
        est = QuantizedHiddenMarkovEstimator(2, pseudo_count=0.5, k_max=12, len_estimator=CategoricalEstimator())
        model, ll_first, ll_last = self._fit(est)

        self.assertIsInstance(model, QuantizedHiddenMarkovModelDistribution)
        self.assertTrue(0.0 < model.theta < 1.0)
        self.assertTrue(np.issubdtype(model.transition_exponents.dtype, np.integer))
        self.assertTrue(np.issubdtype(model.emission_exponents.dtype, np.integer))
        # with a pseudo count there are no structural zeros, and k_max caps the exponents
        for exps in (model.transition_exponents, model.emission_exponents, model.initial_exponents):
            self.assertTrue(np.all(exps >= 0))
            self.assertTrue(np.all(exps <= 12))
        self.assertTrue(np.isfinite(ll_last))
        self.assertGreaterEqual(ll_last, ll_first - 1.0e-6)

    def test_fixed_theta(self):
        est = QuantizedHiddenMarkovEstimator(
            2, pseudo_count=0.5, k_max=12, fixed_theta=0.5, len_estimator=CategoricalEstimator()
        )
        model, _, ll_last = self._fit(est, its=5)
        self.assertEqual(model.theta, 0.5)
        self.assertTrue(np.isfinite(ll_last))

    def test_stationary_init_mode_fit(self):
        est = QuantizedHiddenMarkovEstimator(
            2, pseudo_count=0.5, k_max=12, init_mode="stationary", len_estimator=CategoricalEstimator()
        )
        model, _, ll_last = self._fit(est, its=5)
        self.assertIsNone(model.initial_exponents)
        self.assertTrue(np.allclose(model.w @ model.transitions, model.w))
        self.assertTrue(np.isfinite(ll_last))

    def test_structural_zero_for_unseen_level(self):
        est = QuantizedHiddenMarkovEstimator(
            2, levels=["a", "b", "c", "d"], pseudo_count=None, k_max=12, len_estimator=CategoricalEstimator()
        )
        model, _, _ = self._fit(est, its=3)
        d_col = model.levels.index("d")
        self.assertTrue(np.all(model.emission_exponents[:, d_col] == -1))
        for topic in model.topics:
            self.assertEqual(topic.pmap["d"], 0.0)

    def test_estimator_round_trip_from_distribution(self):
        est = self.gen.estimator(pseudo_count=0.5)
        self.assertIsInstance(est, QuantizedHiddenMarkovEstimator)
        self.assertEqual(est.num_states, 2)
        model, _, ll_last = self._fit(est, its=3)
        self.assertIsInstance(model, QuantizedHiddenMarkovModelDistribution)
        self.assertTrue(np.isfinite(ll_last))

    def test_split_collapsed_states_unit(self):
        # two exchangeable states whose raw counts say state 1 emits symbol 0 less often
        trans_exp = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        emit_exp = np.asarray([[0, 1], [0, 1]], dtype=np.int64)
        trans_counts = np.asarray([[10.0, 10.0], [10.0, 10.0]])
        emit_counts = np.asarray([[12.0, 8.0], [8.0, 12.0]])
        log_theta = np.log(0.5)

        n = _split_collapsed_states(trans_exp, emit_exp, trans_counts, emit_counts, 16, log_theta, np.log(2.0))
        self.assertEqual(n, 1)
        # state 1 should now be one ln(2) gap less likely on symbol 0 than state 0
        self.assertEqual(emit_exp[1, 0], emit_exp[0, 0] + 1)

    def test_split_escapes_collapsed_fixed_point(self):
        gen = QuantizedHiddenMarkovModelDistribution(
            0.5,
            ["a", "b", "c"],
            [[0, 2], [3, 0]],
            [[0, 2, 4], [4, 2, 0]],
            initial_exponents=[0, 1],
            len_dist=CategoricalDistribution({8: 0.5, 9: 0.5}),
        )
        # batched=False keeps the legacy per-draw RNG order so this EM-splitting
        # fixture is unaffected by the vectorized HMM state-path sampling (WS-K).
        data = gen.sampler(seed=2).sample(500, batched=False)

        lls = {}
        for split in (False, True):
            est = QuantizedHiddenMarkovEstimator(
                2, pseudo_count=0.25, k_max=16, len_estimator=CategoricalEstimator(), split_collapsed=split
            )
            encoder = est.accumulator_factory().make().acc_to_encoder()
            enc_data = [(len(data), encoder.seq_encode(data))]
            enc_eval = encoder.seq_encode(data)
            # this symmetric initialization rounds to a collapsed fixed point without splitting
            model = seq_initialize(enc_data, est, RandomState(3), p=1.0)
            model = seq_estimate(enc_data, est, model)
            ll_first = np.sum(model.seq_log_density(enc_eval))
            # 10 refit iterations (down from 24) still clear the +100 margin with ~4x headroom
            # (verified ll_last_on - ll_first_on / ll_last_off is ~440-450 nats at this count, vs. the
            # ~490 nats reached by 24 iterations); data size (500) is load-bearing here -- halving it to
            # 250 was verified to make the split never trigger at all, so it is left unchanged.
            for _ in range(10):
                model = seq_estimate(enc_data, est, model)
            lls[split] = (ll_first, np.sum(model.seq_log_density(enc_eval)))

        ll_first_off, ll_last_off = lls[False]
        ll_first_on, ll_last_on = lls[True]
        # without splitting EM is frozen at the collapsed fixed point
        self.assertAlmostEqual(ll_first_off, ll_last_off, delta=1.0)
        # with splitting it escapes and improves substantially
        self.assertGreater(ll_last_on, ll_first_on + 100.0)
        self.assertGreater(ll_last_on, ll_last_off + 100.0)

    def test_numba_encoder_parity(self):
        dist_nb = make_quantized_dist(use_numba=True)
        dist_np = make_quantized_dist(use_numba=False)
        data = dist_np.sampler(seed=5).sample(20)
        enc_nb = dist_nb.dist_to_encoder().seq_encode(data)
        enc_np = dist_np.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.allclose(dist_nb.seq_log_density(enc_nb), dist_np.seq_log_density(enc_np)))


class QuantizedHmmEngineTestCase(unittest.TestCase):
    """QuantizedHMM inherits the HMM engine scoring + engine-resident E-step (numpy + torch)."""

    def setUp(self):
        self.dist = QuantizedHiddenMarkovModelDistribution(
            0.5,
            ["a", "b", "c"],
            [[0, 2], [3, 0]],
            [[0, 2, 4], [4, 2, 0]],
            initial_exponents=[0, 1],
            len_dist=CategoricalDistribution({4: 0.5, 5: 0.5}),
            use_numba=False,
        )
        self.data = self.dist.sampler(seed=2).sample(40)
        self.engines = [("numpy", NUMPY_ENGINE)]
        if _TORCH is not None:
            self.engines.append(("torch", _TORCH))

    def test_scoring_and_estep_parity(self):
        from mixle.stats.compute.backend import backend_seq_log_density

        est = self.dist.estimator(pseudo_count=0.5)
        enc = self.dist.dist_to_encoder().seq_encode(self.data)
        ref = np.asarray(self.dist.seq_log_density(enc))
        host = est.accumulator_factory().make()
        host.seq_update(enc, np.ones(len(self.data)), self.dist)
        host_value = host.value()
        for name, engine in self.engines:
            with self.subTest(engine=name):
                self.assertTrue(self.dist.supports_engine(engine))
                got = np.asarray(engine.to_numpy(backend_seq_log_density(self.dist, enc, engine)))
                self.assertTrue(np.allclose(got, ref, atol=1.0e-9))
                kernel = self.dist.kernel(engine=engine, estimator=est)
                self.assertEqual(type(kernel).__name__, "HiddenMarkovModelKernel")
                value = kernel.accumulate(enc, np.ones(len(self.data)))
                for k in (1, 2, 3):
                    self.assertTrue(np.allclose(np.asarray(host_value[k]), np.asarray(value[k]), atol=1.0e-8))


if __name__ == "__main__":
    unittest.main()
