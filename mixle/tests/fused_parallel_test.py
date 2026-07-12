"""Chunk-parallel (prange) fused kernels: determinism receipts and parity vs the sequential kernels.

The design claim under test (mixle/stats/compute/fused_codegen.py): chunk boundaries are a pure
function of n -- never of the worker count -- and per-chunk partials combine in fixed order, so
parallel results are BIT-IDENTICAL across reruns and across ``numba.set_num_threads`` settings.
Against the sequential kernel the guarantee is deliberately weaker and stated as such: 1-2 ULP on the
scorer (different fastmath binaries vectorize differently) and ~1e-8-relative on the E-step
reductions (chunk-boundary float re-association).
"""

import unittest

import numpy as np

from mixle.utils.optional_deps import HAS_NUMBA

if HAS_NUMBA:
    import numba

from mixle.stats import (
    CategoricalDistribution,
    CompositeDistribution,
    ExponentialDistribution,
    GaussianDistribution,
    MixtureDistribution,
    MultivariateGaussianDistribution,
    PoissonDistribution,
)
from mixle.stats.compute import fused_codegen as fc


def _mixed_model_and_enc(n=60_000, seed=0):
    rng = np.random.RandomState(seed)
    comps = [
        CompositeDistribution(
            (
                GaussianDistribution(float(m), 1.0 + 0.1 * j),
                PoissonDistribution(2.0 + j),
                ExponentialDistribution(1.0 + 0.2 * j),
                CategoricalDistribution({"a": 0.5, "b": 0.3, "c": 0.2}),
            )
        )
        for j, m in enumerate((-3.0, 0.0, 3.0))
    ]
    model = MixtureDistribution(comps, [0.5, 0.3, 0.2])
    data = [
        (
            float(rng.randn() + (i % 3)),
            int(rng.poisson(3)),
            float(rng.exponential(1.2)),
            ("a", "b", "c")[int(rng.randint(3))],
        )
        for i in range(n)
    ]
    return model, model.dist_to_encoder().seq_encode(data), n


def _flatten_stats(suff):
    """Every scalar in a suff-stat tree, in deterministic order (dict-shaped stats sort by key)."""
    out: list[float] = []

    def walk(v):
        if isinstance(v, dict):
            for key in sorted(v):
                walk(v[key])
        elif isinstance(v, (tuple, list)):
            for piece in v:
                walk(piece)
        elif v is None:
            pass
        else:
            out.extend(np.asarray(v, dtype=np.float64).ravel().tolist())

    walk(suff)
    return np.asarray(out)


@unittest.skipUnless(HAS_NUMBA, "parallel fused kernels require numba")
class ParallelScorerTest(unittest.TestCase):
    def test_parallel_scorer_matches_sequential_to_ulp_and_is_bit_stable(self):
        model, enc, _ = _mixed_model_and_enc()
        seq = fc.fused_seq_log_density(model, enc, parallel=False)
        par = fc.fused_seq_log_density(model, enc, parallel=True)
        rerun = fc.fused_seq_log_density(model, enc, parallel=True)
        np.testing.assert_allclose(par, seq, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.array_equal(par, rerun), "parallel scorer must be bit-identical across reruns")

    def test_parallel_scorer_is_bit_identical_across_worker_counts(self):
        model, enc, _ = _mixed_model_and_enc()
        before = numba.get_num_threads()
        try:
            numba.set_num_threads(max(2, before))
            many = fc.fused_seq_log_density(model, enc, parallel=True)
            numba.set_num_threads(1)
            one = fc.fused_seq_log_density(model, enc, parallel=True)
        finally:
            numba.set_num_threads(before)
        self.assertTrue(
            np.array_equal(many, one),
            "chunk boundaries depend only on n, so the worker count must not change a single bit",
        )


@unittest.skipUnless(HAS_NUMBA, "parallel fused kernels require numba")
class ParallelEstepTest(unittest.TestCase):
    def test_parallel_estep_matches_sequential_and_is_bit_stable(self):
        model, enc, n = _mixed_model_and_enc()
        w = np.ones(n)
        suff_seq, ll_seq = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=False)
        suff_par, ll_par = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
        suff_rerun, ll_rerun = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
        self.assertAlmostEqual(ll_seq, ll_par, delta=abs(ll_seq) * 1e-9)
        np.testing.assert_allclose(_flatten_stats(suff_par), _flatten_stats(suff_seq), rtol=1e-8, atol=1e-10)
        self.assertEqual(ll_par, ll_rerun, "parallel E-step log-likelihood must be bit-identical across reruns")
        self.assertTrue(np.array_equal(_flatten_stats(suff_par), _flatten_stats(suff_rerun)))

    def test_parallel_estep_is_bit_identical_across_worker_counts(self):
        model, enc, n = _mixed_model_and_enc()
        w = np.ones(n)
        before = numba.get_num_threads()
        try:
            numba.set_num_threads(max(2, before))
            suff_many, ll_many = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
            numba.set_num_threads(1)
            suff_one, ll_one = fc.fused_accumulate(model, enc, w, return_ll=True, parallel=True)
        finally:
            numba.set_num_threads(before)
        self.assertEqual(ll_many, ll_one)
        self.assertTrue(np.array_equal(_flatten_stats(suff_many), _flatten_stats(suff_one)))

    def test_matrix_leaf_takes_the_sequential_blas_post_pass_in_both_variants(self):
        rng = np.random.RandomState(1)
        comps = [
            MultivariateGaussianDistribution(mu=np.full(3, float(m)), covar=np.eye(3) * (1.0 + 0.2 * j))
            for j, m in enumerate((-2.0, 2.0))
        ]
        model = MixtureDistribution(comps, [0.5, 0.5])
        data = [rng.randn(3) + (2.0 if i % 2 else -2.0) for i in range(30_000)]
        enc = model.dist_to_encoder().seq_encode(data)
        w = np.ones(len(data))
        suff_seq = fc.fused_accumulate(model, enc, w, parallel=False)
        suff_par = fc.fused_accumulate(model, enc, w, parallel=True)
        np.testing.assert_allclose(_flatten_stats(suff_par), _flatten_stats(suff_seq), rtol=1e-8, atol=1e-10)


@unittest.skipUnless(HAS_NUMBA, "parallel fused kernels require numba")
class QuantizedLseTest(unittest.TestCase):
    def test_quantized_scorer_stays_within_the_qlut_bound(self):
        from mixle.engines.qlut import lse_error_bound

        model, enc, _ = _mixed_model_and_enc(n=20_000)
        exact = fc.fused_seq_log_density(model, enc)
        for bits in (8, 12):
            quant = fc.fused_seq_log_density(model, enc, lse_bits=bits)
            bound = lse_error_bound(bits, 24.0)
            self.assertLessEqual(float(np.abs(quant - exact).max()), bound, f"bits={bits}")
        # parallel x quantized composes, with the same bit-stability contract
        par1 = fc.fused_seq_log_density(model, enc, parallel=True, lse_bits=12)
        par2 = fc.fused_seq_log_density(model, enc, parallel=True, lse_bits=12)
        self.assertTrue(np.array_equal(par1, par2))
        self.assertLessEqual(float(np.abs(par1 - exact).max()), lse_error_bound(12, 24.0))

    def test_quantized_lse_is_opt_in_and_guarded(self):
        model, enc, _ = _mixed_model_and_enc(n=500)
        with self.assertRaises(ValueError):
            fc.fused_seq_log_density(model, enc, lse_bits=0)
        with self.assertRaises(ValueError):
            fc.fused_seq_log_density(model, enc, lse_bits=12, lse_span=-1.0)
        from mixle.stats import GaussianDistribution as G

        nested = MixtureDistribution(
            [MixtureDistribution([G(-2.0, 1.0), G(2.0, 1.0)], [0.5, 0.5]) for _ in range(2)], [0.5, 0.5]
        )
        nenc = nested.dist_to_encoder().seq_encode([0.1, -0.4, 2.2])
        with self.assertRaises(NotImplementedError):
            fc.fused_seq_log_density(nested, nenc, lse_bits=12)


@unittest.skipUnless(HAS_NUMBA, "parallel fused kernels require numba")
class AutoPolicyTest(unittest.TestCase):
    def test_small_inputs_stay_sequential_and_large_engage_parallel(self):
        model, enc, n = _mixed_model_and_enc(n=4_000)
        plan = fc.analyze(model)
        fc._ESTEP_COMPILED.pop((plan.signature, True), None)
        fc.fused_accumulate(model, enc, np.ones(n))  # parallel=None: n is far below _PARALLEL_MIN_OBS
        self.assertNotIn((plan.signature, True), fc._ESTEP_COMPILED, "small n must not take the parallel kernel")

        old_min = fc._PARALLEL_MIN_OBS
        fc._PARALLEL_MIN_OBS = 1_000
        try:
            if numba.get_num_threads() > 1:
                fc.fused_accumulate(model, enc, np.ones(n))
                self.assertIn((plan.signature, True), fc._ESTEP_COMPILED, "auto policy must engage above the floor")
        finally:
            fc._PARALLEL_MIN_OBS = old_min

    def test_chunk_count_is_a_pure_function_of_n(self):
        self.assertEqual(fc._n_chunks(10), 1)
        self.assertEqual(fc._n_chunks(fc._PARALLEL_CHUNK_TARGET * 7), 7)
        self.assertEqual(fc._n_chunks(10**9), fc._PARALLEL_MAX_CHUNKS)


if __name__ == "__main__":
    unittest.main()
