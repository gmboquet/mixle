"""Nested scalar-tree kernels: reduced-precision compute and the chunk-parallel variant.

fused_nested previously ignored ``compute_dtype`` (always float64) and had no prange variant --
both closed here with fused_codegen's exact contracts: float32 runs row arithmetic reduced while
every accumulator stays float64, and the parallel kernels are bit-stable across reruns and worker
counts (fixed chunking, fixed-order combine), agreeing with the sequential kernels to float
re-association tolerance.
"""

import unittest

import numpy as np

from mixle.stats import GaussianDistribution, MixtureDistribution
from mixle.stats.compute import fused_nested as fn
from mixle.utils.optional_deps import HAS_NUMBA


def _nested_model_and_enc(n=50_000, seed=0):
    rng = np.random.RandomState(seed)
    inner = lambda s: MixtureDistribution(  # noqa: E731
        [GaussianDistribution(-3.0 + s, 1.0), GaussianDistribution(3.0 + s, 1.0 + 0.2 * s)], [0.6, 0.4]
    )
    model = MixtureDistribution([inner(0), inner(1), inner(2)], [0.5, 0.3, 0.2])
    data = [float(v) for v in rng.randn(n) * 3]
    return model, model.dist_to_encoder().seq_encode(data), n


def _flatten(suff):
    out = []

    def walk(v):
        if isinstance(v, (tuple, list)):
            for piece in v:
                walk(piece)
        elif v is not None:
            out.extend(np.asarray(v, dtype=np.float64).ravel().tolist())

    walk(suff)
    return np.asarray(out)


@unittest.skipUnless(HAS_NUMBA, "nested fused kernels require numba")
class NestedComputeDtypeTest(unittest.TestCase):
    def test_float32_rows_track_float64_within_reduced_precision(self):
        model, enc, _ = _nested_model_and_enc(n=20_000)
        full = fn.fused_nested_seq_log_density(model, enc)
        reduced = fn.fused_nested_seq_log_density(model, enc, compute_dtype=np.float32)
        np.testing.assert_allclose(reduced, full, rtol=2e-5, atol=2e-5)
        self.assertEqual(reduced.dtype, np.float64, "accumulation and output stay float64")

    def test_non_float32_reduced_dtypes_are_refused(self):
        model, enc, _ = _nested_model_and_enc(n=200)
        with self.assertRaises(ValueError):
            fn.fused_nested_seq_log_density(model, enc, compute_dtype=np.float16)


@unittest.skipUnless(HAS_NUMBA, "nested fused kernels require numba")
class NestedQuantizedLseTest(unittest.TestCase):
    def test_quantized_scorer_stays_within_the_depth_compounded_bound(self):
        from mixle.engines.qlut import lse_error_bound
        from mixle.stats.compute.fused_nested import _mixture_depth, analyze_nested

        model, enc, _ = _nested_model_and_enc(n=15_000)
        root, _ctx = analyze_nested(model)
        depth = _mixture_depth(root)
        self.assertGreaterEqual(depth, 2, "the fixture must be genuinely nested")
        exact = fn.fused_nested_seq_log_density(model, enc)
        for bits in (8, 12):
            quant = fn.fused_nested_seq_log_density(model, enc, lse_bits=bits)
            bound = depth * lse_error_bound(bits, 24.0)
            self.assertLessEqual(float(np.abs(quant - exact).max()), bound, f"bits={bits}")
        par1 = fn.fused_nested_seq_log_density(model, enc, parallel=True, lse_bits=12)
        par2 = fn.fused_nested_seq_log_density(model, enc, parallel=True, lse_bits=12)
        self.assertTrue(np.array_equal(par1, par2), "quantized x parallel keeps bit-stable reruns")

    def test_template_entry_point_forwards_to_nested_trees(self):
        from mixle.engines.qlut import lse_error_bound
        from mixle.stats.compute import fused_codegen as fc

        model, enc, _ = _nested_model_and_enc(n=4_000)
        exact = fc.fused_seq_log_density(model, enc)
        quant = fc.fused_seq_log_density(model, enc, lse_bits=12)
        self.assertLessEqual(float(np.abs(quant - exact).max()), 2 * lse_error_bound(12, 24.0))

    def test_validation(self):
        model, enc, _ = _nested_model_and_enc(n=200)
        with self.assertRaises(ValueError):
            fn.fused_nested_seq_log_density(model, enc, lse_bits=0)


@unittest.skipUnless(HAS_NUMBA, "nested fused kernels require numba")
class NestedParallelTest(unittest.TestCase):
    def test_parallel_scorer_matches_sequential_and_is_bit_stable(self):
        model, enc, _ = _nested_model_and_enc()
        seq = fn.fused_nested_seq_log_density(model, enc, parallel=False)
        par = fn.fused_nested_seq_log_density(model, enc, parallel=True)
        rerun = fn.fused_nested_seq_log_density(model, enc, parallel=True)
        np.testing.assert_allclose(par, seq, rtol=1e-12, atol=1e-12)
        self.assertTrue(np.array_equal(par, rerun))

    def test_parallel_estep_matches_sequential_and_is_bit_stable(self):
        model, enc, n = _nested_model_and_enc()
        w = np.ones(n)
        suff_seq, ll_seq = fn.fused_nested_accumulate(model, enc, w, return_ll=True, parallel=False)
        suff_par, ll_par = fn.fused_nested_accumulate(model, enc, w, return_ll=True, parallel=True)
        suff_rerun, ll_rerun = fn.fused_nested_accumulate(model, enc, w, return_ll=True, parallel=True)
        self.assertAlmostEqual(ll_seq, ll_par, delta=abs(ll_seq) * 1e-9)
        np.testing.assert_allclose(_flatten(suff_par), _flatten(suff_seq), rtol=1e-8, atol=1e-10)
        self.assertEqual(ll_par, ll_rerun)
        self.assertTrue(np.array_equal(_flatten(suff_par), _flatten(suff_rerun)))

    def test_parallel_estep_is_bit_identical_across_worker_counts(self):
        import numba

        model, enc, n = _nested_model_and_enc(n=30_000)
        w = np.ones(n)
        before = numba.get_num_threads()
        try:
            numba.set_num_threads(max(2, before))
            suff_many, ll_many = fn.fused_nested_accumulate(model, enc, w, return_ll=True, parallel=True)
            numba.set_num_threads(1)
            suff_one, ll_one = fn.fused_nested_accumulate(model, enc, w, return_ll=True, parallel=True)
        finally:
            numba.set_num_threads(before)
        self.assertEqual(ll_many, ll_one)
        self.assertTrue(np.array_equal(_flatten(suff_many), _flatten(suff_one)))


if __name__ == "__main__":
    unittest.main()
