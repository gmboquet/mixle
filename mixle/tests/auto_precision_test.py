"""Tests for the data-aware precision policy (mixle.engines.auto_precision)."""

import unittest

import numpy as np

from mixle.engines import NUMPY_ENGINE, TorchEngine, auto_precision, torch
from mixle.engines.precision import _numeric_data_sample


class _FakeGPUEngine:
    """Minimal stand-in for a Torch engine placed on a GPU (no real device required)."""

    name = "torch"
    device = "cuda:0"


class AutoPrecisionTestCase(unittest.TestCase):
    def test_cpu_and_numpy_always_float64(self):
        # float32 is a no-op (or slower) off a GPU torch engine -> always float64.
        well_conditioned = list(np.random.RandomState(0).randn(1000))
        self.assertEqual(auto_precision(well_conditioned, engine=None), "float64")
        self.assertEqual(auto_precision(well_conditioned, engine=NUMPY_ENGINE), "float64")
        cpu_torch = TorchEngine(device="cpu", dtype="float32") if torch is not None else None
        if cpu_torch is not None:
            self.assertEqual(auto_precision(well_conditioned, engine=cpu_torch), "float64")

    def test_gpu_well_conditioned_picks_float32(self):
        data = list(np.random.RandomState(1).randn(2000) * 2.0 + 1.0)
        self.assertEqual(auto_precision(data, engine=_FakeGPUEngine()), "float32")

    def test_gpu_large_magnitude_falls_back_to_float64(self):
        data = list(np.random.RandomState(2).randn(2000) + 1.0e6)  # huge magnitude
        self.assertEqual(auto_precision(data, engine=_FakeGPUEngine()), "float64")

    def test_gpu_wide_dynamic_range_falls_back(self):
        data = list(np.random.RandomState(3).randn(2000) * 1.0e-3 + 50.0)  # amax/spread large
        self.assertEqual(auto_precision(data, engine=_FakeGPUEngine()), "float64")

    def test_gpu_non_numeric_data_falls_back(self):
        docs = [["a", "b", "c"], ["b", "c"], ["a"]]  # categorical/structured -> no numeric sample
        self.assertEqual(auto_precision(docs, engine=_FakeGPUEngine()), "float64")
        self.assertEqual(auto_precision(None, engine=_FakeGPUEngine()), "float64")

    def test_gpu_tail_concentrated_extreme_values_are_caught(self):
        # Regression: the data sample used to be a plain leading prefix (data[:sample_size]) -- a
        # naturally-ordered dataset (sorted, appended-to over time, grouped by source) that stashes
        # extreme values later in the sequence was invisible to the magnitude guard, and float32 got
        # recommended for data that is not actually well-conditioned for it. Must stride the full
        # dataset instead.
        well_conditioned = list(np.random.RandomState(4).randn(600) * 2.0)
        extreme_tail = [1.0e9] * 50  # only appears after the default sample_size=512 prefix
        data = well_conditioned + extreme_tail
        self.assertEqual(auto_precision(data, engine=_FakeGPUEngine()), "float64")

    def test_optimize_precision_auto_matches_default_on_cpu(self):
        # 'auto' on CPU resolves to float64 (keeps the default host path) -> identical fit.
        import io

        from mixle.inference.estimation import optimize
        from mixle.stats import GaussianDistribution, GaussianEstimator, MixtureDistribution, MixtureEstimator

        truth = MixtureDistribution(
            [GaussianDistribution(-3.0, 1.0), GaussianDistribution(0.0, 1.0), GaussianDistribution(4.0, 1.0)],
            [0.4, 0.3, 0.3],
        )
        data = truth.sampler(1).sample(6000)

        def mk():
            return MixtureEstimator([GaussianEstimator()] * 3)

        d = optimize(data, mk(), max_its=12, rng=np.random.RandomState(1), out=io.StringIO())
        a = optimize(data, mk(), max_its=12, rng=np.random.RandomState(1), out=io.StringIO(), precision="auto")
        self.assertTrue(np.allclose(d.w, a.w, atol=1.0e-12))

    def test_plan_precision_auto_resolves(self):
        from mixle.stats import GaussianDistribution, MixtureDistribution
        from mixle.utils.parallel.planner import plan

        truth = MixtureDistribution([GaussianDistribution(-3.0, 1.0), GaussianDistribution(3.0, 1.0)], [0.5, 0.5])
        data = truth.sampler(1).sample(2000)
        p = plan(data=data, model=truth, precision="auto")
        # CPU planning -> float64 sizing.
        self.assertEqual(p.dtype_bytes, 8)

    def test_numeric_sample_extraction(self):
        self.assertIsNone(_numeric_data_sample(None))
        self.assertIsNone(_numeric_data_sample(["x", "y"]))
        self.assertTrue(np.allclose(np.sort(_numeric_data_sample([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0]))
        # tuples / composite records flatten to their numeric fields
        s = _numeric_data_sample([(1.0, 2), (3.0, 4)])
        self.assertTrue(np.allclose(np.sort(s), [1.0, 2.0, 3.0, 4.0]))
        # vector observations
        s = _numeric_data_sample([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        self.assertEqual(s.size, 4)

    def test_numeric_sample_strides_across_the_whole_dataset(self):
        well_conditioned = list(np.random.RandomState(5).randn(600))
        extreme_tail = [1.0e9] * 50
        s = _numeric_data_sample(well_conditioned + extreme_tail, sample_size=512)
        self.assertTrue(np.any(np.abs(s) > 1.0e6))  # the tail must show up in the sample


class PrecisionNameHygieneTestCase(unittest.TestCase):
    """Sub-byte / FP8 / microscaling / codebook names are rejected with an actionable error rather
    than a cryptic numpy TypeError -- there is no native CPU arithmetic for them (numba cannot compile
    below float32). Supported compute precisions are float32 (reduced) and float64 (default)."""

    def test_supported_compute_precisions_normalize(self):
        from mixle.engines.precision import normalize_numpy_dtype

        self.assertIsNone(normalize_numpy_dtype(None))
        self.assertEqual(normalize_numpy_dtype("float32"), np.dtype(np.float32))
        self.assertEqual(normalize_numpy_dtype("fp32"), np.dtype(np.float32))
        self.assertEqual(normalize_numpy_dtype("float64"), np.dtype(np.float64))
        self.assertEqual(normalize_numpy_dtype(np.float32), np.dtype(np.float32))

    def test_subbyte_and_quant_formats_are_rejected(self):
        from mixle.engines.precision import normalize_numpy_dtype

        for name in ("fp8", "e4m3", "e5m2", "fp6", "fp4", "e2m1", "float4", "float2", "float3", "mxfp4", "nf4", "int8"):
            with self.assertRaises(ValueError) as ctx:
                normalize_numpy_dtype(name)
            self.assertIn("float32", str(ctx.exception))  # the message must point at what IS supported

    def test_bfloat16_rejected_on_numpy(self):
        from mixle.engines.precision import normalize_numpy_dtype

        with self.assertRaises(ValueError):
            normalize_numpy_dtype("bf16")


if __name__ == "__main__":
    unittest.main()


try:
    from mixle.utils.optional_deps import HAS_NUMBA
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


class MinimalPrecisionColdStartTestCase(unittest.TestCase):
    """optimize(precision="minimal") must plan against a REAL model on cold starts.

    Planning used to run against ``prev_estimate`` before initialization, so every cold-start fit
    hit the planner's no-model branch and silently allocated float64; the fix defers planning until
    the initialized model exists, and the decision is disclosed on the estimator and the ``out``
    stream either way.
    """

    def _fixture(self, offset=0.0, n=4000):
        from mixle.stats import GaussianEstimator, MixtureEstimator

        rng = np.random.RandomState(0)
        data = np.concatenate([rng.normal(offset + 6.0 * c, 1.0, n // 4) for c in range(4)])
        rng.shuffle(data)
        estimator = MixtureEstimator([GaussianEstimator() for _ in range(4)])
        return data, estimator

    @unittest.skipUnless(HAS_NUMBA, "the fp32 plan requires the fused numba kernel")
    def test_cold_start_selects_float32_on_a_safe_mixture(self):
        from mixle.inference.estimation import optimize

        data, estimator = self._fixture()
        fitted = optimize(data, estimator, precision="minimal", max_its=4, print_iter=0, delta=None)

        plan = estimator.last_precision_plan
        self.assertEqual(np.dtype(plan.compute_dtype), np.dtype(np.float32))
        self.assertTrue(plan.reduced())
        enc = fitted.dist_to_encoder().seq_encode(data)
        self.assertTrue(np.isfinite(float(np.sum(fitted.seq_log_density(enc)))))

    def test_danger_zone_magnitude_stays_float64(self):
        from mixle.inference.estimation import optimize

        data, estimator = self._fixture(offset=2.0e6)  # |x| > 1e6: outside the validated fp32 band
        optimize(data, estimator, precision="minimal", max_its=2, print_iter=0, delta=None)

        plan = estimator.last_precision_plan
        self.assertEqual(np.dtype(plan.compute_dtype), np.dtype(np.float64))
        self.assertFalse(plan.reduced())

    @unittest.skipUnless(HAS_NUMBA, "the fp32 plan requires the fused numba kernel")
    def test_warm_start_records_the_plan_too(self):
        from mixle.inference.estimation import optimize
        from mixle.stats import GaussianDistribution, MixtureDistribution

        data, estimator = self._fixture()
        proto = MixtureDistribution([GaussianDistribution(6.0 * c, 2.0) for c in range(4)], [0.25] * 4)
        optimize(data, estimator, precision="minimal", prev_estimate=proto, max_its=2, print_iter=0, delta=None)

        plan = estimator.last_precision_plan
        self.assertEqual(np.dtype(plan.compute_dtype), np.dtype(np.float32))

    @unittest.skipUnless(HAS_NUMBA, "the fp32 plan requires the fused numba kernel")
    def test_out_stream_discloses_the_allocation(self):
        import io

        from mixle.inference.estimation import optimize

        data, estimator = self._fixture()
        buf = io.StringIO()
        optimize(data, estimator, precision="minimal", max_its=2, print_iter=0, delta=None, out=buf)
        self.assertIn("precision=minimal: float32", buf.getvalue())
