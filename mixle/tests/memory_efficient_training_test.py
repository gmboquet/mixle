"""Acceptance tests for mixle.models.memory_efficient_training (roadmap F6: memory efficiency for
training state).

Each test class targets one line of the F6 acceptance criteria directly:

1. ``LossParityTest`` -- train the same small model for the same steps on the same data with
   plain fp32 Adam vs. ``CompressedAdam`` (G4/int8-compressed moments); final loss within a stated
   tolerance.
2. ``MeasuredMemoryCutTest`` -- a real byte-count comparison of compressed vs. fp32 optimizer
   state for a realistic (Adam-second-moment-like) tensor.
3. ``AdversarialFallbackTest`` -- both compression paths (G4 and int8) correctly fall back to
   dense storage on tensors that are genuinely hostile to them.
4. ``SelectiveRecomputePolicyTest`` -- the cost model recommends recompute for a memory-heavy/
   cheap-to-recompute block and NOT for a memory-light/expensive-to-recompute block (a real
   decision-boundary test), plus fp8 hardening's overflow/underflow/fallback guards.
"""

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mixle.models.memory_efficient_training import (  # noqa: E402
    CompressedAdam,
    CompressedMomentEncoding,
    CompressedOptimizerState,
    RecomputeDecision,
    SelectiveRecomputePolicy,
    compress_moment,
    decompress_moment,
    dequantize_int8_blockwise,
    fp8_cast_with_guard,
    quantize_int8_blockwise,
)
from mixle.models.transformer import build_causal_lm  # noqa: E402


def _adam_second_moment_like(rng: np.random.RandomState, n: int, steps: int = 50, beta2: float = 0.98) -> np.ndarray:
    """Same synthetic Adam second-moment construction as G4's own acceptance test
    (mixle.tests.sorted_profile_quantizer_test) -- an EMA of squared heavy-tailed gradients, strictly
    positive and bounded away from zero, matching a real optimizer buffer far better than a single
    squared-Gaussian draw."""
    v = np.zeros(n)
    for _ in range(steps):
        g = rng.normal(0.0, 1.0, size=n) * (1.0 + 0.15 * np.abs(rng.standard_t(df=4, size=n)))
        v = beta2 * v + (1.0 - beta2) * g * g
    return v


class LossParityTest(unittest.TestCase):
    """Train a tiny CausalLM two ways -- plain fp32 AdamW vs. CompressedAdam -- for the SAME steps
    on the SAME data, and confirm final loss is within a stated tolerance."""

    def _train(self, optimizer_factory, steps: int = 40):
        torch.manual_seed(0)
        model = build_causal_lm(vocab=13, d_model=16, n_layer=2, n_head=2, block=8)
        opt = optimizer_factory(model.parameters())

        gen = torch.Generator().manual_seed(1234)
        xs = [torch.randint(0, 13, (4, 8), generator=gen).float() for _ in range(steps)]
        ys = [torch.randint(0, 13, (4,), generator=gen) for _ in range(steps)]

        model.train()
        losses = []
        for x, y in zip(xs, ys):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        return losses

    def test_compressed_adam_matches_fp32_adam_loss(self):
        # torch.optim.Adam (not AdamW): CompressedAdam implements plain Adam (no decoupled weight
        # decay), so the fp32 baseline must be the SAME algorithm -- AdamW's default
        # weight_decay=0.01 would make this an apples-to-oranges comparison unrelated to
        # compression.
        fp32_losses = self._train(lambda params: torch.optim.Adam(params, lr=3e-3))
        compressed_losses = self._train(
            lambda params: CompressedAdam(params, lr=3e-3, compression_method="auto", min_size_for_g4=32)
        )

        fp32_final = fp32_losses[-1]
        compressed_final = compressed_losses[-1]
        rel_diff = abs(compressed_final - fp32_final) / max(abs(fp32_final), 1e-8)
        # Loose but real: compressed moments are lossy (int8 for these small per-layer tensors,
        # since min_size_for_g4=32 keeps most of them under G4's own G4-worth-it floor), so exact
        # match is not expected -- a 25% relative final-loss tolerance is the stated bound.
        self.assertLess(
            rel_diff,
            0.25,
            f"fp32 final loss={fp32_final:.4f}, compressed final loss={compressed_final:.4f}, rel_diff={rel_diff:.3f}",
        )
        # Both runs should actually be learning (loss decreasing), not just coincidentally close.
        self.assertLess(compressed_final, compressed_losses[0])
        self.assertLess(fp32_final, fp32_losses[0])

    def test_dense_compression_state_round_trip_is_lossless(self):
        """compression_method='dense' takes no lossy shortcut at all: a CompressedOptimizerState
        set with real (Adam-second-moment-like) m/v tensors and immediately read back should
        recover them to float32 precision (~1e-6 relative), not the ~1e-2-scale error int8/G4
        introduce (see MeasuredMemoryCutTest). This is checked at the STATE level (compress then
        immediately decompress) rather than through a multi-step training loop: Adam's own update
        is chaotically sensitive to ANY floating-point perturbation once a parameter's v_hat
        approaches eps^2 scale (empirically confirmed independently of this module -- even
        torch.optim.Adam's different-but-mathematically-equivalent internal operation order
        diverges measurably from a manual same-formula loop after a few steps), so a training-loop
        comparison would be testing float64<->float32 rounding noise amplified by chaos, not
        whether dense compression itself is lossy.
        """
        rng = np.random.RandomState(4)
        shape = (24,)  # matches a real Block's qkv bias shape (3 * d_model for d_model=8)
        m = torch.from_numpy(rng.normal(0.0, 0.1, size=shape)).float()
        v = torch.from_numpy(_adam_second_moment_like(rng, shape[0])).float()

        state = CompressedOptimizerState(shape, method="dense")
        state.set(m, v)
        m_back, v_back = state.get(dtype=torch.float32)

        np.testing.assert_allclose(m_back.numpy(), m.numpy(), rtol=1e-6, atol=1e-8)
        # v round-trips through CompressedOptimizerState's sqrt(v)-then-square transform (see that
        # class's docstring): still float32-precision lossless, just not literally bit-identical
        # to the untransformed input at the very last ULP.
        np.testing.assert_allclose(v_back.numpy(), v.numpy(), rtol=1e-5, atol=1e-8)

    def test_dense_compression_end_to_end_tracks_fp32_adam_closely(self):
        """A short (5-step) end-to-end sanity check that CompressedAdam(dense) tracks a real
        fp32 Adam training run closely -- not bit-exact (see the state-level test above for why),
        but far tighter than the lossy int8/G4 paths get in LossParityTest (25% final-loss
        tolerance): dense compression should track to within a few percent of fp32 Adam's loss at
        every step, not just the final one.
        """
        fp32_losses = self._train(lambda params: torch.optim.Adam(params, lr=3e-3), steps=5)
        dense_losses = self._train(lambda params: CompressedAdam(params, lr=3e-3, compression_method="dense"), steps=5)
        for i, (fp32_loss, dense_loss) in enumerate(zip(fp32_losses, dense_losses)):
            rel_diff = abs(dense_loss - fp32_loss) / max(abs(fp32_loss), 1e-8)
            self.assertLess(rel_diff, 0.05, f"step {i}: fp32={fp32_loss:.4f} dense={dense_loss:.4f}")


class MeasuredMemoryCutTest(unittest.TestCase):
    """Real byte-count comparison, compressed vs. fp32, for a realistic optimizer-state tensor."""

    def test_g4_and_int8_both_cut_memory_vs_fp32(self):
        rng = np.random.RandomState(0)
        n = 16384  # < 2**16 so G4's permutation indices fit in uint16, per its own module docstring
        v = _adam_second_moment_like(rng, n)
        fp32_bytes = v.astype(np.float32).nbytes
        self.assertEqual(fp32_bytes, n * 4)

        g4_encoding = compress_moment(v, method="g4", min_size_for_g4=1)
        self.assertEqual(g4_encoding.method, "g4")
        g4_bytes = g4_encoding.nbytes()
        self.assertLess(g4_bytes, fp32_bytes)
        g4_ratio = fp32_bytes / g4_bytes
        self.assertGreater(g4_ratio, 1.5)

        int8_encoding = compress_moment(v, method="int8")
        self.assertEqual(int8_encoding.method, "int8")
        int8_bytes = int8_encoding.nbytes()
        self.assertLess(int8_bytes, fp32_bytes)
        int8_ratio = fp32_bytes / int8_bytes
        self.assertGreater(int8_ratio, 3.5)  # ~4x: int8 codes + small per-block scale overhead

        # Reconstruction stays close for both -- a real, measured relative error, not just a size claim.
        g4_recon = decompress_moment(g4_encoding)
        int8_recon = decompress_moment(int8_encoding)
        g4_rel_err = np.linalg.norm(g4_recon - v) / np.linalg.norm(v)
        int8_rel_err = np.linalg.norm(int8_recon - v) / np.linalg.norm(v)
        self.assertLess(g4_rel_err, 0.15)
        self.assertLess(int8_rel_err, 0.15)

    def test_compressed_optimizer_state_reports_combined_nbytes(self):
        rng = np.random.RandomState(1)
        shape = (128, 128)
        m = torch.zeros(shape)
        v = torch.from_numpy(_adam_second_moment_like(rng, m.numel()).reshape(shape)).float()

        state = CompressedOptimizerState(shape, method="auto", min_size_for_g4=1)
        state.set(m, v)
        fp32_state_bytes = 2 * m.numel() * 4  # m + v, both fp32 -- "states are 2x params fp32" per F6 spec
        self.assertLess(state.nbytes(), fp32_state_bytes)


class AdversarialFallbackTest(unittest.TestCase):
    """Construct tensors genuinely hostile to each compression path and confirm the dense
    fallback correctly engages -- mirroring G4's own dense-fallback test pattern
    (mixle.tests.sorted_profile_quantizer_test.DenseFallbackTest)."""

    def test_g4_falls_back_on_bimodal_tensor(self):
        rng = np.random.RandomState(0)
        # Well-separated bimodal data: a unimodal Gaussian tail fit cannot match this, exactly the
        # scenario G4's own dense-fallback test uses.
        bimodal = np.concatenate([rng.normal(-10.0, 0.5, 5000), rng.normal(10.0, 0.5, 5000)])
        encoding = compress_moment(bimodal, method="g4", min_size_for_g4=1)
        self.assertEqual(encoding.method, "dense")
        recon = decompress_moment(encoding)
        np.testing.assert_allclose(recon, bimodal.astype(np.float32), rtol=1e-5)

    def test_int8_falls_back_on_single_outlier_block(self):
        # One block dominated by a single extreme outlier: the block's dynamic scale is set by
        # the outlier, crushing every other value in that block to (or near) zero -- genuinely
        # hostile to blockwise int8, unlike G4's per-tensor distribution fit which is untouched by
        # a single-value spike this small relative to the whole tensor.
        block_size = 256
        block = np.full(block_size, 1e-6)
        block[0] = 1e6
        encoding = compress_moment(block, method="int8", int8_block_size=block_size)
        self.assertEqual(encoding.method, "dense")

    def test_int8_does_not_fall_back_on_well_behaved_tensor(self):
        rng = np.random.RandomState(2)
        tame = rng.normal(0.0, 1.0, size=4096)
        encoding = compress_moment(tame, method="int8")
        self.assertEqual(encoding.method, "int8")

    def test_auto_picker_falls_back_all_the_way_to_dense_on_a_tensor_hostile_to_both(self):
        # A tensor built to defeat BOTH G4 (multi-modal -> bad KS fit) AND int8 (each mode's block
        # dominated by within-block magnitude spread) -- the auto picker should still land on a
        # trustworthy (dense) representation rather than silently accepting a bad compressed one.
        rng = np.random.RandomState(3)
        block_size = 256
        n_blocks = 20
        chunks = []
        for i in range(n_blocks):
            center = -10.0 if i % 2 == 0 else 10.0
            chunk = rng.normal(center, 0.01, block_size)
            chunk[0] = center * 1000.0  # per-block outlier spike, on top of the bimodal structure
            chunks.append(chunk)
        hostile = np.concatenate(chunks)
        encoding = compress_moment(hostile, method="auto", min_size_for_g4=1, int8_block_size=block_size)
        self.assertEqual(encoding.method, "dense")
        recon = decompress_moment(encoding)
        np.testing.assert_allclose(recon, hostile.astype(np.float32), rtol=1e-5)


class Fp8HardeningTest(unittest.TestCase):
    """fp8 hardening: real overflow/underflow detection with graceful fallback."""

    def test_well_behaved_tensor_uses_fp8(self):
        t = torch.randn(1024) * 0.1
        result = fp8_cast_with_guard(t)
        self.assertTrue(result.used_fp8)
        self.assertEqual(str(result.tensor.dtype), "torch.float8_e4m3fn")

    def test_overflow_triggers_fallback(self):
        t = torch.tensor([1.0, 1.0e5, -3.0])  # 1e5 exceeds float8_e4m3fn's ~448 max
        result = fp8_cast_with_guard(t)
        self.assertFalse(result.used_fp8)
        self.assertIn("overflow", result.reason)
        self.assertEqual(result.tensor.dtype, torch.bfloat16)

    def test_underflow_triggers_fallback(self):
        # float8_e4m3fn's smallest representable magnitudes are ~1e-3 (subnormals) to ~2e-2
        # (normals); a tensor of mostly ~1e-4-scale nonzero values flushes almost entirely to zero.
        t = torch.full((2048,), 1.0e-4)
        result = fp8_cast_with_guard(t)
        self.assertFalse(result.used_fp8)
        self.assertIn("underflow", result.reason)
        self.assertGreater(result.underflow_fraction, 0.5)

    def test_non_finite_input_triggers_fallback(self):
        t = torch.tensor([1.0, float("inf"), 2.0])
        result = fp8_cast_with_guard(t)
        self.assertFalse(result.used_fp8)
        self.assertEqual(result.tensor.dtype, torch.bfloat16)


class SelectiveRecomputePolicyTest(unittest.TestCase):
    """Cost-model decision-boundary test: memory-heavy/cheap-to-recompute -> recompute;
    memory-light/expensive-to-recompute -> do not recompute -- mirroring D6's own compile-
    economics decision-boundary test pattern."""

    def test_memory_heavy_cheap_block_is_recommended_for_recompute(self):
        policy = SelectiveRecomputePolicy()
        decision = policy.decide_block(block_index=0, activation_bytes=1.0e9, recompute_flops=1.0)
        self.assertIsInstance(decision, RecomputeDecision)
        self.assertTrue(decision.should_recompute)
        self.assertGreater(decision.net_benefit, 0.0)

    def test_memory_light_expensive_block_is_not_recommended_for_recompute(self):
        policy = SelectiveRecomputePolicy()
        decision = policy.decide_block(block_index=1, activation_bytes=1.0, recompute_flops=1.0e9)
        self.assertFalse(decision.should_recompute)
        self.assertLess(decision.net_benefit, 0.0)

    def test_apply_to_model_sets_a_per_block_list_and_forward_still_works(self):
        model = build_causal_lm(vocab=11, d_model=64, n_layer=3, n_head=4, block=512)
        policy = SelectiveRecomputePolicy(memory_value_per_byte=1.0, flop_cost_per_unit=1e-12)
        decisions = policy.apply_to_model(model, batch=8, seq_len=512)

        self.assertEqual(len(decisions), 3)
        self.assertIsInstance(model.gradient_checkpointing, list)
        self.assertEqual(len(model.gradient_checkpointing), 3)
        # A tiny recompute-cost weight relative to a large batch*seq_len activation footprint
        # should make recompute worth it for at least one block -- otherwise this test would not
        # actually be exercising the per-block-list code path in transformer.py's forward.
        self.assertTrue(any(model.gradient_checkpointing))

        model.train()
        x = torch.randint(0, 11, (2, 512)).float()
        out = model(x)
        self.assertEqual(tuple(out.shape), (2, 11))
        loss = torch.nn.functional.cross_entropy(out, torch.randint(0, 11, (2,)))
        loss.backward()  # must not raise -- the per-block list path must be backward-compatible

    def test_bool_flag_still_works_after_per_block_extension(self):
        """Regression: the original all-or-nothing bool flag (mixle.tests.grad_control_test) must
        still work unchanged after extending forward() to also accept a per-block list."""
        model = build_causal_lm(vocab=11, d_model=16, n_layer=2, n_head=2, block=8, gradient_checkpointing=True)
        self.assertTrue(model.gradient_checkpointing)
        model.train()
        out = model(torch.randint(0, 11, (3, 8)).float())
        self.assertEqual(tuple(out.shape), (3, 11))

    def test_decisions_follow_actual_heterogeneous_block_manifest(self):
        class SmallBlock(torch.nn.Module):
            d_model = 4

            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

        class LargeBlock(torch.nn.Module):
            d_model = 4

            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Sequential(torch.nn.Linear(4, 100), torch.nn.Linear(100, 4))

        class EditedModel:
            d_model = 4
            n_layer = 99  # deliberately stale metadata
            blocks = torch.nn.ModuleList([SmallBlock(), LargeBlock()])

        decisions = SelectiveRecomputePolicy(memory_value_per_byte=1.0, flop_cost_per_unit=0.1).decide_model(
            EditedModel(),
            batch=1,
            seq_len=1,
        )
        self.assertEqual(len(decisions), 2)
        self.assertTrue(decisions[0].should_recompute)
        self.assertFalse(decisions[1].should_recompute)
        self.assertLess(decisions[0].recompute_flops, decisions[1].recompute_flops)

    def test_recompute_policy_rejects_invalid_dimensions_and_costs(self):
        for kwargs in (
            {"memory_value_per_byte": -1.0},
            {"flop_cost_per_unit": np.nan},
        ):
            with self.assertRaises(ValueError):
                SelectiveRecomputePolicy(**kwargs)
        policy = SelectiveRecomputePolicy()
        for args in ((-1, 1.0, 1.0), (0, -1.0, 1.0), (0, 1.0, np.inf)):
            with self.assertRaises((TypeError, ValueError)):
                policy.decide_block(*args)
        model = build_causal_lm(vocab=5, d_model=4, n_layer=1, n_head=1, block=4)
        for kwargs in (
            {"batch": 0, "seq_len": 4},
            {"batch": 1, "seq_len": -1},
            {"batch": 1, "seq_len": 4, "dtype_bytes": 0},
        ):
            with self.assertRaises(ValueError):
                policy.decide_model(model, **kwargs)


class Int8BlockwiseUnitTest(unittest.TestCase):
    """Direct unit coverage of the int8 quantize/dequantize round trip, independent of the
    higher-level compress_moment picker."""

    def test_round_trip_is_low_error_on_smooth_data(self):
        rng = np.random.RandomState(0)
        flat = rng.normal(0.0, 2.0, size=5000)
        codes, scales = quantize_int8_blockwise(flat, block_size=512)
        recon = dequantize_int8_blockwise(codes, scales, block_size=512)
        rel_err = np.linalg.norm(recon - flat) / np.linalg.norm(flat)
        self.assertLess(rel_err, 0.02)  # 8-bit blockwise quantization is a tight fit for smooth data


class CompressionContractTest(unittest.TestCase):
    def test_fp8_contract_rejects_unknown_formats_fallbacks_and_thresholds(self):
        tensor = torch.ones(4)
        with self.assertRaises(ValueError):
            fp8_cast_with_guard(tensor, fp8_dtype="float32")
        with self.assertRaises(ValueError):
            fp8_cast_with_guard(tensor, fallback_dtype=torch.int32)
        for threshold in (-0.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                fp8_cast_with_guard(tensor, underflow_fraction_threshold=threshold)
        with self.assertRaises(TypeError):
            fp8_cast_with_guard(np.ones(4))

    def test_int8_payload_rejects_nonfinite_incomplete_and_mistyped_data(self):
        with self.assertRaises(ValueError):
            quantize_int8_blockwise(np.array([1.0, np.nan]))
        for block_size in (0, -1):
            with self.assertRaises(ValueError):
                quantize_int8_blockwise(np.ones(2), block_size=block_size)
        codes = np.array([1, 2, 3], dtype=np.int8)
        with self.assertRaises(ValueError):
            dequantize_int8_blockwise(codes.astype(np.int16), np.ones(1), block_size=3)
        with self.assertRaises(ValueError):
            dequantize_int8_blockwise(codes, np.array([], dtype=np.float32), block_size=3)
        with self.assertRaises(ValueError):
            dequantize_int8_blockwise(codes, np.array([np.nan]), block_size=3)

        malformed = CompressedMomentEncoding(
            method="int8",
            shape=(4,),
            int8_codes=codes,
            int8_scales=np.ones(1, dtype=np.float32),
            int8_block_size=3,
        )
        with self.assertRaisesRegex(ValueError, "requires 4"):
            decompress_moment(malformed)

    def test_unknown_method_is_rejected_and_fallback_reason_is_explicit(self):
        with self.assertRaises(ValueError):
            compress_moment(np.ones(4, dtype=np.float32), method="innt8")
        hostile = np.full(256, 1.0e-6, dtype=np.float32)
        hostile[0] = 1.0e6
        encoding = compress_moment(hostile, method="int8", int8_block_size=256)
        self.assertEqual(encoding.requested_method, "int8")
        self.assertEqual(encoding.method, "dense")
        self.assertIsNotNone(encoding.fallback_reason)

    def test_dense_encoding_preserves_float64_exactly(self):
        value = np.array([1.0 + 2.0**-40], dtype=np.float64)
        encoding = compress_moment(value, method="dense")
        self.assertEqual(encoding.source_dtype, np.dtype(np.float64).str)
        restored = decompress_moment(encoding)
        self.assertEqual(restored.dtype, np.float64)
        np.testing.assert_array_equal(restored, value)

    def test_g4_encoding_round_trips_through_tensor_only_schema(self):
        values = np.random.RandomState(9).normal(size=4096).astype(np.float32)
        encoding = compress_moment(values, method="g4", min_size_for_g4=1)
        self.assertEqual(encoding.method, "g4")
        rebuilt = CompressedMomentEncoding.from_tensor_state(encoding.to_tensor_state())
        np.testing.assert_array_equal(decompress_moment(rebuilt), decompress_moment(encoding))

    def test_compressed_adam_uses_declared_stable_state_precision(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
        optimizer = CompressedAdam([parameter], lr=1.0e-2, compression_method="dense")
        parameter.grad = torch.tensor([0.5], dtype=torch.float16)
        optimizer.step()
        compressed = optimizer.state[parameter]["compressed"]
        m, v = compressed.get()
        self.assertEqual(m.dtype, torch.float32)
        self.assertEqual(v.dtype, torch.float32)

    def test_compressed_adam_validates_controls_and_gradients(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        for kwargs in (
            {"lr": np.nan},
            {"eps": 0.0},
            {"weight_decay": -1.0},
            {"betas": (1.0, 0.9)},
            {"compression_method": "innt8"},
            {"state_dtype": torch.float16},
        ):
            with self.assertRaises((TypeError, ValueError)):
                CompressedAdam([parameter], **kwargs)

        optimizer = CompressedAdam([parameter])
        parameter.grad = torch.tensor([1.0, np.nan])
        with self.assertRaises(FloatingPointError):
            optimizer.step()

        embedding = torch.nn.Embedding(5, 2, sparse=True)
        sparse_optimizer = CompressedAdam(embedding.parameters())
        embedding(torch.tensor([1, 2])).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            sparse_optimizer.step()

    def test_optimizer_checkpoint_is_tensor_only_and_resumable(self):
        first = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        optimizer = CompressedAdam([first], lr=1.0e-2, compression_method="dense")
        first.grad = torch.tensor([0.25, -0.5])
        optimizer.step()
        checkpoint = optimizer.state_dict()
        tensor_state = next(iter(checkpoint["state"].values()))["compressed_tensor_state"]

        def assert_tensor_tree(value):
            if isinstance(value, dict):
                for child in value.values():
                    assert_tensor_tree(child)
            else:
                self.assertIsInstance(value, torch.Tensor)

        assert_tensor_tree(tensor_state)
        second = torch.nn.Parameter(first.detach().clone())
        resumed = CompressedAdam([second], lr=1.0e-2, compression_method="dense")
        resumed.load_state_dict(checkpoint)
        first.grad = torch.tensor([-0.1, 0.2])
        second.grad = first.grad.clone()
        optimizer.step()
        resumed.step()
        torch.testing.assert_close(second, first)


if __name__ == "__main__":
    unittest.main()
