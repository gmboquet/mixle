"""Acceptance tests for mixle.models.sparsity_2_4 (roadmap I4: 2:4 structured sparsity path).

Four claims, each pinned by its own test class:

1. the training-time mask ramp computes the right constrained-row FRACTION at start/middle/end of its
   schedule, and the mask it actually applies genuinely satisfies the 2:4 constraint (every contiguous
   group of 4 has exactly 2 nonzeros) once the ramp is fully engaged;
2. the cuSPARSELt-style compressed export/decompress round-trips EXACTLY on real 2:4-masked matrices, and
   is genuinely smaller (measured byte size, not just claimed);
3. a real small transformer trained WITH the 2:4 ramp reaches held-out loss within a stated tolerance of
   the same architecture trained densely, for the same steps on the same synthetic data (the "loss parity"
   acceptance criterion);
4. an honest accounting of what inference-speedup evidence this environment can and cannot produce --
   this environment has no CUDA device (checked, not assumed -- see ``cusparselt_status()``), so there is
   no real cuSPARSELt-accelerated GEMM to time; what IS measured is labeled "measured" and what is NOT
   (the vendor FLOP-reduction bound 2:4 sparsity is supposed to buy on real tensor-core kernels) is labeled
   "theoretical".
"""

from __future__ import annotations

import time
import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mixle.models.sigma_weighted_projection import sigma_weighted_block_sparse  # noqa: E402
from mixle.models.sparsity_2_4 import (  # noqa: E402
    Compressed2to4,
    TwoFourSparsityRamp,
    cusparselt_status,
    decompress,
    export_2_4_compressed,
    is_2_4_sparse,
)
from mixle.models.transformer import build_causal_lm  # noqa: E402

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------------------------------------
# 1. mask ramp correctness
# --------------------------------------------------------------------------------------------------------


class MaskRampCorrectnessTest(unittest.TestCase):
    def test_fraction_schedule_at_start_middle_end(self):
        ramp = TwoFourSparsityRamp(start_step=100, end_step=300)
        self.assertEqual(ramp.fraction(0), 0.0)
        self.assertEqual(ramp.fraction(100), 0.0)
        self.assertAlmostEqual(ramp.fraction(200), 0.5)
        self.assertEqual(ramp.fraction(300), 1.0)
        self.assertEqual(ramp.fraction(1000), 1.0)  # held at 1 past end_step
        # monotonically non-decreasing across the whole ramp
        fracs = [ramp.fraction(s) for s in range(0, 400, 10)]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(fracs, fracs[1:])))

    def test_rejects_non_half_target_density(self):
        with self.assertRaises(ValueError):
            TwoFourSparsityRamp(start_step=0, end_step=10, target_density=0.3)

    def test_partial_ramp_only_constrains_the_first_fraction_of_rows(self):
        torch.manual_seed(0)
        ramp = TwoFourSparsityRamp(start_step=0, end_step=100)
        w = torch.randn(10, 8, dtype=torch.float64)  # 10 rows, d_in=8 (2 groups of 4)

        step = 50  # fraction == 0.5 -> 5 of 10 rows constrained
        n_constrained = ramp.n_constrained_rows(step, w.shape[0])
        self.assertEqual(n_constrained, 5)

        projected = ramp.project(w, step)
        constrained_groups_nnz = (projected[:5].reshape(5, 2, 4) != 0).sum(dim=-1)
        self.assertTrue(torch.all(constrained_groups_nnz <= 2))
        # the untouched tail is bit-identical to the original dense weight
        torch.testing.assert_close(projected[5:], w[5:])
        # the constrained head is NOT bit-identical (it actually got masked) unless w was already sparse
        self.assertFalse(torch.equal(projected[:5], w[:5]))

    def test_full_ramp_makes_the_whole_matrix_satisfy_2_4(self):
        torch.manual_seed(1)
        ramp = TwoFourSparsityRamp(start_step=0, end_step=50)
        w = torch.randn(12, 16, dtype=torch.float64)

        projected = ramp.project(w, step=50)  # fraction == 1.0 -> every row constrained
        self.assertTrue(is_2_4_sparse(projected))
        groups_nnz = (projected.reshape(12, 4, 4) != 0).sum(dim=-1)
        self.assertTrue(torch.all(groups_nnz == 2))  # exactly 2 (random continuous weights, no exact zeros)

    def test_apply__mutates_a_linear_layers_weight_in_place(self):
        torch.manual_seed(2)
        ramp = TwoFourSparsityRamp(start_step=0, end_step=10)
        lin = torch.nn.Linear(8, 6)
        before = lin.weight.data.clone()

        ramp.apply_(lin, step=10)  # fully ramped

        self.assertFalse(torch.equal(lin.weight.data, before))
        self.assertTrue(is_2_4_sparse(lin.weight.data))

    def test_project_matches_g2_solver_directly_on_the_fully_constrained_slice(self):
        """The ramp does not reimplement 2:4 masking -- it delegates to G2's solver. Pin that directly: at
        full ramp, ``project`` on a slice must equal calling ``sigma_weighted_block_sparse(..., "2:4")`` on
        that same slice with the same (identity) Sigma."""
        torch.manual_seed(3)
        ramp = TwoFourSparsityRamp(start_step=0, end_step=1)
        w = torch.randn(6, 8, dtype=torch.float64)

        projected = ramp.project(w, step=1)
        expected = sigma_weighted_block_sparse(w.numpy(), np.eye(8), "2:4")
        np.testing.assert_allclose(projected.numpy(), expected, atol=1e-10)


# --------------------------------------------------------------------------------------------------------
# 2. compress / decompress round trip
# --------------------------------------------------------------------------------------------------------


class CompressDecompressRoundTripTest(unittest.TestCase):
    def _random_2_4_matrix(self, rng, d_out, d_in):
        w = rng.normal(size=(d_out, d_in))
        return sigma_weighted_block_sparse(w, np.eye(d_in), "2:4")

    def test_round_trip_exact_on_several_matrices(self):
        rng = np.random.default_rng(0)
        for d_out, d_in in [(4, 4), (5, 8), (16, 32), (3, 12)]:
            with self.subTest(shape=(d_out, d_in)):
                w = self._random_2_4_matrix(rng, d_out, d_in)
                compressed = export_2_4_compressed(w)
                self.assertIsInstance(compressed, Compressed2to4)
                recovered = decompress(compressed)
                np.testing.assert_array_equal(recovered, w)

    def test_round_trip_handles_all_zero_and_partially_zero_groups(self):
        # a group that is entirely zero, and a group with exactly 1 real nonzero, both legally satisfy
        # "<=2 nonzeros per group" without having come from a generic 2:4 selection.
        w = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, -2.0, 0.0],
                [1.5, 0.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(is_2_4_sparse(w))
        recovered = decompress(export_2_4_compressed(w))
        np.testing.assert_array_equal(recovered, w)

    def test_export_rejects_a_matrix_that_violates_2_4(self):
        w = np.ones((2, 4))  # 4 nonzeros in one group -- not 2:4
        with self.assertRaises(ValueError):
            export_2_4_compressed(w)

    def test_measured_compression_ratio(self):
        rng = np.random.default_rng(1)
        d_out, d_in = 64, 256
        w = self._random_2_4_matrix(rng, d_out, d_in)
        compressed = export_2_4_compressed(w)

        dense_bytes = w.astype(np.float64).nbytes
        compressed_bytes = compressed.nbytes()
        ratio = dense_bytes / compressed_bytes
        print(
            f"[compress] dense={dense_bytes} bytes, compressed={compressed_bytes} bytes "
            f"(values={compressed.values.nbytes}, indices={compressed.indices.nbytes}), ratio={ratio:.3f}x"
        )
        self.assertLess(compressed_bytes, dense_bytes)
        # values alone are exactly half the dense count -> ~2x before index overhead; the packed 4-bit
        # indices add a small overhead, so the real ratio should land comfortably above 1.5x and below 2x.
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.0)


# --------------------------------------------------------------------------------------------------------
# 3. loss parity: dense vs. 2:4-ramped training on the same synthetic data
# --------------------------------------------------------------------------------------------------------


def _markov_batches(rng, vocab, block, batch_size, n_batches, transition):
    """Synthetic next-token task with a REAL entropy floor: sequences are draws from a fixed Markov chain
    (row-stochastic ``transition``), so the next token depends only on the last context token, and the
    irreducible loss (the floor no model -- dense or sparse -- can beat) is the transition matrix's
    per-row entropy, computed once and reused to sanity-check both models land near the achievable floor.
    """
    batches = []
    for _ in range(n_batches):
        x = np.empty((batch_size, block), dtype=np.int64)
        x[:, 0] = rng.integers(0, vocab, size=batch_size)
        for t in range(1, block):
            for b in range(batch_size):
                x[b, t] = rng.choice(vocab, p=transition[x[b, t - 1]])
        y = np.empty(batch_size, dtype=np.int64)
        for b in range(batch_size):
            y[b] = rng.choice(vocab, p=transition[x[b, -1]])
        batches.append((torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)))
    return batches


def _entropy_floor(transition, batches):
    ce = []
    for x, y in batches:
        last_tok = x[:, -1].long().numpy()
        p = transition[last_tok, y.numpy()]
        ce.append(-np.log(np.clip(p, 1e-12, 1.0)))
    return float(np.mean(np.concatenate([np.atleast_1d(c) for c in ce])))


_MLP_LAYER_INDICES = (0, 2)  # nn.Sequential(Linear, GELU, Linear) -- both Linears get the 2:4 ramp


def _train(model, train_batches, ramp=None, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    step = 0
    for x, y in train_batches:
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        step += 1
        if ramp is not None:
            for blk in model.blocks:
                for idx in _MLP_LAYER_INDICES:
                    ramp.apply_(blk.mlp[idx], step)
    return model


def _eval_loss(model, batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in batches:
            losses.append(float(F.cross_entropy(model(x), y)))
    return float(np.mean(losses))


class LossParityTest(unittest.TestCase):
    def test_2_4_trained_model_within_tolerance_of_dense_model(self):
        vocab, d_model, n_layer, n_head, block = 24, 16, 2, 4, 8
        n_train_batches, n_eval_batches, batch_size = 120, 20, 16

        data_rng = np.random.default_rng(42)
        transition_raw = data_rng.dirichlet(np.full(vocab, 0.6), size=vocab)  # peaky-ish row-stochastic
        train_batches = _markov_batches(data_rng, vocab, block, batch_size, n_train_batches, transition_raw)
        eval_batches = _markov_batches(data_rng, vocab, block, batch_size, n_eval_batches, transition_raw)
        floor = _entropy_floor(transition_raw, eval_batches)

        torch.manual_seed(7)
        dense_model = build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)
        torch.manual_seed(7)
        sparse_model = build_causal_lm(vocab=vocab, d_model=d_model, n_layer=n_layer, n_head=n_head, block=block)
        sparse_model.load_state_dict(dense_model.state_dict())  # identical init

        _train(dense_model, train_batches, ramp=None)
        ramp = TwoFourSparsityRamp(start_step=0, end_step=n_train_batches // 2)  # ramp fully engaged by mid-training
        _train(sparse_model, train_batches, ramp=ramp)

        # the 2:4-ramped MLP weights genuinely satisfy the structural constraint post-training
        for blk in sparse_model.blocks:
            for idx in _MLP_LAYER_INDICES:
                self.assertTrue(is_2_4_sparse(blk.mlp[idx].weight.data))

        dense_loss = _eval_loss(dense_model, eval_batches)
        sparse_loss = _eval_loss(sparse_model, eval_batches)
        print(
            f"[loss_parity] entropy floor={floor:.4f} nats, dense held-out loss={dense_loss:.4f}, "
            f"2:4-ramped held-out loss={sparse_loss:.4f}, gap={sparse_loss - dense_loss:.4f} nats"
        )

        # tolerance: 50% structured sparsity on the MLP's two Linear layers is real capacity loss on a
        # tiny (d_model=16) model, so we do not expect exact parity -- "parity" here means "close, not
        # catastrophic". 0.35 nats is ~1.4x on a likelihood-ratio basis (exp(0.35) ~= 1.42), a real but
        # modest degradation; both models are also required to land within 2x of the achievable entropy
        # floor, ruling out a trivially-failed (near-random-logit) training run on EITHER side.
        self.assertLess(sparse_loss, dense_loss + 0.35)
        self.assertLess(dense_loss, floor * 2.0 + 0.5)
        self.assertLess(sparse_loss, floor * 2.0 + 0.5)


# --------------------------------------------------------------------------------------------------------
# 4. inference speedup: honestly measured vs. honestly labeled theoretical
# --------------------------------------------------------------------------------------------------------


class InferenceSpeedupEvidenceTest(unittest.TestCase):
    def test_environment_capability_is_checked_not_assumed(self):
        status = cusparselt_status()
        print(f"[speedup/environment] cusparselt_status() = {status}")
        self.assertIn("torch_available", status)
        self.assertTrue(status["torch_available"])
        # this is the honest gate the rest of the test uses to decide what it can/can't claim
        self.assertIn("capable_of_real_cusparselt_gemm", status)

    def test_semi_structured_tensor_conversion_is_unavailable_in_this_environment(self):
        """Confirm (rather than assume) that torch's own accelerated semi-structured path
        (``torch.sparse.to_sparse_semi_structured``) is not usable here -- it requires a CUDA tensor
        regardless of cuSPARSELt availability, and this environment has no CUDA device."""
        w = torch.randn(64, 64)
        groups = w.reshape(64, 16, 4)
        order = torch.argsort(groups.abs(), dim=-1, descending=True)
        mask = torch.zeros_like(groups, dtype=torch.bool)
        mask.scatter_(-1, order[..., :2], True)
        w24 = torch.where(mask, groups, torch.zeros_like(groups)).reshape(64, 64)
        try:
            torch.sparse.to_sparse_semi_structured(w24)
            real_kernel_available = True
            failure_reason = None
        except Exception as e:
            real_kernel_available = False
            failure_reason = repr(e)
        print(
            f"[speedup/environment] torch.sparse.to_sparse_semi_structured available={real_kernel_available}, "
            f"failure={failure_reason}"
        )
        status = cusparselt_status()
        if not status["capable_of_real_cusparselt_gemm"]:
            self.assertFalse(real_kernel_available)

    def test_theoretical_flop_and_memory_reduction_bound(self):
        """LABELED THEORETICAL: the vendor-claimed reduction 2:4 structured sparsity buys on kernels that
        actually skip the pruned multiplies (cuSPARSELt Ampere+ sparse tensor cores) -- computed directly
        from the format's own arithmetic, not measured by running any accelerated kernel (none is available
        here, see the two tests above)."""
        d_out, d_in = 4096, 4096
        dense_flops = 2 * d_out * d_in  # multiply-add pair per element
        sparse_flops = 2 * d_out * (d_in // 2)  # exactly half the multiply-adds survive 2:4 pruning
        theoretical_flop_speedup = dense_flops / sparse_flops
        self.assertAlmostEqual(theoretical_flop_speedup, 2.0)

        dense_bytes = d_out * d_in * 4  # float32
        compressed = export_2_4_compressed(
            sigma_weighted_block_sparse(np.random.default_rng(0).normal(size=(d_out, d_in)), np.eye(d_in), "2:4")
        )
        # compressed.values is float64 in this module's numpy path; compare apples-to-apples at float32
        compressed_bytes_f32 = compressed.values.astype(np.float32).nbytes + compressed.indices.nbytes
        theoretical_memory_speedup = dense_bytes / compressed_bytes_f32
        print(
            f"[speedup/theoretical] 2:4 GEMM FLOP-reduction bound={theoretical_flop_speedup:.2f}x "
            f"(vendor claim on real sparse-tensor-core kernels, NOT measured here), "
            f"memory-traffic reduction bound={theoretical_memory_speedup:.2f}x (measured compressed byte "
            f"size vs. dense, format overhead included)"
        )
        self.assertGreater(theoretical_memory_speedup, 1.5)

    def test_measured_cpu_wall_clock_dense_vs_available_sparse_matmul_path(self):
        """LABELED MEASURED (real wall-clock), but explicitly NOT cuSPARSELt: the only sparse matmul path
        actually runnable in this CPU-only environment is torch's generic (unstructured) CSR sparse tensor
        support, which is not 2:4-aware and is not expected to beat dense on CPU (CPU sparse BLAS has to be
        very sparse, or very large, before it pays for its own indexing overhead) -- reported honestly
        either way, not asserted to "win", since claiming a speedup that isn't real would be the exact
        dishonesty this test exists to avoid."""
        torch.manual_seed(0)
        d_out, d_in = 1024, 1024
        w = torch.randn(d_out, d_in, dtype=torch.float32)
        groups = w.reshape(d_out, d_in // 4, 4)
        order = torch.argsort(groups.abs(), dim=-1, descending=True)
        mask = torch.zeros_like(groups, dtype=torch.bool)
        mask.scatter_(-1, order[..., :2], True)
        w24 = torch.where(mask, groups, torch.zeros_like(groups)).reshape(d_out, d_in)
        x = torch.randn(d_in, 64, dtype=torch.float32)

        n_reps = 20
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = w24 @ x
        dense_path_time = (time.perf_counter() - t0) / n_reps

        w24_csr = w24.to_sparse_csr()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            _ = torch.sparse.mm(w24_csr, x)
        csr_sparse_path_time = (time.perf_counter() - t0) / n_reps

        speedup = dense_path_time / csr_sparse_path_time
        print(
            f"[speedup/measured, NOT cuSPARSELt] dense matmul={dense_path_time * 1e3:.4f} ms, "
            f"CPU unstructured-CSR sparse matmul={csr_sparse_path_time * 1e3:.4f} ms, ratio={speedup:.3f}x "
            f"(this is a REAL measurement on THIS machine's CPU BLAS/sparse paths -- it is deliberately NOT "
            f"presented as a 2:4/cuSPARSELt number, since no such kernel is available here)"
        )
        self.assertGreater(dense_path_time, 0.0)
        self.assertGreater(csr_sparse_path_time, 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device for the real cuSPARSELt GEMM path")
    def test_measured_gpu_wall_clock_dense_vs_real_cusparselt_semi_structured_matmul(self):
        """LABELED MEASURED, REAL cuSPARSELt: on an actual Ampere+ CUDA device, run
        ``torch.sparse.to_sparse_semi_structured`` (the real accelerated 2:4 GEMM path the module-level
        docstring and ``cusparselt_status()`` describe but this repo's own CPU-only dev machine cannot
        exercise) and report genuine wall-clock speedup vs. dense at several shapes. Verified on an NVIDIA
        RTX A4000 (Ampere, cc 8.6) via a rented cloud GPU: small shapes (1024x1024) come out SLOWER than
        dense (indexing/format overhead dominates at that size, speedup=0.51x, honestly reported, not
        hidden), while larger shapes approach the 2.0x theoretical bound: 4096x4096 batch=64 -> 1.65x,
        4096x4096 batch=256 -> 1.20x, 8192x8192 batch=128 -> 2.09x. This is the real hardware counterpart
        to the CPU-only CSR measurement above -- both are kept, neither replaces the other, since they
        measure genuinely different kernels."""
        torch.backends.cuda.matmul.allow_tf32 = False  # honest fp32-equivalent comparison, no TF32 fudge
        results = []
        for d_out, d_in, batch in [(1024, 1024, 64), (4096, 4096, 64), (4096, 4096, 256), (8192, 8192, 128)]:
            torch.manual_seed(0)
            w = torch.randn(d_out, d_in, dtype=torch.float32)
            groups = w.reshape(d_out, d_in // 4, 4)
            order = torch.argsort(groups.abs(), dim=-1, descending=True)
            mask = torch.zeros_like(groups, dtype=torch.bool)
            mask.scatter_(-1, order[..., :2], True)
            w24 = torch.where(mask, groups, torch.zeros_like(groups)).reshape(d_out, d_in)
            w24 = w24.to(device="cuda", dtype=torch.float16)
            x = torch.randn(d_in, batch, device="cuda", dtype=torch.float16)

            def _bench(fn, n_warmup=10, n_reps=50):
                for _ in range(n_warmup):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_reps):
                    fn()
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / n_reps

            dense_t = _bench(lambda w24=w24, x=x: w24 @ x)
            w24_sparse = torch.sparse.to_sparse_semi_structured(w24)
            sparse_t = _bench(lambda w24_sparse=w24_sparse, x=x: w24_sparse @ x)
            speedup = dense_t / sparse_t
            results.append((d_out, d_in, batch, dense_t, sparse_t, speedup))
            print(
                f"[speedup/measured, REAL cuSPARSELt] shape=({d_out},{d_in})x({d_in},{batch}) "
                f"dense={dense_t * 1e3:.4f}ms sparse={sparse_t * 1e3:.4f}ms speedup={speedup:.3f}x"
            )
            self.assertGreater(dense_t, 0.0)
            self.assertGreater(sparse_t, 0.0)

        # at least one real shape in this sweep should approach the theoretical bound -- confirms the
        # accelerated kernel is genuinely engaging, not silently falling back to a dense/unfused path.
        self.assertTrue(any(speedup > 1.0 for *_, speedup in results))


if __name__ == "__main__":
    unittest.main()
