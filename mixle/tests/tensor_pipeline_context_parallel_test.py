"""TP/PP/CP correctness for :class:`~mixle.models.transformer.CausalLM` (F1: parallelism atop FSDP2).

What this test module DOES verify (exact/near-exact, small-scale, in-process): the actual sharding /
partition / reconciliation MATH for tensor, pipeline, and context parallelism is correct -- the sharded
forward reproduces the dense (un-sharded) model's forward, for real ``CausalLM`` weights.

What this test module DOES NOT and CANNOT verify: the roadmap's F1 acceptance criterion ("70B-config
across >=512 GPUs at published-comparable MFU") requires 512+ real accelerators and a multi-week training
run -- there is no way to honestly measure that number on a laptop/CI box, and no test here claims to.
This module tests the INTEGRATION LAYER (the sharding plan + reconstruction correctness + the
``lm.fit(distributed=True, tp_size=..., pp_size=..., cp_size=...)`` surface) that a real multi-GPU run
would sit on top of.
"""

import unittest

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class TensorParallelTest(unittest.TestCase):
    def test_tp_attention_matches_dense_attention(self):
        from mixle.models.transformer import CausalAttention
        from mixle.utils.parallel.tensor_pipeline_context_parallel import tp_attention_forward, tp_shard_attention

        torch.manual_seed(0)
        d_model, n_head, b, t = 64, 8, 3, 16
        attn = CausalAttention(d_model, n_head).eval()
        x = torch.randn(b, t, d_model)
        with torch.no_grad():
            dense = attn(x)
            for tp_size in (1, 2, 4):
                shards = tp_shard_attention(attn, tp_size)
                self.assertEqual(len(shards), tp_size)
                sharded = tp_attention_forward(x, shards)
                self.assertTrue(
                    torch.allclose(dense, sharded, atol=1e-5, rtol=1e-4),
                    "tp_size=%d attention output diverges from dense" % tp_size,
                )

    def test_tp_shard_causal_lm_matches_dense_forward(self):
        from mixle.models.transformer import build_causal_lm
        from mixle.utils.parallel.tensor_pipeline_context_parallel import tp_forward_causal_lm, tp_shard_causal_lm

        torch.manual_seed(1)
        vocab, d_model, n_layer, n_head, block = 37, 48, 3, 6, 12
        model = build_causal_lm(vocab, d_model, n_layer, n_head, block).eval()
        x = torch.randint(0, vocab, (4, block)).float()
        with torch.no_grad():
            dense = model(x)
            for tp_size in (1, 2, 3, 6):
                shard = tp_shard_causal_lm(model, tp_size)
                sharded = tp_forward_causal_lm(model, x, shard)
                self.assertTrue(
                    torch.allclose(dense, sharded, atol=1e-4, rtol=1e-4),
                    "tp_size=%d CausalLM output diverges from dense" % tp_size,
                )

    def test_tp_requires_head_divisible_shard_count(self):
        from mixle.models.transformer import CausalAttention
        from mixle.utils.parallel.tensor_pipeline_context_parallel import tp_shard_attention

        attn = CausalAttention(32, 4)
        with self.assertRaises(AssertionError):
            tp_shard_attention(attn, 3)  # 4 heads not divisible by 3 ranks


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class PipelineParallelTest(unittest.TestCase):
    def test_pp_partition_and_pipeline_forward_match_dense(self):
        from mixle.models.transformer import build_causal_lm
        from mixle.utils.parallel.tensor_pipeline_context_parallel import pipeline_forward, pp_partition_causal_lm

        torch.manual_seed(2)
        vocab, d_model, n_layer, n_head, block = 29, 32, 6, 4, 10
        model = build_causal_lm(vocab, d_model, n_layer, n_head, block).eval()
        x = torch.randint(0, vocab, (8, block)).float()
        with torch.no_grad():
            dense = model(x)
            for pp_size in (1, 2, 3, 6):
                stages = pp_partition_causal_lm(model, pp_size)
                self.assertEqual(len(stages), pp_size)
                self.assertEqual(sum(len(s.stage_blocks) for s in stages), n_layer)
                for n_micro in (1, 2, 4):
                    out = pipeline_forward(stages, x, n_micro)
                    self.assertTrue(
                        torch.allclose(dense, out, atol=1e-5, rtol=1e-4),
                        "pp_size=%d n_microbatches=%d pipeline output diverges from dense" % (pp_size, n_micro),
                    )

    def test_pp_stages_own_disjoint_contiguous_blocks(self):
        from mixle.models.transformer import build_causal_lm
        from mixle.utils.parallel.tensor_pipeline_context_parallel import pp_partition_causal_lm

        model = build_causal_lm(11, d_model=16, n_layer=7, n_head=2, block=8)
        stages = pp_partition_causal_lm(model, 3)
        sizes = [len(s.stage_blocks) for s in stages]
        self.assertEqual(sum(sizes), 7)
        self.assertTrue(max(sizes) - min(sizes) <= 1)  # balanced partition
        self.assertIsNotNone(stages[0].tok)
        self.assertIsNone(stages[-1].tok)
        self.assertIsNotNone(stages[-1].head)
        self.assertIsNone(stages[0].head)


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class ContextParallelTest(unittest.TestCase):
    def test_cp_attention_matches_dense_attention(self):
        from mixle.models.transformer import CausalAttention
        from mixle.utils.parallel.tensor_pipeline_context_parallel import cp_attention_forward, cp_shard_sequence

        torch.manual_seed(3)
        d_model, n_head, b, t = 40, 4, 2, 24
        attn = CausalAttention(d_model, n_head).eval()
        x = torch.randn(b, t, d_model)
        with torch.no_grad():
            dense = attn(x)
            for cp_size in (1, 2, 3, 4):
                if t % cp_size:
                    continue
                chunks = cp_shard_sequence(x, cp_size)
                out_chunks = cp_attention_forward(attn, chunks)
                sharded = torch.cat(out_chunks, dim=1)
                self.assertTrue(
                    torch.allclose(dense, sharded, atol=1e-5, rtol=1e-4),
                    "cp_size=%d attention output diverges from dense" % cp_size,
                )

    def test_cp_forward_causal_lm_matches_dense_all_positions(self):
        from mixle.models.language_model import _forward_all_positions
        from mixle.models.transformer import build_causal_lm
        from mixle.utils.parallel.tensor_pipeline_context_parallel import cp_forward_causal_lm

        torch.manual_seed(4)
        vocab, d_model, n_layer, n_head, block = 23, 32, 3, 4, 24
        model = build_causal_lm(vocab, d_model, n_layer, n_head, block).eval()
        x = torch.randint(0, vocab, (3, block)).float()
        with torch.no_grad():
            dense = _forward_all_positions(model, x)
            for cp_size in (1, 2, 4, 6):
                sharded = cp_forward_causal_lm(model, x, cp_size)
                self.assertTrue(
                    torch.allclose(dense, sharded, atol=1e-4, rtol=1e-4),
                    "cp_size=%d full-model output diverges from dense at some position" % cp_size,
                )


@unittest.skipUnless(_HAS_TORCH, "torch not installed")
class FitDistributedSurfaceTest(unittest.TestCase):
    """Smoke test: ``lm.fit(distributed=True, tp_size=..., pp_size=..., cp_size=...)`` runs and trains.

    This exercises the EXTENDED ``LM.fit`` signature end to end at small (single-process, simulated-rank)
    scale -- it does NOT run on real multi-GPU hardware (none is available here) and is not a claim about
    MFU or scale; it confirms the declarative surface accepts and threads the new parallelism knobs.
    """

    def _corpus(self):
        text = "mixle composes the layer above the trainer, not the trainer itself. " * 12
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        return np.array([stoi[c] for c in text]), len(chars)

    def test_fit_distributed_with_tp_pp_cp_smoke(self):
        from mixle.models.language_model import LM

        ids, v = self._corpus()
        lm = LM(vocab=v, d_model=32, n_layer=4, n_head=4, block=16)
        before = float(lm.nll(ids))
        lm.fit(ids, distributed=True, epochs=2, batch_size=16, tp_size=2, pp_size=2, cp_size=2)
        after = float(lm.nll(ids))
        self.assertTrue(np.isfinite(after))
        self.assertLessEqual(after, before + 5.0)  # training ran and produced a sane (finite, bounded) loss

    def test_fit_distributed_default_still_works_without_parallelism_dims(self):
        from mixle.models.language_model import LM

        ids, v = self._corpus()
        lm = LM(vocab=v, d_model=32, n_layer=2, n_head=2, block=16)
        lm.fit(ids, distributed=True, epochs=1, batch_size=16)  # tp_size=pp_size=cp_size=1 default: unchanged path
        self.assertTrue(np.isfinite(float(lm.nll(ids))))


if __name__ == "__main__":
    unittest.main()
