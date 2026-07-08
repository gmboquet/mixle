"""TP/PP/CP for :class:`~mixle.models.transformer.CausalLM`, atop the existing FSDP2 support (F1).

``mixle/models/transformer.py`` names the destination directly: "At frontier scale the same module is
what a vendored TorchTitan/Megatron trainer shards (FSDP2/TP/PP)." ``torch_neural.py`` already gives the
data-parallel dimension (DDP on CPU, FSDP2/ZeRO-3 on CUDA). This module adds the three ORTHOGONAL sharding
dimensions a frontier trainer composes with FSDP2 -- "N-D parallelism": FSDP2 shards params/optimizer
state across the data-parallel group while TP/PP/CP further shard the MODEL and the SEQUENCE across
independent device groups:

* **TP** (:class:`ColumnParallelLinear` / :class:`RowParallelLinear`, :func:`tp_shard_causal_lm`) -- splits
  ``CausalAttention``'s ``qkv``/``proj`` and the MLP's two ``Linear`` layers across ``tp_size`` ranks
  (Megatron-style: column-parallel then row-parallel, so exactly one all-reduce per sublayer), by HEAD for
  attention (each rank owns whole heads, never a fraction of one) and by hidden-unit block for the MLP.
* **PP** (:func:`pp_partition_causal_lm`, :func:`pipeline_forward`) -- splits ``model.blocks`` into
  ``pp_size`` contiguous stages (stage 0 also owns the embeddings, the last stage also owns
  ``ln``/``head``), and runs a GPipe-style microbatched pipeline: stages are threads connected by
  queues, so microbatches genuinely overlap in flight across "devices" (this repo's existing
  thread-based distributed-simulation pattern -- see ``multiprocessing.py`` / ``mpi.py``).
* **CP** (:func:`cp_shard_sequence`, :func:`cp_forward_causal_lm`) -- splits the SEQUENCE into
  ``cp_size`` contiguous chunks. Token/position embeddings, ``LayerNorm``, the MLP, and the LM head are
  all per-position and need no communication; only attention needs the other chunks' K/V, so each rank
  computes its local K/V, all-gathers everyone else's (one collective per block), and runs LOCAL causal
  attention with an explicit offset mask (its query positions against the FULL key sequence). This is the
  "simpler chunked approach" the roadmap calls out as an acceptable scope cut vs. incremental ring-attention:
  it reconstructs bit-for-bit-equivalent output (same collective volume as ring attention, just gathered
  up front instead of streamed rank-to-rank) and is exact and testable without incremental overlap.

None of this touches real multi-GPU: there are no 512 A100s in this environment (or in CI), so the
roadmap's "70B-config across >=512 GPUs at published-comparable MFU" acceptance number is NOT measured
here and cannot honestly be claimed from a laptop/CI run -- see the test module's docstring for what IS
verified (exact-match correctness of the TP/PP/CP mechanism at small scale). What's built here is the
real sharding/reconstruction MATH, following the structure a TorchTitan integration would slot into
(``tp_size``/``pp_size``/``cp_size`` device-mesh axes orthogonal to FSDP2's data-parallel axis); a full
TorchTitan integration would additionally need: real multi-GPU process groups per axis (NCCL, not the
in-process simulation here), overlap of TP's all-reduce with compute, 1F1B (not GPipe fill-drain)
pipeline scheduling, and incremental ring-attention communication for CP's memory profile at long context.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False


# ===================================================================================================
# TP -- tensor parallelism: split individual Linear weight matrices across ranks
# ===================================================================================================
if _HAS_TORCH:

    @dataclass
    class ColumnParallelLinear:
        """A ``Linear``'s OUTPUT dimension split across ranks; reconstruction is a concat (all-gather).

        ``weight[r]`` is a contiguous row-block of the dense weight (``out_features`` split into
        ``n_ranks`` chunks); ``bias[r]`` the matching bias slice (or ``None``). Each rank's local
        matmul ``x @ weight[r].T + bias[r]`` is exactly the corresponding output slice of the dense
        layer, so concatenating the ranks' outputs along the last dim reconstructs the dense output.
        """

        weight: list[Any]
        bias: list[Any] | None

        @classmethod
        def shard(cls, linear: nn.Linear, n_ranks: int) -> ColumnParallelLinear:
            w = torch.chunk(linear.weight.detach(), n_ranks, dim=0)
            b = torch.chunk(linear.bias.detach(), n_ranks, dim=0) if linear.bias is not None else None
            return cls(weight=list(w), bias=(list(b) if b is not None else None))

        def forward_shard(self, x: Any, rank: int) -> Any:
            b = self.bias[rank] if self.bias is not None else None
            return F.linear(x, self.weight[rank], b)

        def forward(self, x: Any) -> Any:
            """Reference/non-distributed reconstruction: run every shard and all-gather (concat)."""
            return torch.cat([self.forward_shard(x, r) for r in range(len(self.weight))], dim=-1)

    @dataclass
    class RowParallelLinear:
        """A ``Linear``'s INPUT dimension split across ranks; reconstruction is a sum (all-reduce).

        ``weight[r]`` is a contiguous column-block of the dense weight (``in_features`` split into
        ``n_ranks`` chunks). Each rank's local matmul against ITS input slice sums, across ranks, to
        the dense output; the bias is carried by rank 0 only (added once) so the sum stays exact.
        """

        weight: list[Any]
        bias: Any | None  # carried by rank 0 only

        @classmethod
        def shard(cls, linear: nn.Linear, n_ranks: int) -> RowParallelLinear:
            w = torch.chunk(linear.weight.detach(), n_ranks, dim=1)
            b = linear.bias.detach() if linear.bias is not None else None
            return cls(weight=list(w), bias=b)

        def forward_shard(self, x_shard: Any, rank: int) -> Any:
            out = F.linear(x_shard, self.weight[rank])
            if rank == 0 and self.bias is not None:
                out = out + self.bias
            return out

        def forward(self, x_shards: list[Any]) -> Any:
            """Reference/non-distributed reconstruction: sum every shard's partial output (all-reduce)."""
            parts = [self.forward_shard(x_shards[r], r) for r in range(len(self.weight))]
            out = parts[0]
            for p in parts[1:]:
                out = out + p
            return out

    def _qkv_head_row_groups(n_head: int, tp_size: int) -> list[list[int]]:
        """Row indices into a ``(3*d_model, d_model)`` qkv weight for each rank's WHOLE heads.

        ``qkv``'s output is laid out ``[q_heads..., k_heads..., v_heads...]`` (the reshape in
        ``CausalAttention.forward`` is ``(3, h, d//h)``, row-major -- q/k/v are the outermost block, then
        head, then head-dim). A rank must own the same head index in q, k, AND v (attention needs all
        three for the heads it computes), so the row groups are non-contiguous slices across the three
        q/k/v blocks -- this returns, per rank, the full list of output rows it owns.
        """
        assert n_head % tp_size == 0, "n_head must be divisible by tp_size"
        heads_per_rank = n_head // tp_size
        groups: list[list[int]] = []
        for r in range(tp_size):
            head_ids = range(r * heads_per_rank, (r + 1) * heads_per_rank)
            groups.append(list(head_ids))
        return groups

    @dataclass
    class TPAttentionShard:
        """One rank's shard of a ``CausalAttention``: whole heads of qkv (column) + matching proj rows (row)."""

        n_head_local: int
        qkv_weight: Any
        qkv_bias: Any
        proj_weight: Any  # (d_model, head_dim * n_head_local) -- row-parallel input block
        proj_bias: Any | None  # rank 0 only

    def tp_shard_attention(attn: nn.Module, tp_size: int) -> list[TPAttentionShard]:
        """Shard a :class:`~mixle.models.transformer.CausalAttention` into ``tp_size`` head-parallel ranks."""
        h = attn.h
        d_model = attn.qkv.in_features
        dh = d_model // h
        groups = _qkv_head_row_groups(h, tp_size)
        shards = []
        for r, head_ids in enumerate(groups):
            row_idx: list[int] = []
            for qkv_block in range(3):
                base = qkv_block * h * dh
                for hid in head_ids:
                    row_idx.extend(range(base + hid * dh, base + hid * dh + dh))
            idx = torch.as_tensor(row_idx, dtype=torch.long)
            qkv_w = attn.qkv.weight.detach().index_select(0, idx)
            qkv_b = attn.qkv.bias.detach().index_select(0, idx) if attn.qkv.bias is not None else None
            col_idx = torch.as_tensor([hid * dh + k for hid in head_ids for k in range(dh)], dtype=torch.long)
            proj_w = attn.proj.weight.detach().index_select(1, col_idx)
            proj_b = attn.proj.bias.detach() if (r == 0 and attn.proj.bias is not None) else None
            shards.append(
                TPAttentionShard(
                    n_head_local=len(head_ids), qkv_weight=qkv_w, qkv_bias=qkv_b, proj_weight=proj_w, proj_bias=proj_b
                )
            )
        return shards

    def tp_attention_forward(x: Any, shards: list[TPAttentionShard]) -> Any:
        """Run head-parallel attention across the (simulated) ranks and reconstruct the dense output.

        Each rank: local qkv projection (its whole heads only) -> local causal attention -> partial
        ``(b, t, head_dim * n_head_local)`` activation. All-gather (concat, in rank order == head order)
        reconstructs the ``o`` the dense ``CausalAttention`` would compute; the row-parallel ``proj`` then
        sums the ranks' partial output projections (all-reduce) plus rank 0's bias -- exactly the dense
        ``proj(o)``.
        """
        b, t, _ = x.shape
        outs = []
        for sh in shards:
            qkv = F.linear(x, sh.qkv_weight, sh.qkv_bias)
            qkv = qkv.reshape(b, t, 3, sh.n_head_local, -1).permute(2, 0, 3, 1, 4)
            o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
            outs.append(o.transpose(1, 2).reshape(b, t, -1))
        parts = [F.linear(outs[r], sh.proj_weight) for r, sh in enumerate(shards)]
        out = parts[0] + (shards[0].proj_bias if shards[0].proj_bias is not None else 0.0)
        for p in parts[1:]:
            out = out + p
        return out

    @dataclass
    class TPBlockShard:
        attn: list[TPAttentionShard]  # per-rank attention shard (shared across the block's ranks)
        mlp_fc1: ColumnParallelLinear
        mlp_fc2: RowParallelLinear

    def tp_shard_block(block: nn.Module, tp_size: int) -> TPBlockShard:
        fc1, _, fc2 = block.mlp[0], block.mlp[1], block.mlp[2]
        return TPBlockShard(
            attn=tp_shard_attention(block.attn, tp_size),
            mlp_fc1=ColumnParallelLinear.shard(fc1, tp_size),
            mlp_fc2=RowParallelLinear.shard(fc2, tp_size),
        )

    def tp_block_forward(x: Any, ln1: nn.Module, ln2: nn.Module, shard: TPBlockShard) -> Any:
        x = x + tp_attention_forward(ln1(x), shard.attn)
        h = ln2(x)
        gelu_shards = [F.gelu(shard.mlp_fc1.forward_shard(h, r)) for r in range(len(shard.mlp_fc1.weight))]
        return x + shard.mlp_fc2.forward(gelu_shards)

    @dataclass
    class TPCausalLMShard:
        blocks: list[TPBlockShard]
        tp_size: int

    def tp_shard_causal_lm(model: nn.Module, tp_size: int) -> TPCausalLMShard:
        """Shard every block of a :class:`~mixle.models.transformer.CausalLM` for ``tp_size``-way TP.

        Token/position embeddings and the final ``ln``/``head`` are NOT sharded here (they are cheap
        relative to attention/MLP and, per Megatron, are typically the sequence-/vocab-parallel dimension
        rather than TP proper) -- this covers the attention+MLP sharding the spec calls out explicitly.
        """
        return TPCausalLMShard(blocks=[tp_shard_block(blk, tp_size) for blk in model.blocks], tp_size=tp_size)

    def tp_forward_causal_lm(model: nn.Module, x: Any, tp_shard: TPCausalLMShard) -> Any:
        """Forward an input through the TP-sharded blocks (attention/MLP), embeddings/head run dense."""
        x = x.long()
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        h = model.tok(x) + model.pos(pos)[None, :, :]
        for blk, shard in zip(model.blocks, tp_shard.blocks):
            h = tp_block_forward(h, blk.ln1, blk.ln2, shard)
        return model.head(model.ln(h))[:, -1]


# ===================================================================================================
# PP -- pipeline parallelism: split model.blocks into stages, microbatch across simulated "devices"
# ===================================================================================================
if _HAS_TORCH:

    class PPStage(nn.Module):
        """One pipeline stage: a contiguous slice of ``model.blocks``, optionally with embeddings and/or
        the final ``ln``/``head`` (stage 0 embeds, the last stage projects to logits)."""

        def __init__(
            self, blocks: list[nn.Module], *, tok: Any = None, pos: Any = None, ln: Any = None, head: Any = None
        ) -> None:
            super().__init__()
            self.stage_blocks = nn.ModuleList(blocks)
            self.tok = tok
            self.pos = pos
            self.ln = ln
            self.head = head

        def forward(self, x: Any) -> Any:
            if self.tok is not None:
                x = x.long()
                t = x.shape[1]
                pos_ids = torch.arange(t, device=x.device)
                h = self.tok(x) + self.pos(pos_ids)[None, :, :]
            else:
                h = x
            for blk in self.stage_blocks:
                h = blk(h)
            if self.head is not None:
                h = self.head(self.ln(h))[:, -1]  # next-token logits from the last position -> (batch, vocab)
            return h

    def pp_partition_causal_lm(model: nn.Module, pp_size: int) -> list[PPStage]:
        """Split ``model.blocks`` into ``pp_size`` contiguous stages (GPipe-style layer partition).

        Stage 0 additionally owns the token/position embeddings; the LAST stage additionally owns the
        final ``ln``/``head`` -- so stage 0 takes raw token ids and the last stage emits logits, and
        every intermediate stage is a pure activation-in/activation-out block group (what gets pipelined).
        """
        n = len(model.blocks)
        assert n >= pp_size >= 1, "pp_size must be between 1 and the number of blocks"
        base, rem = divmod(n, pp_size)
        stages, start = [], 0
        for i in range(pp_size):
            size = base + (1 if i < rem else 0)
            blocks = list(model.blocks[start : start + size])
            stages.append(
                PPStage(
                    blocks,
                    tok=model.tok if i == 0 else None,
                    pos=model.pos if i == 0 else None,
                    ln=model.ln if i == pp_size - 1 else None,
                    head=model.head if i == pp_size - 1 else None,
                )
            )
            start += size
        return stages

    def pipeline_forward(stages: list[PPStage], x: Any, n_microbatches: int) -> Any:
        """GPipe-style microbatched pipeline: split ``x``'s batch dim, run stages as threads-with-queues.

        Each stage is a thread reading its input queue and writing to the next stage's; the driver feeds
        microbatches into stage 0's queue back-to-back (no waiting for one to finish before starting the
        next), so microbatches genuinely overlap in flight across stages -- the "devices" this repo's
        existing thread-based distributed-simulation tests stand in for real ranks with (see
        ``multiprocessing.py``). Since every op here (LayerNorm, attention, MLP, embeddings) is
        batch-independent, splitting the batch into microbatches and reassembling in order is exactly
        equivalent to running the whole batch through the un-partitioned model.
        """
        b = x.shape[0]
        assert n_microbatches >= 1
        chunks = list(torch.chunk(x, n_microbatches, dim=0)) if b >= n_microbatches else [x]
        n_stages = len(stages)
        qs: list[queue.Queue] = [queue.Queue() for _ in range(n_stages + 1)]
        errors: list[BaseException] = []

        def run_stage(i: int) -> None:
            stage = stages[i]
            while True:
                item = qs[i].get()
                if item is None:
                    qs[i + 1].put(None)
                    return
                idx, tensor = item
                try:
                    with torch.no_grad():
                        out = stage(tensor)
                except BaseException as exc:  # noqa: BLE001 - surface on the driver thread
                    errors.append(exc)
                    qs[i + 1].put(None)
                    return
                qs[i + 1].put((idx, out))

        threads = [threading.Thread(target=run_stage, args=(i,), daemon=True) for i in range(n_stages)]
        for th in threads:
            th.start()
        for idx, chunk in enumerate(chunks):
            qs[0].put((idx, chunk))
        qs[0].put(None)

        results: dict[int, Any] = {}
        seen_sentinel = False
        while len(results) < len(chunks):
            item = qs[n_stages].get()
            if item is None:
                seen_sentinel = True
                break
            idx, out = item
            results[idx] = out
        for th in threads:
            th.join(timeout=30)
        if errors:
            raise errors[0]
        if not seen_sentinel and len(results) < len(chunks):  # pragma: no cover - defensive
            raise RuntimeError("pipeline_forward: stage threads exited before producing all microbatches")
        ordered = [results[i] for i in range(len(chunks))]
        return torch.cat(ordered, dim=0)


# ===================================================================================================
# CP -- context (sequence) parallelism: split the sequence, reconcile attention via a K/V all-gather
# ===================================================================================================
if _HAS_TORCH:

    def cp_shard_sequence(x: Any, cp_size: int) -> list[Any]:
        """Split a ``(batch, seq)`` (or ``(batch, seq, ...)``) tensor into ``cp_size`` contiguous sequence
        chunks along dim 1 -- each rank keeps one chunk resident (never materializes the full sequence)."""
        return list(torch.chunk(x, cp_size, dim=1))

    def _cp_causal_mask(q_start: int, q_len: int, k_len: int, device: Any) -> Any:
        """Boolean ``(q_len, k_len)`` mask: query at absolute position ``q_start + i`` may attend to key
        position ``j`` iff ``j <= q_start + i`` -- the causal rule, but for a Q chunk that starts partway
        through the sequence and a K range covering the WHOLE sequence up to (and including) that chunk."""
        q_pos = torch.arange(q_start, q_start + q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        return k_pos <= q_pos

    def cp_attention_forward(attn: nn.Module, chunks: list[Any]) -> list[Any]:
        """Context-parallel attention: each rank computes local Q/K/V, all-gathers K/V (one collective),
        then runs LOCAL causal attention of its Q chunk against the FULL (gathered) K/V with an explicit
        offset causal mask. Returns the per-rank output chunks (concat along seq to reconstruct the dense
        ``CausalAttention`` output) -- this is the "simpler chunked" CP scope noted in the module
        docstring: same total K/V communication volume as ring attention, gathered eagerly instead of
        streamed incrementally rank-to-rank (that overlap is the piece a real ring-attention CP would add).
        """
        h, d_model = attn.h, attn.qkv.in_features
        dh = d_model // h
        cp_size = len(chunks)
        local_qkv = []
        for c in chunks:
            b, t, _ = c.shape
            qkv = attn.qkv(c).reshape(b, t, 3, h, dh).permute(2, 0, 3, 1, 4)  # (3, b, h, t, dh)
            local_qkv.append(qkv)
        full_k = torch.cat([qkv[1] for qkv in local_qkv], dim=2)  # all-gather K along seq -> (b, h, T, dh)
        full_v = torch.cat([qkv[2] for qkv in local_qkv], dim=2)  # all-gather V along seq
        outs = []
        q_start = 0
        for qkv in local_qkv:
            q = qkv[0]
            q_len = q.shape[2]
            mask = _cp_causal_mask(q_start, q_len, full_k.shape[2], q.device)
            o = F.scaled_dot_product_attention(q, full_k, full_v, attn_mask=mask)
            outs.append(attn.proj(o.transpose(1, 2).reshape(q.shape[0], q_len, -1)))
            q_start += q_len
        return outs

    def cp_forward_causal_lm(model: nn.Module, x: Any, cp_size: int) -> Any:
        """Full CP forward: per-block, only attention needs the K/V all-gather -- embeddings, LayerNorm,
        MLP, and the LM head are all per-position (no communication) and run locally on each chunk.
        Returns per-position logits for the WHOLE sequence (``(batch, seq, vocab)``), reconstructed by
        concatenating the ranks' chunks -- so CP correctness is checked at every position, not just last.
        """
        x = x.long()
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        h_full = model.tok(x) + model.pos(pos)[None, :, :]
        chunks = cp_shard_sequence(h_full, cp_size)
        for blk in model.blocks:
            ln1_chunks = [blk.ln1(c) for c in chunks]
            attn_out = cp_attention_forward(blk.attn, ln1_chunks)
            chunks = [c + a for c, a in zip(chunks, attn_out)]
            chunks = [c + blk.mlp(blk.ln2(c)) for c in chunks]
        logits = [model.head(model.ln(c)) for c in chunks]
        return torch.cat(logits, dim=1)


# ===================================================================================================
# lm.fit(distributed=True, tp_size=..., pp_size=..., cp_size=...) plan validation
# ===================================================================================================
if _HAS_TORCH:

    def validate_tp_pp_cp_plan(model: nn.Module, tp_size: int = 1, pp_size: int = 1, cp_size: int = 1) -> None:
        """Validate a ``(tp_size, pp_size, cp_size)`` plan against a real :class:`CausalLM`'s dimensions.

        Raises ``ValueError`` with an actionable message if the plan does not divide the model cleanly --
        the same checks :func:`tp_shard_causal_lm` / :func:`pp_partition_causal_lm` / :func:`cp_shard_sequence`
        enforce structurally, surfaced up front so ``lm.fit(distributed=True, ...)`` fails fast on a bad
        plan instead of partway through a run. This is the plan-construction half of the ``tp_size``/
        ``pp_size``/``cp_size`` knobs on :meth:`~mixle.models.language_model.LM.fit`; wiring the validated
        plan into per-axis NCCL process groups (real SPMD TP/PP/CP execution, composed with the existing
        FSDP2 data-parallel group) is the multi-GPU piece this environment cannot exercise -- see the
        module docstring and ``torch_neural.py``'s FSDP2 CUDA branch, which carries the identical caveat
        ("correct per the API, only exercised on multi-GPU").
        """
        tp_size, pp_size, cp_size = int(tp_size), int(pp_size), int(cp_size)
        if tp_size < 1 or pp_size < 1 or cp_size < 1:
            raise ValueError("tp_size/pp_size/cp_size must be >= 1, got %r" % ((tp_size, pp_size, cp_size),))
        n_head = int(model.n_head)
        n_layer = len(model.blocks)
        block = int(model.block)
        if n_head % tp_size:
            raise ValueError("tp_size=%d must divide n_head=%d evenly" % (tp_size, n_head))
        if pp_size > n_layer:
            raise ValueError("pp_size=%d cannot exceed n_layer=%d" % (pp_size, n_layer))
        if block % cp_size:
            raise ValueError("cp_size=%d must divide block=%d evenly" % (cp_size, block))


__all__ = [
    "validate_tp_pp_cp_plan",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TPAttentionShard",
    "TPBlockShard",
    "TPCausalLMShard",
    "tp_shard_attention",
    "tp_attention_forward",
    "tp_shard_block",
    "tp_block_forward",
    "tp_shard_causal_lm",
    "tp_forward_causal_lm",
    "PPStage",
    "pp_partition_causal_lm",
    "pipeline_forward",
    "cp_shard_sequence",
    "cp_attention_forward",
    "cp_forward_causal_lm",
]
