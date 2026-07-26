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
  computes local Q/K/V and streams preceding K/V blocks through an online-softmax accumulator. The
  in-process reference holds all simulated ranks, but no rank's attention calculation concatenates a
  full-sequence K/V tensor; the returned receipt records that memory invariant.

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
from numbers import Integral
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

    class ColumnParallelLinear(nn.Module):
        """A ``Linear``'s OUTPUT dimension split across ranks; reconstruction is a concat (all-gather).

        ``weight[r]`` is a contiguous row-block of the dense weight (``out_features`` split into
        ``n_ranks`` chunks); ``bias[r]`` the matching bias slice (or ``None``). Each rank's local
        matmul ``x @ weight[r].T + bias[r]`` is exactly the corresponding output slice of the dense
        layer, so concatenating the ranks' outputs along the last dim reconstructs the dense output.
        """

        def __init__(self, weight: list[Any], bias: list[Any] | None) -> None:
            super().__init__()
            if not weight:
                raise ValueError("column-parallel weight must contain at least one shard.")
            self.weight = nn.ParameterList([nn.Parameter(shard.detach().clone()) for shard in weight])
            self.bias = (
                nn.ParameterList([nn.Parameter(shard.detach().clone()) for shard in bias]) if bias is not None else None
            )
            if self.bias is not None and len(self.bias) != len(self.weight):
                raise ValueError("column-parallel bias must contain exactly one slice per weight shard.")

        @classmethod
        def shard(cls, linear: nn.Linear, n_ranks: int) -> ColumnParallelLinear:
            n_ranks = _positive_parallel_size(n_ranks, "n_ranks")
            if linear.out_features % n_ranks:
                raise ValueError("n_ranks must divide linear.out_features exactly.")
            w = torch.chunk(linear.weight, n_ranks, dim=0)
            b = torch.chunk(linear.bias, n_ranks, dim=0) if linear.bias is not None else None
            return cls(weight=list(w), bias=(list(b) if b is not None else None))

        def forward_shard(self, x: Any, rank: int) -> Any:
            rank = _rank_index(rank, len(self.weight))
            b = self.bias[rank] if self.bias is not None else None
            return F.linear(x, self.weight[rank], b)

        def forward(self, x: Any) -> Any:
            """Reference/non-distributed reconstruction: run every shard and all-gather (concat)."""
            return torch.cat([self.forward_shard(x, r) for r in range(len(self.weight))], dim=-1)

    class RowParallelLinear(nn.Module):
        """A ``Linear``'s INPUT dimension split across ranks; reconstruction is a sum (all-reduce).

        ``weight[r]`` is a contiguous column-block of the dense weight (``in_features`` split into
        ``n_ranks`` chunks). Each rank's local matmul against ITS input slice sums, across ranks, to
        the dense output; the bias is carried by rank 0 only (added once) so the sum stays exact.
        """

        def __init__(self, weight: list[Any], bias: Any | None) -> None:
            super().__init__()
            if not weight:
                raise ValueError("row-parallel weight must contain at least one shard.")
            self.weight = nn.ParameterList([nn.Parameter(shard.detach().clone()) for shard in weight])
            self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

        @classmethod
        def shard(cls, linear: nn.Linear, n_ranks: int) -> RowParallelLinear:
            n_ranks = _positive_parallel_size(n_ranks, "n_ranks")
            if linear.in_features % n_ranks:
                raise ValueError("n_ranks must divide linear.in_features exactly.")
            w = torch.chunk(linear.weight, n_ranks, dim=1)
            b = linear.bias if linear.bias is not None else None
            return cls(weight=list(w), bias=b)

        def forward_shard(self, x_shard: Any, rank: int) -> Any:
            rank = _rank_index(rank, len(self.weight))
            out = F.linear(x_shard, self.weight[rank])
            if rank == 0 and self.bias is not None:
                out = out + self.bias
            return out

        def forward(self, x_shards: list[Any]) -> Any:
            """Reference/non-distributed reconstruction: sum every shard's partial output (all-reduce)."""
            if not isinstance(x_shards, list) or len(x_shards) != len(self.weight):
                raise ValueError(f"x_shards must contain exactly {len(self.weight)} rank-local tensors.")
            parts = [self.forward_shard(x_shards[r], r) for r in range(len(self.weight))]
            out = parts[0]
            for p in parts[1:]:
                out = out + p
            return out

    def _positive_parallel_size(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive exact integer.")
        return int(value)

    def _rank_index(value: Any, n_ranks: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) < n_ranks:
            raise ValueError(f"rank must be an exact integer in [0, {n_ranks}).")
        return int(value)

    def _qkv_head_row_groups(n_head: int, tp_size: int) -> list[list[int]]:
        """Row indices into a ``(3*d_model, d_model)`` qkv weight for each rank's WHOLE heads.

        ``qkv``'s output is laid out ``[q_heads..., k_heads..., v_heads...]`` (the reshape in
        ``CausalAttention.forward`` is ``(3, h, d//h)``, row-major -- q/k/v are the outermost block, then
        head, then head-dim). A rank must own the same head index in q, k, AND v (attention needs all
        three for the heads it computes), so the row groups are non-contiguous slices across the three
        q/k/v blocks -- this returns, per rank, the full list of output rows it owns.
        """
        tp_size = _positive_parallel_size(tp_size, "tp_size")
        if n_head % tp_size:
            raise ValueError("n_head must be divisible by tp_size.")
        heads_per_rank = n_head // tp_size
        groups: list[list[int]] = []
        for r in range(tp_size):
            head_ids = range(r * heads_per_rank, (r + 1) * heads_per_rank)
            groups.append(list(head_ids))
        return groups

    class TPAttentionShard(nn.Module):
        """One rank's shard of a ``CausalAttention``: whole heads of qkv (column) + matching proj rows (row)."""

        def __init__(
            self,
            *,
            n_head_local: int,
            qkv_weight: Any,
            qkv_bias: Any,
            proj_weight: Any,
            proj_bias: Any | None,
        ) -> None:
            super().__init__()
            self.n_head_local = n_head_local
            self.qkv_weight = nn.Parameter(qkv_weight.detach().clone())
            self.qkv_bias = nn.Parameter(qkv_bias.detach().clone()) if qkv_bias is not None else None
            self.proj_weight = nn.Parameter(proj_weight.detach().clone())
            self.proj_bias = nn.Parameter(proj_bias.detach().clone()) if proj_bias is not None else None

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
            idx = torch.as_tensor(row_idx, device=attn.qkv.weight.device, dtype=torch.long)
            qkv_w = attn.qkv.weight.index_select(0, idx)
            qkv_b = attn.qkv.bias.index_select(0, idx) if attn.qkv.bias is not None else None
            col_idx = torch.as_tensor(
                [hid * dh + k for hid in head_ids for k in range(dh)],
                device=attn.proj.weight.device,
                dtype=torch.long,
            )
            proj_w = attn.proj.weight.index_select(1, col_idx)
            proj_b = attn.proj.bias if (r == 0 and attn.proj.bias is not None) else None
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
        if not isinstance(shards, (list, nn.ModuleList)) or not shards:
            raise ValueError("shards must contain at least one TPAttentionShard.")
        if any(not isinstance(shard, TPAttentionShard) for shard in shards):
            raise TypeError("every attention shard must be a TPAttentionShard.")
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

    class TPBlockShard(nn.Module):
        def __init__(
            self,
            *,
            attn: list[TPAttentionShard],
            mlp_fc1: ColumnParallelLinear,
            mlp_fc2: RowParallelLinear,
        ) -> None:
            super().__init__()
            if len(attn) != len(mlp_fc1.weight) or len(attn) != len(mlp_fc2.weight):
                raise ValueError("every TP block component must contain the same number of rank shards.")
            self.attn = nn.ModuleList(attn)
            self.mlp_fc1 = mlp_fc1
            self.mlp_fc2 = mlp_fc2
            self.tp_size = len(attn)

    def tp_shard_block(block: nn.Module, tp_size: int) -> TPBlockShard:
        fc1, _, fc2 = block.mlp[0], block.mlp[1], block.mlp[2]
        return TPBlockShard(
            attn=tp_shard_attention(block.attn, tp_size),
            mlp_fc1=ColumnParallelLinear.shard(fc1, tp_size),
            mlp_fc2=RowParallelLinear.shard(fc2, tp_size),
        )

    def tp_block_forward(x: Any, ln1: nn.Module, ln2: nn.Module, shard: TPBlockShard) -> Any:
        if (
            len(shard.attn) != shard.tp_size
            or len(shard.mlp_fc1.weight) != shard.tp_size
            or len(shard.mlp_fc2.weight) != shard.tp_size
        ):
            raise ValueError("TP block has missing or extra rank-local shards.")
        x = x + tp_attention_forward(ln1(x), shard.attn)
        h = ln2(x)
        gelu_shards = [F.gelu(shard.mlp_fc1.forward_shard(h, r)) for r in range(len(shard.mlp_fc1.weight))]
        return x + shard.mlp_fc2.forward(gelu_shards)

    class TPCausalLMShard(nn.Module):
        def __init__(self, *, blocks: list[TPBlockShard], tp_size: int) -> None:
            super().__init__()
            if any(block.tp_size != tp_size for block in blocks):
                raise ValueError("every TP block must match the plan's tp_size.")
            self.blocks = nn.ModuleList(blocks)
            self.tp_size = tp_size

    def tp_shard_causal_lm(model: nn.Module, tp_size: int) -> TPCausalLMShard:
        """Shard every block of a :class:`~mixle.models.transformer.CausalLM` for ``tp_size``-way TP.

        Token/position embeddings and the final ``ln``/``head`` are NOT sharded here (they are cheap
        relative to attention/MLP and, per Megatron, are typically the sequence-/vocab-parallel dimension
        rather than TP proper) -- this covers the attention+MLP sharding the spec calls out explicitly.
        """
        tp_size = _positive_parallel_size(tp_size, "tp_size")
        return TPCausalLMShard(
            blocks=[tp_shard_block(blk, tp_size) for blk in model.blocks],
            tp_size=tp_size,
        )

    def tp_trainable_parameters(model: nn.Module, tp_shard: TPCausalLMShard) -> list[nn.Parameter]:
        """Return the unique parameters used by :func:`tp_forward_causal_lm`.

        Dense attention/MLP parameters are replaced by the shard-owned copies and therefore excluded;
        embeddings, normalization layers, and the tied output head remain dense and are included once.
        This list is the explicit optimizer ownership contract for the in-process trainable reference.
        """
        if not isinstance(tp_shard, TPCausalLMShard) or len(tp_shard.blocks) != len(model.blocks):
            raise ValueError("tp_shard must match the model's block count.")
        replaced = {
            id(parameter)
            for block in model.blocks
            for module in (block.attn, block.mlp)
            for parameter in module.parameters()
        }
        parameters = [parameter for parameter in model.parameters() if id(parameter) not in replaced]
        parameters.extend(tp_shard.parameters())
        if len({id(parameter) for parameter in parameters}) != len(parameters):
            raise RuntimeError("TP optimizer ownership contains duplicate parameters.")
        return parameters

    def tp_forward_causal_lm(model: nn.Module, x: Any, tp_shard: TPCausalLMShard) -> Any:
        """Forward an input through the TP-sharded blocks (attention/MLP), embeddings/head run dense."""
        x = x.long()
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        h = model.tok(x) + model.pos(pos)[None, :, :]
        if len(tp_shard.blocks) != len(model.blocks):
            raise ValueError("TP shard must contain exactly one shard group per model block.")
        for blk, shard in zip(model.blocks, tp_shard.blocks, strict=True):
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
        pp_size = _positive_parallel_size(pp_size, "pp_size")
        if pp_size > n:
            raise ValueError("pp_size must not exceed the number of model blocks.")
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
        if not isinstance(stages, list) or not stages or any(not isinstance(stage, PPStage) for stage in stages):
            raise ValueError("stages must be a non-empty list of PPStage modules.")
        if stages[0].tok is None or stages[0].pos is None:
            raise ValueError("the first pipeline stage must own token and position embeddings.")
        if stages[-1].ln is None or stages[-1].head is None:
            raise ValueError("the last pipeline stage must own final normalization and output head.")
        if any(stage.tok is not None or stage.pos is not None for stage in stages[1:]):
            raise ValueError("only the first pipeline stage may own embeddings.")
        if any(stage.ln is not None or stage.head is not None for stage in stages[:-1]):
            raise ValueError("only the last pipeline stage may own the final normalization and head.")
        if not torch.is_tensor(x) or x.ndim < 1 or x.shape[0] == 0:
            raise ValueError("x must be a non-empty tensor with a batch dimension.")
        b = x.shape[0]
        n_microbatches = _positive_parallel_size(n_microbatches, "n_microbatches")
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
# CP -- context parallelism: split the sequence, stream K/V blocks through online softmax
# ===================================================================================================
if _HAS_TORCH:

    def cp_shard_sequence(x: Any, cp_size: int) -> list[Any]:
        """Split a ``(batch, seq)`` (or ``(batch, seq, ...)``) tensor into ``cp_size`` contiguous sequence
        chunks along dim 1."""
        if not torch.is_tensor(x) or x.ndim < 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("x must be a non-empty tensor with batch and sequence dimensions.")
        cp_size = _positive_parallel_size(cp_size, "cp_size")
        if cp_size > x.shape[1]:
            raise ValueError("cp_size cannot exceed the sequence length.")
        chunks = list(torch.tensor_split(x, cp_size, dim=1))
        if len(chunks) != cp_size or any(chunk.shape[1] == 0 for chunk in chunks):
            raise RuntimeError("context sharding did not produce exactly one non-empty chunk per rank.")
        return chunks

    @dataclass(frozen=True)
    class CPAttentionReceipt:
        algorithm: str
        full_sequence_tokens: int
        peak_key_block_tokens_per_rank: int
        materialized_full_kv_per_rank: bool
        streamed_kv_blocks: int

    def _streaming_causal_attention(q: Any, local_qkv: list[Any], query_rank: int) -> Any:
        """Exact online-softmax attention over preceding K/V blocks without concatenating them."""
        scale = q.shape[-1] ** -0.5
        running_max = q.new_full(q.shape[:-1], float("-inf"))
        running_denominator = q.new_zeros(q.shape[:-1])
        running_numerator = q.new_zeros(q.shape)
        for source_rank in range(query_rank + 1):
            key = local_qkv[source_rank][1]
            value = local_qkv[source_rank][2]
            logits = torch.einsum("bhqd,bhkd->bhqk", q, key) * scale
            if source_rank == query_rank:
                local_mask = (
                    torch.arange(key.shape[2], device=q.device)[None, :]
                    <= torch.arange(
                        q.shape[2],
                        device=q.device,
                    )[:, None]
                )
                logits = logits.masked_fill(~local_mask[None, None], float("-inf"))
            block_max = logits.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            block_weights = torch.exp(logits - new_max.unsqueeze(-1))
            running_numerator = running_numerator * old_scale.unsqueeze(-1) + torch.einsum(
                "bhqk,bhkd->bhqd",
                block_weights,
                value,
            )
            running_denominator = running_denominator * old_scale + block_weights.sum(dim=-1)
            running_max = new_max
        if not bool(torch.isfinite(running_numerator).all()) or not bool(torch.isfinite(running_denominator).all()):
            raise FloatingPointError("context-parallel online softmax produced non-finite state.")
        if bool((running_denominator <= 0).any()):
            raise FloatingPointError("context-parallel online softmax has a non-positive denominator.")
        return running_numerator / running_denominator.unsqueeze(-1)

    def cp_attention_forward(
        attn: nn.Module,
        chunks: list[Any],
        *,
        return_receipt: bool = False,
    ) -> Any:
        """Context-parallel attention with block-streaming exact online softmax.

        Each simulated rank computes local Q/K/V and consumes only the preceding rank-local K/V blocks.
        It retains a query-sized numerator/denominator and one K/V block at a time, never a concatenated
        full-sequence K/V tensor. ``return_receipt=True`` returns the explicit memory/collective receipt.
        """
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a non-empty list of rank-local tensors.")
        if any(
            not torch.is_tensor(chunk)
            or chunk.ndim != 3
            or chunk.shape[0] != chunks[0].shape[0]
            or chunk.shape[2] != chunks[0].shape[2]
            or chunk.device != chunks[0].device
            for chunk in chunks
        ):
            raise ValueError("all context chunks must have compatible non-empty (batch, sequence, model) shape.")
        if any(chunk.shape[1] == 0 for chunk in chunks):
            raise ValueError("context chunks must have non-empty local sequence dimensions.")
        if not isinstance(return_receipt, bool):
            raise ValueError("return_receipt must be boolean.")
        h, d_model = attn.h, attn.qkv.in_features
        dh = d_model // h
        local_qkv = []
        for c in chunks:
            b, t, _ = c.shape
            qkv = attn.qkv(c).reshape(b, t, 3, h, dh).permute(2, 0, 3, 1, 4)  # (3, b, h, t, dh)
            local_qkv.append(qkv)
        outs = []
        for query_rank, qkv in enumerate(local_qkv):
            q = qkv[0]
            q_len = q.shape[2]
            o = _streaming_causal_attention(q, local_qkv, query_rank)
            outs.append(attn.proj(o.transpose(1, 2).reshape(q.shape[0], q_len, -1)))
        receipt = CPAttentionReceipt(
            algorithm="block_streaming_online_softmax",
            full_sequence_tokens=sum(chunk.shape[1] for chunk in chunks),
            peak_key_block_tokens_per_rank=max(chunk.shape[1] for chunk in chunks),
            materialized_full_kv_per_rank=False,
            streamed_kv_blocks=sum(rank + 1 for rank in range(len(chunks))),
        )
        return (outs, receipt) if return_receipt else outs

    def cp_forward_causal_lm(model: nn.Module, x: Any, cp_size: int) -> Any:
        """Full CP forward: per-block, only attention streams remote K/V blocks -- embeddings, LayerNorm,
        MLP, and the LM head are all per-position and run locally on each chunk.
        Returns per-position logits for the WHOLE sequence (``(batch, seq, vocab)``), reconstructed by
        concatenating the ranks' chunks -- so CP correctness is checked at every position, not just last.
        """
        x = x.long()
        token_chunks = cp_shard_sequence(x, cp_size)
        chunks = []
        position = 0
        for token_chunk in token_chunks:
            positions = torch.arange(
                position,
                position + token_chunk.shape[1],
                device=x.device,
            )
            chunks.append(model.tok(token_chunk) + model.pos(positions)[None, :, :])
            position += token_chunk.shape[1]
        for blk in model.blocks:
            ln1_chunks = [blk.ln1(c) for c in chunks]
            attn_out = cp_attention_forward(blk.attn, ln1_chunks)
            if len(attn_out) != len(chunks):
                raise RuntimeError("context attention returned the wrong number of rank-local outputs.")
            chunks = [c + a for c, a in zip(chunks, attn_out, strict=True)]
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
    "tp_trainable_parameters",
    "tp_forward_causal_lm",
    "PPStage",
    "pp_partition_causal_lm",
    "pipeline_forward",
    "CPAttentionReceipt",
    "cp_shard_sequence",
    "cp_attention_forward",
    "cp_forward_causal_lm",
]
