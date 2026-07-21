"""A causal decoder-only Transformer as a torch module -- the engine behind the declarative AR-LM surface.

Built lazily by the :class:`mixle.ppl.core.Transformer` predictor token. ``forward(x)`` takes a ``(batch, block)``
context of token ids (accepted as float so it rides the ``SoftmaxNeuralLeaf`` float path, cast to long inside)
and returns next-token logits ``(batch, vocab)`` from the last position. So
``Categorical(logits=Transformer(out=V))`` is *exactly* next-token prediction ``p(token | context)``, fit by the
standard ``estimate()`` loop whose cross-entropy is ``-log p`` -- no new training machinery.

Attention is ``F.scaled_dot_product_attention`` with CUDA FlashAttention dispatch when available. This module runs
single-process here, while larger training stacks can shard the same architecture externally.

The ``nn.Module`` subclasses are defined at MODULE level (not nested inside ``build_causal_lm``) so a trained
LM pickles/saves: a function-local class has no importable qualname and ``torch.save``/``pickle`` cannot find it.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

if _HAS_TORCH:

    class CausalAttention(nn.Module):
        def __init__(self, d_model: int, n_head: int) -> None:
            super().__init__()
            self.h = n_head
            # ``parallelize_module`` shards qkv by output features.  The public
            # forward therefore has to distinguish global from rank-local heads.
            self.tp_size = 1
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.proj = nn.Linear(d_model, d_model)
            # muP attention scaling (see mixle.models.mup): standard attention scales QK^T by
            # 1/sqrt(head_dim); muP (Tensor Programs V, Table 3) instead requires 1/head_dim so the
            # pre-softmax logit scale stays width-independent under the "hidden" role's init/lr rules.
            # Off by default (the standard 1/sqrt(head_dim) scaling); mixle.models.mup.apply_mup_init
            # turns it on for a model being run under muP.
            self.mup_attention = False

        def forward(self, x: Any) -> Any:
            b, t, d = x.shape
            head_dim = d // self.h
            qkv_projection = self.qkv(x)
            local_width = qkv_projection.shape[-1]
            local_heads = local_width // (3 * head_dim)
            if local_heads * 3 * head_dim != local_width:
                raise ValueError("the local qkv width must contain complete attention heads.")
            qkv = qkv_projection.reshape(b, t, 3, local_heads, head_dim).permute(2, 0, 3, 1, 4)
            scale = 1.0 / head_dim if self.mup_attention else None
            o = F.scaled_dot_product_attention(
                qkv[0], qkv[1], qkv[2], is_causal=True, scale=scale
            )  # FlashAttention path (CUDA)
            return self.proj(o.transpose(1, 2).reshape(b, t, d))

    class Block(nn.Module):
        def __init__(self, d_model: int, n_head: int) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.attn = CausalAttention(d_model, n_head)
            self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

        def forward(self, x: Any) -> Any:
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class CausalLM(nn.Module):
        def __init__(
            self, vocab: int, d_model: int, n_layer: int, n_head: int, block: int, embedding: Any = None
        ) -> None:
            super().__init__()
            # record the shape so a trained module can be rebuilt from hyperparameters on load
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.block = int(block)
            self.tok = embedding if embedding is not None else nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(block, d_model)
            self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying
            # activation (gradient) checkpointing: recompute block activations in backward instead of
            # storing them -- the standard memory/compute trade for long blocks or deep stacks. A plain
            # attribute (not a ctor arg) so modules saved before the flag existed rebuild unchanged.
            # Either a single bool (all-or-nothing, the original behavior) or a per-block list/tuple of
            # bools of length n_layer (F6's selective per-block policy -- see
            # mixle.models.memory_efficient_training.SelectiveRecomputePolicy.apply_to_model, which sets
            # exactly this attribute from a real memory-vs-recompute-FLOPs cost model).
            self.gradient_checkpointing = False

        def _checkpoint_block(self, i: int) -> bool:
            gc = getattr(self, "gradient_checkpointing", False)
            if isinstance(gc, bool):
                return gc
            return bool(gc[i])  # per-block list/tuple, one entry per self.blocks

        def forward(
            self,
            x: Any,
            *,
            position_ids: Any = None,
            return_all_logits: bool = False,
        ) -> Any:
            """Score a token block with optional global positions.

            ``position_ids`` is the hook context parallelism needs after the
            sequence is sharded: local token chunks retain their positions in
            the global sequence.  The default return remains last-token logits
            for compatibility; training asks for all positions explicitly.
            """
            x = x.long()
            t = x.shape[1]
            if position_ids is None:
                position_ids = torch.arange(t, device=x.device)
            position_ids = position_ids.long()
            if position_ids.ndim == 1:
                position_embeddings = self.pos(position_ids)[None, :, :]
            elif position_ids.ndim == 2:
                position_embeddings = self.pos(position_ids)
            else:
                raise ValueError("position_ids must have shape (sequence,) or (batch, sequence).")
            h = self.tok(x) + position_embeddings
            for i, blk in enumerate(self.blocks):
                if self._checkpoint_block(i) and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
                else:
                    h = blk(h)
            logits = self.head(self.ln(h))
            return logits if return_all_logits else logits[:, -1]


def build_causal_lm(
    vocab: int,
    d_model: int = 128,
    n_layer: int = 3,
    n_head: int = 4,
    block: int = 64,
    embedding: Any = None,
    gradient_checkpointing: bool = False,
) -> Any:
    """Build a causal decoder-only Transformer LM (token+pos embeddings, pre-norm blocks, weight-tied head).

    ``embedding`` optionally injects a *shared* token ``nn.Embedding`` (``vocab x d_model``) to use in place of a
    fresh one -- so several language models can tie the same word embedding and train it jointly (the weight-tied
    head follows it). Its shape must match ``(vocab, d_model)``.

    ``gradient_checkpointing=True`` recomputes block activations during backward instead of storing them --
    identical gradients (pinned by test) for a large activation-memory cut on deep stacks or long blocks.
    The flag is a plain module attribute, so it can also be toggled on an existing model -- including to a
    per-block list/tuple of bools (one per ``n_layer``) rather than a single all-or-nothing bool, for F6's
    cost-model-driven selective policy (``mixle.models.memory_efficient_training.SelectiveRecomputePolicy``).
    """
    from mixle.models.embedding import resolve_embedding

    embedding = resolve_embedding(embedding, vocab, d_model)  # CategoricalEmbedding | nn.Embedding | None -> module
    lm = CausalLM(vocab, d_model, n_layer, n_head, block, embedding=embedding)
    lm.gradient_checkpointing = bool(gradient_checkpointing)
    return lm
