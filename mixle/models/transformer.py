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


def _positive_int(value: Any, name: str) -> int:
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_architecture(vocab: Any, d_model: Any, n_layer: Any, n_head: Any, block: Any) -> tuple[int, ...]:
    values = tuple(
        _positive_int(value, name)
        for value, name in (
            (vocab, "vocab"),
            (d_model, "d_model"),
            (n_layer, "n_layer"),
            (n_head, "n_head"),
            (block, "block"),
        )
    )
    _, model_width, _, heads, _ = values
    if model_width % heads != 0:
        raise ValueError(f"d_model ({model_width}) must be divisible by n_head ({heads})")
    return values


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
            vocab, d_model, n_layer, n_head, block = _validate_architecture(vocab, d_model, n_layer, n_head, block)
            if embedding is not None:
                if not isinstance(embedding, nn.Embedding):
                    raise TypeError("embedding must be a torch.nn.Embedding")
                if embedding.num_embeddings != vocab or embedding.embedding_dim != d_model:
                    raise ValueError(
                        f"embedding must have shape ({vocab}, {d_model}), got "
                        f"({embedding.num_embeddings}, {embedding.embedding_dim})"
                    )
            # record the shape so a trained module can be rebuilt from hyperparameters on load
            self.vocab = vocab
            self.d_model = d_model
            self.n_layer = n_layer
            self.n_head = n_head
            self.block = block
            self.tok = embedding if embedding is not None else nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(block, d_model)
            self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying
            # Persistent production readout scale. Standard parametrization uses 1; apply_mup_init()
            # replaces it with target_width/base_width ** -1 so every logits path implements the
            # output-role part of the muP abc-parametrization.
            self.register_buffer("mup_output_multiplier", torch.tensor(1.0), persistent=True)
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
            return gc[i]

        def _validate_checkpoint_policy(self) -> None:
            policy = getattr(self, "gradient_checkpointing", False)
            if isinstance(policy, bool):
                return
            if not isinstance(policy, (list, tuple)):
                raise ValueError("gradient_checkpointing must be a bool or a per-block list/tuple of bools")
            if len(policy) != len(self.blocks):
                raise ValueError(f"gradient_checkpointing has {len(policy)} entries; expected {len(self.blocks)}")
            if any(not isinstance(value, bool) for value in policy):
                raise ValueError("every per-block gradient_checkpointing entry must be a bool")

        def _validated_ids(
            self,
            value: Any,
            *,
            name: str,
            expected_shapes: tuple[tuple[int, ...], ...],
            upper_bound: int,
        ) -> Any:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            if tuple(value.shape) not in expected_shapes:
                expected = " or ".join(str(shape) for shape in expected_shapes)
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
            if value.device != self.tok.weight.device:
                raise ValueError(f"{name} is on {value.device}; model token embeddings are on {self.tok.weight.device}")
            if value.dtype == torch.bool or (not value.is_floating_point() and not value.dtype.is_signed):
                raise ValueError(f"{name} must use a signed integer or floating-point dtype")
            if value.is_floating_point():
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{name} must contain only finite values")
                if not bool((value == torch.round(value)).all()):
                    raise ValueError(f"{name} must contain integer-valued entries")
            ids = value.long()
            if not bool(((ids >= 0) & (ids < upper_bound)).all()):
                raise ValueError(f"{name} entries must lie in [0, {upper_bound})")
            return ids

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
            if not isinstance(x, torch.Tensor):
                raise TypeError("x must be a torch tensor")
            if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
                raise ValueError("x must have non-empty shape (batch, sequence)")
            batch, t = x.shape
            if t > self.block:
                raise ValueError(f"x sequence length {t} exceeds configured block size {self.block}")
            if not isinstance(return_all_logits, bool):
                raise ValueError("return_all_logits must be a boolean")
            self._validate_checkpoint_policy()
            x = self._validated_ids(
                x,
                name="x",
                expected_shapes=((batch, t),),
                upper_bound=self.vocab,
            )
            if position_ids is None:
                position_ids = torch.arange(t, device=x.device)
            position_ids = self._validated_ids(
                position_ids,
                name="position_ids",
                expected_shapes=((t,), (batch, t)),
                upper_bound=self.block,
            )
            if position_ids.ndim == 1:
                position_embeddings = self.pos(position_ids)[None, :, :]
            else:
                position_embeddings = self.pos(position_ids)
            h = self.tok(x) + position_embeddings
            for i, blk in enumerate(self.blocks):
                if self._checkpoint_block(i) and self.training and torch.is_grad_enabled():
                    h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
                else:
                    h = blk(h)
            logits = self.head(self.ln(h))
            logits = logits * getattr(self, "mup_output_multiplier", 1.0)
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
    if not _HAS_TORCH:
        raise ImportError("build_causal_lm requires torch.")
    vocab, d_model, n_layer, n_head, block = _validate_architecture(vocab, d_model, n_layer, n_head, block)
    if not isinstance(gradient_checkpointing, bool):
        raise ValueError("gradient_checkpointing must be a boolean")

    from mixle.models.embedding import resolve_embedding

    embedding = resolve_embedding(embedding, vocab, d_model)  # CategoricalEmbedding | nn.Embedding | None -> module
    lm = CausalLM(vocab, d_model, n_layer, n_head, block, embedding=embedding)
    lm.gradient_checkpointing = gradient_checkpointing
    return lm
