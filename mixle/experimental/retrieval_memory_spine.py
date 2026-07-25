"""E6: retrieval memory over frozen past -- a :class:`~mixle.experimental.context_spine.ContextMechanism`
that pairs E1's local sliding window with an unbounded-length, stop-gradient kNN index of everything that
has scrolled out of the local window.

**Why this exists.** ``SlidingWindowSpine`` (E1) only ever attends to the last ``window`` tokens -- anything
older is simply gone, so needle-in-a-haystack facts planted before the window fall out of reach no matter how
long training runs. ``RetrievalMemorySpine`` keeps the same local window for near-range recall, but ALSO
archives only keys/values evicted from the raw local cache into a per-layer index carried in the state,
and does a brute-force kNN lookup of that index for each query, attending
over the top-``retrieval_k`` hits alongside the local window in one combined softmax.

**The non-differentiable boundary (read this before touching the backward pass).** The index is written by
``.detach()``ed tensors from PAST steps -- steps whose own backward graph has already been consumed by
``train_tbptt``'s TBPTT boundary. Nothing in this module tries to differentiate through how those entries
were produced. What DOES stay exact: the retrieval and combination happening THIS step -- ``topk`` selection
of which entries to look at is a discrete, gradient-free op (like sparse/MoE routing), but the softmax
attention over the selected top-k values is full-precision autograd, so gradients flow exactly (no
straight-through / relaxation approximation) into this step's query and output projections. Net effect:
exact gradients through the retrieval OPERATION, zero gradient into the frozen index CONTENTS. Every
``step()`` call documents this on the returned state as ``state.receipt["differentiable_boundary"]`` (a
receipt field, not just a docstring claim -- see the roadmap card, ``notes/standout-roadmap-tasks.md`` E6).

**State cost.** The literal index tensors dominate the state's byte footprint (``O(total tokens streamed)``
unless ``max_index_tokens`` caps it), but backward-pass memory is ``O(window + retrieval_k)`` per query
rather than ``O(index length)`` -- the whole point of gathering only the top-k hits before running the
differentiable softmax. The state receipt reports archived tokens, cache/index overlap, and current index
length so comparisons with other context mechanisms can use measured state rather than a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

# Reused, not reimplemented: E1's RoPE math is exactly what this mechanism needs. Local cache keys stay raw
# and are rotated from their absolute positions at each read; archived index keys are frozen post-RoPE once.
# E2-E6 are documented to "differ only in what step's carried state contains"; positional encoding remains
# shared substrate, not a per-mechanism choice.
if _HAS_TORCH:
    from mixle.experimental.context_spine import _apply_rope, _rope_angles  # noqa: E402

__all__ = [
    "RETRIEVAL_MEMORY_UNAVAILABLE_PIECES",
    "RetrievalMemoryState",
    "RetrievalMemorySpine",
]

# Retained for compatibility with the roadmap-era API. All formerly missing comparison pieces now exist.
RETRIEVAL_MEMORY_UNAVAILABLE_PIECES: dict[str, str] = {}


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("mixle.experimental.retrieval_memory_spine requires torch")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive exact integer, got {value}")
    return int(value)


@dataclass
class RetrievalMemoryState:
    """Per-layer local window cache (same shape/convention as ``SlidingWindowState``) plus a per-layer
    detached kNN index of every earlier chunk's post-RoPE keys/values.

    ``cache_k`` stores raw, unrotated keys so every key receives RoPE exactly once per attention evaluation.
    ``index_k`` stores detached post-RoPE keys only after their positions leave that cache.
    ``index_pos`` is the exact position ledger used to prove the cache and index are disjoint.

    ``receipt``: honest, per-step bookkeeping -- see :meth:`RetrievalMemorySpine.step`. Carried forward on
    the state (rather than a third return value) because :class:`~mixle.experimental.context_spine.ContextMechanism`
    fixes ``step``'s return shape to ``(new_state, mean_loss)``; the state IS this mechanism's output.
    """

    cache_k: list[Any] = field(default_factory=list)
    cache_v: list[Any] = field(default_factory=list)
    index_k: list[Any] = field(default_factory=list)
    index_v: list[Any] = field(default_factory=list)
    index_pos: list[Any] = field(default_factory=list)
    batch_size: int = 0
    pos: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)


if _HAS_TORCH:

    def _merge_attention_heads(out: Any) -> Any:
        """``(batch, head, time, dim)`` to ``(batch, time, head*dim)`` without reordering tokens."""
        if out.ndim != 4:
            raise ValueError("attention output must have shape (batch, head, time, dim)")
        b, _n_head, t, _head_dim = out.shape
        return out.transpose(1, 2).contiguous().reshape(b, t, -1)

    class RetrievalMemorySpine(nn.Module):
        """E6: E1's local sliding window plus a brute-force kNN retrieval index over detached past chunks.

        ``window``: local causal attention span, identical semantics to ``SlidingWindowSpine.window``.
        ``retrieval_k``: how many index entries each query attends over (the "top-k" of the E6 card).
        ``max_index_tokens``: FIFO cap on total archived tokens per layer (``None`` = unbounded). Caps the
        brute-force kNN's ``O(chunk * index_len)`` score matrix and the state's byte footprint; oldest
        entries are evicted first once the cap is exceeded.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int = 64,
            retrieval_k: int = 4,
            max_index_tokens: int | None = None,
        ) -> None:
            super().__init__()
            self.vocab = _positive_integer(vocab, "vocab")
            self.d_model = _positive_integer(d_model, "d_model")
            self.n_layer = _positive_integer(n_layer, "n_layer")
            self.n_head = _positive_integer(n_head, "n_head")
            self.window = _positive_integer(window, "window")
            self.retrieval_k = _positive_integer(retrieval_k, "retrieval_k")
            if self.d_model % self.n_head != 0:
                raise ValueError(f"d_model={self.d_model} must be divisible by n_head={self.n_head}")
            self.head_dim = d_model // n_head
            self.max_index_tokens = (
                None if max_index_tokens is None else _positive_integer(max_index_tokens, "max_index_tokens")
            )

            self.tok = nn.Embedding(self.vocab, self.d_model)
            self.qkv = nn.ModuleList([nn.Linear(self.d_model, 3 * self.d_model) for _ in range(self.n_layer)])
            self.proj = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(self.n_layer)])
            self.ln1 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layer)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layer)])
            self.mlp = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.d_model, 4 * self.d_model),
                        nn.GELU(),
                        nn.Linear(4 * self.d_model, self.d_model),
                    )
                    for _ in range(self.n_layer)
                ]
            )
            self.ln_f = nn.LayerNorm(self.d_model)
            self.head = nn.Linear(self.d_model, self.vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying, matching CausalLM / SlidingWindowSpine's convention.

        def init_state(self, batch_size: int, *, device: str = "cpu") -> RetrievalMemoryState:
            batch_size = _positive_integer(batch_size, "batch_size")
            torch.empty(0, device=device)  # validate the requested device before returning a state bound to it
            return RetrievalMemoryState(
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                index_k=[None] * self.n_layer,
                index_v=[None] * self.n_layer,
                index_pos=[None] * self.n_layer,
                batch_size=batch_size,
                pos=0,
            )

        def detach(self, state: RetrievalMemoryState) -> RetrievalMemoryState:
            """Stop-gradient the local window cache (cuts the TBPTT graph, same as E1). The index is
            already detached at write time (see :meth:`step`), so this only needs to re-detach the cache."""
            return RetrievalMemoryState(
                cache_k=[k.detach() if k is not None else None for k in state.cache_k],
                cache_v=[v.detach() if v is not None else None for v in state.cache_v],
                index_k=list(state.index_k),
                index_v=list(state.index_v),
                index_pos=list(state.index_pos),
                batch_size=state.batch_size,
                pos=state.pos,
                receipt=dict(state.receipt),
            )

        def step(self, state: RetrievalMemoryState, chunk: tuple[Any, Any]) -> tuple[RetrievalMemoryState, Any]:
            if not isinstance(state, RetrievalMemoryState):
                raise TypeError("state must be a RetrievalMemoryState")
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise TypeError("chunk must be an (input_tokens, target_tokens) tuple")
            x, y = chunk
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.dtype != torch.long or y.dtype != torch.long:
                raise TypeError("input and target tokens must be torch.long tensors")
            if x.ndim != 2 or y.shape != x.shape or x.shape[1] == 0:
                raise ValueError("input and target tokens must have equal non-empty (batch, time) shape")
            b, t = x.shape
            if state.batch_size != b:
                raise ValueError(f"state batch_size={state.batch_size} does not match chunk batch_size={b}")
            if isinstance(state.pos, bool) or not isinstance(state.pos, Integral) or state.pos < 0:
                raise ValueError("state.pos must be a non-negative exact integer")
            state_lists = {
                "cache_k": state.cache_k,
                "cache_v": state.cache_v,
                "index_k": state.index_k,
                "index_v": state.index_v,
                "index_pos": state.index_pos,
            }
            for name, values in state_lists.items():
                if len(values) != self.n_layer:
                    raise ValueError(f"state.{name} must contain exactly {self.n_layer} layer entries")
            if bool(((x < 0) | (x >= self.vocab) | (y < 0) | (y >= self.vocab)).any().item()):
                raise ValueError(f"input and target token IDs must lie in [0, {self.vocab})")
            device = x.device
            if y.device != device:
                raise ValueError("input and target tokens must be on the same device")
            query_positions = torch.arange(state.pos, state.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_index_k: list[Any] = []
            new_index_v: list[Any] = []
            new_index_pos: list[Any] = []
            retrieved_counts: list[int] = []
            index_lens_before: list[int] = []
            index_lens_after: list[int] = []
            archived_counts: list[int] = []
            cache_index_overlaps: list[int] = []
            dual_visible_counts: list[int] = []

            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q_raw, k_raw, v_raw = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

                cache_k_raw, cache_v_raw = state.cache_k[layer], state.cache_v[layer]
                if (cache_k_raw is None) != (cache_v_raw is None):
                    raise ValueError(f"state cache key/value presence differs at layer {layer}")
                if cache_k_raw is not None:
                    expected_tail = (b, self.n_head, self.head_dim)
                    if cache_k_raw.shape[0:1] + cache_k_raw.shape[2:] != expected_tail:
                        raise ValueError(f"state cache shape is incompatible at layer {layer}")
                    if cache_v_raw.shape != cache_k_raw.shape:
                        raise ValueError(f"state cache key/value shapes differ at layer {layer}")
                    if cache_k_raw.device != device or cache_v_raw.device != device:
                        raise ValueError(f"state cache and chunk devices differ at layer {layer}")
                    cache_len = cache_k_raw.shape[1]
                    if cache_len > self.window or cache_len > state.pos:
                        raise ValueError(f"state cache length is invalid at layer {layer}")
                    key_positions = torch.arange(state.pos - cache_len, state.pos + t, device=device)
                    k_full_raw = torch.cat([cache_k_raw, k_raw], dim=1)
                    v_full_raw = torch.cat([cache_v_raw, v_raw], dim=1)
                else:
                    key_positions = query_positions
                    k_full_raw, v_full_raw = k_raw, v_raw

                sin_q, cos_q = _rope_angles(query_positions, self.head_dim)
                sin_k, cos_k = _rope_angles(key_positions, self.head_dim)
                q = _apply_rope(q_raw, sin_q, cos_q)
                k_full = _apply_rope(k_full_raw, sin_k, cos_k)

                delta = query_positions[:, None] - key_positions[None, :]  # (t, len(local keys))
                allowed = (delta >= 0) & (delta < self.window)
                local_mask = torch.zeros(t, key_positions.shape[0], device=device)
                local_mask = local_mask.masked_fill(~allowed, float("-inf"))

                qh = q.transpose(1, 2)  # (b, n_head, t, head_dim)
                kh_local = k_full.transpose(1, 2)  # (b, n_head, len(local keys), head_dim)
                vh_local = v_full_raw.transpose(1, 2)
                local_scores = (qh @ kh_local.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, L)
                local_scores = local_scores + local_mask[None, None]
                local_v_expand = vh_local.unsqueeze(2).expand(b, self.n_head, t, kh_local.shape[2], self.head_dim)

                keep = min(self.window, k_full_raw.shape[1])
                evict_count = k_full_raw.shape[1] - keep
                index_k_layer, index_v_layer = state.index_k[layer], state.index_v[layer]
                index_pos_layer = state.index_pos[layer]
                if not ((index_k_layer is None) == (index_v_layer is None) == (index_pos_layer is None)):
                    raise ValueError(f"state index key/value/position presence differs at layer {layer}")
                index_len_before = 0 if index_k_layer is None else index_k_layer.shape[1]
                index_lens_before.append(index_len_before)
                if index_k_layer is not None:
                    if index_k_layer.shape != (b, index_len_before, self.n_head, self.head_dim):
                        raise ValueError(f"state index key shape is incompatible at layer {layer}")
                    if index_v_layer.shape != index_k_layer.shape or index_pos_layer.shape != (index_len_before,):
                        raise ValueError(f"state index value/position shape is incompatible at layer {layer}")
                    if index_pos_layer.dtype != torch.long:
                        raise TypeError(f"state index positions must use torch.long at layer {layer}")
                    if (
                        index_k_layer.device != device
                        or index_v_layer.device != device
                        or index_pos_layer.device != device
                    ):
                        raise ValueError(f"state index and chunk devices differ at layer {layer}")
                    if index_len_before and (
                        bool((index_pos_layer[1:] <= index_pos_layer[:-1]).any().item())
                        or int(index_pos_layer[-1].item()) >= state.pos - cache_len
                    ):
                        raise ValueError(
                            f"state index positions must be unique, ordered, and outside the cache at layer {layer}"
                        )

                archived_k = k_full[:, :evict_count].detach()
                archived_v = v_full_raw[:, :evict_count].detach()
                archived_pos = key_positions[:evict_count].detach()
                archived_counts.append(evict_count)
                if index_k_layer is not None:
                    archived_k = torch.cat([index_k_layer, archived_k], dim=1)
                    archived_v = torch.cat([index_v_layer, archived_v], dim=1)
                    archived_pos = torch.cat([index_pos_layer, archived_pos], dim=0)
                if self.max_index_tokens is not None and archived_k.shape[1] > self.max_index_tokens:
                    archived_k = archived_k[:, -self.max_index_tokens :]
                    archived_v = archived_v[:, -self.max_index_tokens :]
                    archived_pos = archived_pos[-self.max_index_tokens :]

                index_len_after = archived_k.shape[1]
                index_lens_after.append(index_len_after)
                if index_len_after > 0:
                    k_eff = min(self.retrieval_k, index_len_after)
                    index_kh = archived_k.transpose(1, 2)  # (b, n_head, index_len, head_dim) -- detached.
                    index_vh = archived_v.transpose(1, 2)
                    retrieval_scores = (qh @ index_kh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b,nh,t,index_len)
                    retrieval_delta = query_positions[:, None] - archived_pos[None, :]
                    retrieval_allowed = retrieval_delta >= self.window
                    retrieval_scores = retrieval_scores.masked_fill(~retrieval_allowed[None, None], float("-inf"))
                    topk_scores, topk_idx = torch.topk(retrieval_scores, k=k_eff, dim=-1)  # each (b, nh, t, k_eff)
                    idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
                    index_v_expand = index_vh.unsqueeze(2).expand(b, self.n_head, t, index_len_after, self.head_dim)
                    retrieved_v = torch.gather(index_v_expand, dim=3, index=idx_expand)  # (b,nh,t,k_eff,hd)

                    combined_scores = torch.cat([local_scores, topk_scores], dim=-1)
                    combined_v = torch.cat([local_v_expand, retrieved_v], dim=-2)
                    retrieved_counts.append(int(retrieval_allowed.sum(-1).clamp(max=k_eff).max().item()))
                    same_position = key_positions[:, None] == archived_pos[None, :]
                    dual_visible = allowed[:, :, None] & retrieval_allowed[:, None, :] & same_position[None, :, :]
                    dual_visible_counts.append(int(dual_visible.sum().item()))
                else:
                    combined_scores = local_scores
                    combined_v = local_v_expand
                    retrieved_counts.append(0)
                    dual_visible_counts.append(0)

                attn = combined_scores.softmax(dim=-1)
                out = _merge_attention_heads(torch.einsum("bhtk,bhtkd->bhtd", attn, combined_v))
                h = h + self.proj[layer](out)
                h = h + self.mlp[layer](self.ln2[layer](h))

                cache_k_next = k_full_raw[:, -keep:]
                cache_v_next = v_full_raw[:, -keep:]
                cache_pos_next = key_positions[-keep:]
                overlap = (
                    torch.isin(cache_pos_next, archived_pos).sum().item()
                    if archived_pos.numel() and cache_pos_next.numel()
                    else 0
                )
                cache_index_overlaps.append(int(overlap))
                new_cache_k.append(cache_k_next)
                new_cache_v.append(cache_v_next)
                new_index_k.append(archived_k)
                new_index_v.append(archived_v)
                new_index_pos.append(archived_pos)

            logits = self.head(self.ln_f(h))  # (b, t, vocab)
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            receipt = {
                "differentiable_boundary": (
                    "gradients flow exactly through this step's local window and the retrieval softmax over "
                    "the selected top-k index entries; the index CONTENTS (written by past, now-detached "
                    "steps) carry no gradient -- torch.topk's selection is itself non-differentiable (a "
                    "discrete routing choice, like sparse/MoE gating), not a relaxation."
                ),
                "retrieval_k_requested": self.retrieval_k,
                "retrieved_per_layer": retrieved_counts,
                "index_len_before_per_layer": index_lens_before,
                "index_len_per_layer": index_lens_after,
                "archived_this_step_per_layer": archived_counts,
                "cache_index_position_overlap_per_layer": cache_index_overlaps,
                "dual_visible_position_count_per_layer": dual_visible_counts,
            }

            new_state = RetrievalMemoryState(
                cache_k=new_cache_k,
                cache_v=new_cache_v,
                index_k=new_index_k,
                index_v=new_index_v,
                index_pos=new_index_pos,
                batch_size=state.batch_size,
                pos=state.pos + t,
                receipt=receipt,
            )
            return new_state, loss
