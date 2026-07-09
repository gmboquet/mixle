"""E6: retrieval memory over frozen past -- a :class:`~mixle.experimental.context_spine.ContextMechanism`
that pairs E1's local sliding window with an unbounded-length, stop-gradient kNN index of everything that
has scrolled out of the training horizon.

**Why this exists.** ``SlidingWindowSpine`` (E1) only ever attends to the last ``window`` tokens -- anything
older is simply gone, so needle-in-a-haystack facts planted before the window fall out of reach no matter how
long training runs. ``RetrievalMemorySpine`` keeps the same local window for near-range recall, but ALSO
archives every processed chunk's post-RoPE keys/values (detached) into a per-layer index carried in the
state, and on every subsequent step does a brute-force kNN lookup of that index for each query, attending
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
differentiable softmax. ``notes/standout-roadmap-tasks.md``'s E6 card asks for this mechanism's state cost
"at a fraction of E2's" (moment-closure attention). **E2 does not exist on any branch reachable from this
worktree's base as of this writing** (see ``RETRIEVAL_MEMORY_UNAVAILABLE_PIECES`` below, matching the
convention ``mixle/task/pilot_ladder.py`` uses for roadmap pieces it cannot reach) -- there is nothing to
compare against yet. ``mixle/tests/retrieval_memory_spine_test.py`` instead measures and asserts this
mechanism's OWN state bytes-per-token, honestly, and leaves the E2 ratio as a documented placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

# Reused, not reimplemented: E1's RoPE math is exactly what this mechanism needs for both the local window
# and the archived index (entries are stored post-RoPE, same as SlidingWindowSpine's cache -- see that
# module's docstring). E2-E6 are documented to "differ only in what step's carried state contains"; the
# positional-encoding convention is shared substrate, not a per-mechanism choice.
if _HAS_TORCH:
    from mixle.experimental.context_spine import _apply_rope, _rope_angles  # noqa: E402

__all__ = [
    "RETRIEVAL_MEMORY_UNAVAILABLE_PIECES",
    "RetrievalMemoryState",
    "RetrievalMemorySpine",
]

#: roadmap sub-pieces this module cannot reach from this worktree's base, and exactly why -- see the module
#: docstring and ``mixle/task/pilot_ladder.py``'s ``PILOT_LADDER_UNAVAILABLE_PIECES`` for the same convention.
RETRIEVAL_MEMORY_UNAVAILABLE_PIECES: dict[str, str] = {
    "E2": (
        "moment-closure attention (roadmap E2) does not exist on any branch reachable from this worktree's "
        "base (release/0.7.0 -> chunked-recurrent-spine -> long-context-referee) as of "
        "this writing -- it was being built in parallel on its own branch and never reached origin. The E6 "
        "card's acceptance criterion ('at a fraction of E2's state cost') cannot be checked against a real "
        "E2 measurement; this module instead reports RetrievalMemorySpine's own measured state bytes-per-"
        "token (see mixle/tests/retrieval_memory_spine_test.py) and leaves the E2 ratio as a documented "
        "placeholder rather than a fabricated number."
    ),
}


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover - torch is optional
        raise ImportError("mixle.experimental.retrieval_memory_spine requires torch")


@dataclass
class RetrievalMemoryState:
    """Per-layer local window cache (same shape/convention as ``SlidingWindowState``) plus a per-layer
    detached kNN index of every earlier chunk's post-RoPE keys/values.

    ``index_k``/``index_v``: ``(batch, index_len, n_head, head_dim)`` or ``None`` before the first archive.
    Entries are appended once per ``step`` call (this chunk's own keys/values, never the local window's
    carried-over tail -- that would double-archive the same tokens every step). Always detached at write
    time; see the module docstring for the non-differentiable-boundary contract this enforces.

    ``receipt``: honest, per-step bookkeeping -- see :meth:`RetrievalMemorySpine.step`. Carried forward on
    the state (rather than a third return value) because :class:`~mixle.experimental.context_spine.ContextMechanism`
    fixes ``step``'s return shape to ``(new_state, mean_loss)``; the state IS this mechanism's output.
    """

    cache_k: list[Any] = field(default_factory=list)
    cache_v: list[Any] = field(default_factory=list)
    index_k: list[Any] = field(default_factory=list)
    index_v: list[Any] = field(default_factory=list)
    pos: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)


if _HAS_TORCH:

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
            assert d_model % n_head == 0
            assert window >= 1, "window=0 would leave the first chunk with no valid local keys at all"
            assert retrieval_k >= 1
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = int(window)
            self.retrieval_k = int(retrieval_k)
            self.max_index_tokens = None if max_index_tokens is None else int(max_index_tokens)

            self.tok = nn.Embedding(vocab, d_model)
            self.qkv = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(n_layer)])
            self.proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layer)])
            self.ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
            self.ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
            self.mlp = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
                    for _ in range(n_layer)
                ]
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # weight tying, matching CausalLM / SlidingWindowSpine's convention.

        def init_state(self, batch_size: int, *, device: str = "cpu") -> RetrievalMemoryState:
            del batch_size  # both caches grow lazily from None on first step, like SlidingWindowSpine's.
            del device
            return RetrievalMemoryState(
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                index_k=[None] * self.n_layer,
                index_v=[None] * self.n_layer,
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
                pos=state.pos,
                receipt=dict(state.receipt),
            )

        def step(self, state: RetrievalMemoryState, chunk: tuple[Any, Any]) -> tuple[RetrievalMemoryState, Any]:
            x, y = chunk
            b, t = x.shape
            device = x.device
            query_positions = torch.arange(state.pos, state.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_index_k: list[Any] = []
            new_index_v: list[Any] = []
            retrieved_counts: list[int] = []
            index_lens: list[int] = []

            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (b, t, n_head, head_dim)

                cache_k, cache_v = state.cache_k[layer], state.cache_v[layer]
                if cache_k is not None:
                    cache_len = cache_k.shape[1]
                    key_positions = torch.arange(state.pos - cache_len, state.pos + t, device=device)
                    k_full = torch.cat([cache_k, k], dim=1)
                    v_full = torch.cat([cache_v, v], dim=1)
                else:
                    key_positions = query_positions
                    k_full, v_full = k, v

                sin_q, cos_q = _rope_angles(query_positions, self.head_dim)
                sin_k, cos_k = _rope_angles(key_positions, self.head_dim)
                q = _apply_rope(q, sin_q, cos_q)
                k_full = _apply_rope(k_full, sin_k, cos_k)
                # This chunk's own post-RoPE keys/values -- the slice that gets archived into the index
                # below (never the carried-over cache tail, or every step would re-archive old tokens).
                k_chunk, v_chunk = k_full[:, -t:], v_full[:, -t:]

                delta = query_positions[:, None] - key_positions[None, :]  # (t, len(local keys))
                allowed = (delta >= 0) & (delta < self.window)
                local_mask = torch.zeros(t, key_positions.shape[0], device=device)
                local_mask = local_mask.masked_fill(~allowed, float("-inf"))

                qh = q.transpose(1, 2)  # (b, n_head, t, head_dim)
                kh_local = k_full.transpose(1, 2)  # (b, n_head, len(local keys), head_dim)
                vh_local = v_full.transpose(1, 2)
                local_scores = (qh @ kh_local.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, L)
                local_scores = local_scores + local_mask[None, None]
                local_v_expand = vh_local.unsqueeze(2).expand(b, self.n_head, t, kh_local.shape[2], self.head_dim)

                index_k_layer, index_v_layer = state.index_k[layer], state.index_v[layer]
                index_len = 0 if index_k_layer is None else index_k_layer.shape[1]
                index_lens.append(index_len)
                if index_len > 0:
                    k_eff = min(self.retrieval_k, index_len)
                    index_kh = index_k_layer.transpose(1, 2)  # (b, n_head, index_len, head_dim) -- detached.
                    index_vh = index_v_layer.transpose(1, 2)
                    retrieval_scores = (qh @ index_kh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b,nh,t,index_len)
                    topk_scores, topk_idx = torch.topk(retrieval_scores, k=k_eff, dim=-1)  # each (b, nh, t, k_eff)
                    idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
                    index_v_expand = index_vh.unsqueeze(2).expand(b, self.n_head, t, index_len, self.head_dim)
                    retrieved_v = torch.gather(index_v_expand, dim=3, index=idx_expand)  # (b,nh,t,k_eff,hd)

                    combined_scores = torch.cat([local_scores, topk_scores], dim=-1)
                    combined_v = torch.cat([local_v_expand, retrieved_v], dim=-2)
                    retrieved_counts.append(k_eff)
                else:
                    combined_scores = local_scores
                    combined_v = local_v_expand
                    retrieved_counts.append(0)

                attn = combined_scores.softmax(dim=-1)
                out = torch.einsum("bhtk,bhtkd->bhtd", attn, combined_v).reshape(b, t, self.d_model)
                h = h + self.proj[layer](out)
                h = h + self.mlp[layer](self.ln2[layer](h))

                keep = min(self.window, k_full.shape[1])
                new_cache_k.append(k_full[:, -keep:] if keep > 0 else None)
                new_cache_v.append(v_full[:, -keep:] if keep > 0 else None)

                archived_k = k_chunk.detach()
                archived_v = v_chunk.detach()
                if index_k_layer is not None:
                    archived_k = torch.cat([index_k_layer, archived_k], dim=1)
                    archived_v = torch.cat([index_v_layer, archived_v], dim=1)
                if self.max_index_tokens is not None and archived_k.shape[1] > self.max_index_tokens:
                    archived_k = archived_k[:, -self.max_index_tokens :]
                    archived_v = archived_v[:, -self.max_index_tokens :]
                new_index_k.append(archived_k)
                new_index_v.append(archived_v)

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
                "index_len_per_layer": index_lens,
            }

            new_state = RetrievalMemoryState(
                cache_k=new_cache_k,
                cache_v=new_cache_v,
                index_k=new_index_k,
                index_v=new_index_v,
                pos=state.pos + t,
                receipt=receipt,
            )
            return new_state, loss
