"""E1: the chunked-recurrent training spine every Track-E long-context mechanism plugs into.

See ``notes/designs/E1.md`` for the design decisions (RoPE over learned-absolute position embeddings,
windowed-mask derivation, why ``detach_horizon`` only ever cuts the backward graph and never the forward
one, and why this ships its own tiny TBPTT driver instead of routing through ``GradLeaf``).

``mixle.models.transformer.CausalLM`` and ``mixle.models.streaming_transformer_leaf.StreamingTransformer``
both train bounded, independent micro-batches with a learned position table capped at ``block`` tokens --
neither carries state across calls. ``ContextMechanism`` is the minimal protocol that adds streaming +
carried state + truncated-backprop-through-time (TBPTT) on top, without touching either of those modules.
``SlidingWindowSpine`` is the E1 baseline mechanism (Transformer-XL-style stop-gradient KV carry); E2-E6
differ only in what ``step``'s carried state contains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False


@runtime_checkable
class ContextMechanism(Protocol):
    """The substrate contract every Track-E long-context mechanism implements.

    ``step`` is per-position teacher-forced (returns the mean loss over every position in the chunk, not
    just the last one -- unlike ``CausalLM.forward``, which returns only the last position's logits).
    """

    def init_state(self, batch_size: int, *, device: str = "cpu") -> Any:
        """A fresh state for ``batch_size`` independent streams (empty cache / zero memory)."""
        ...

    def step(self, state: Any, chunk: tuple[Any, Any]) -> tuple[Any, Any]:
        """``chunk = (x, y)``, ``(batch, T)`` long tensors. Returns ``(new_state, mean_loss)``."""
        ...

    def detach(self, state: Any) -> Any:
        """Stop-gradient the carried state (cuts the TBPTT backward graph at this point)."""
        ...


@dataclass
class SlidingWindowState:
    """Per-layer stop-gradient KV cache plus the running absolute position counter (see E1.md's RoPE note)."""

    cache_k: list[Any] = field(default_factory=list)  # per layer: (batch, cache_len<=window, n_head, head_dim) | None
    cache_v: list[Any] = field(default_factory=list)
    pos: int = 0


if _HAS_TORCH:

    def _rotate_half(x: Any) -> Any:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _rope_angles(positions: Any, head_dim: int, base: float = 10000.0) -> tuple[Any, Any]:
        """``(sin, cos)`` of shape ``(len(positions), head_dim)`` -- each half-pair shares one rotation frequency."""
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]  # (len, head_dim/2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # (len, head_dim)
        return freqs.sin(), freqs.cos()

    def _apply_rope(x: Any, sin: Any, cos: Any) -> Any:
        """``x``: ``(batch, T, n_head, head_dim)``; ``sin``/``cos``: ``(T, head_dim)``."""
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
        return x * cos + _rotate_half(x) * sin

    class SlidingWindowSpine(nn.Module):
        """E1 baseline: sliding-window exact attention with a stop-gradient carried KV cache (Transformer-XL style).

        ``window=None`` (or ``window >= `` any sequence length this mechanism will ever see) makes ``step``
        compute ordinary full causal self-attention with no truncation -- the "full-attention-equivalent"
        configuration ``notes/designs/E1.md`` uses as the acceptance baseline, computed by the exact same code
        path multi-chunk streaming uses (not a second, independently-written transformer).
        """

        def __init__(
            self, vocab: int, *, d_model: int = 32, n_layer: int = 2, n_head: int = 2, window: int | None = 64
        ) -> None:
            super().__init__()
            assert d_model % n_head == 0
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = None if window is None else int(window)

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
            self.head.weight = self.tok.weight  # weight tying, matching CausalLM's convention -- nn.Module.parameters()
            # dedupes shared tensors, so this doesn't double-count in the optimizer's param group.

        def init_state(self, batch_size: int, *, device: str = "cpu") -> SlidingWindowState:
            del batch_size  # cache grows lazily from None on first step; shape doesn't need to be pre-declared
            return SlidingWindowState(cache_k=[None] * self.n_layer, cache_v=[None] * self.n_layer, pos=0)

        def detach(self, state: SlidingWindowState) -> SlidingWindowState:
            return SlidingWindowState(
                cache_k=[k.detach() if k is not None else None for k in state.cache_k],
                cache_v=[v.detach() if v is not None else None for v in state.cache_v],
                pos=state.pos,
            )

        def step(self, state: SlidingWindowState, chunk: tuple[Any, Any]) -> tuple[SlidingWindowState, Any]:
            x, y = chunk
            b, t = x.shape
            device = x.device
            query_positions = torch.arange(state.pos, state.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
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

                delta = query_positions[:, None] - key_positions[None, :]  # (t, len(keys))
                allowed = (delta >= 0) & (delta < self.window) if self.window is not None else (delta >= 0)
                mask = torch.zeros(t, key_positions.shape[0], device=device)
                mask = mask.masked_fill(~allowed, float("-inf"))

                qh = q.transpose(1, 2)  # (b, n_head, t, head_dim)
                kh = k_full.transpose(1, 2)  # (b, n_head, len(keys), head_dim)
                vh = v_full.transpose(1, 2)
                attn = (qh @ kh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, len(keys))
                attn = attn + mask[None, None]
                attn = attn.softmax(dim=-1)
                out = (attn @ vh).transpose(1, 2).reshape(b, t, self.d_model)  # (b, t, d_model)
                h = h + self.proj[layer](out)
                h = h + self.mlp[layer](self.ln2[layer](h))

                keep = self.window if self.window is not None else k_full.shape[1]
                new_cache_k.append(k_full[:, -keep:])
                new_cache_v.append(v_full[:, -keep:])

            logits = self.head(self.ln_f(h))  # (b, t, vocab)
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            new_state = SlidingWindowState(cache_k=new_cache_k, cache_v=new_cache_v, pos=state.pos + t)
            return new_state, loss


def train_tbptt(
    mechanism: ContextMechanism,
    state: Any,
    chunks: Any,
    opt: Any,
    *,
    detach_horizon: int = 1,
) -> dict[str, Any]:
    """Stream ``chunks`` through ``mechanism``, TBPTT-training with the given optimizer.

    Every ``detach_horizon`` chunks (or at end of stream, whichever comes first): backward the mean
    accumulated loss, step the optimizer, then ``mechanism.detach(state)`` to cut the graph before
    continuing. ``detach_horizon=1`` is literal per-chunk stop-gradient (Transformer-XL); a horizon
    spanning the whole stream means no mid-stream detach happens at all (see ``notes/designs/E1.md``).
    Returns ``{"losses": [float, ...], "state": final_state}`` -- one loss per chunk, detached telemetry.
    """
    losses: list[float] = []
    acc_loss = None
    acc_count = 0
    for chunk in chunks:
        state, loss = mechanism.step(state, chunk)
        losses.append(float(loss.detach()))
        acc_loss = loss if acc_loss is None else acc_loss + loss
        acc_count += 1
        if acc_count >= detach_horizon:
            opt.zero_grad()
            (acc_loss / acc_count).backward()
            opt.step()
            state = mechanism.detach(state)
            acc_loss, acc_count = None, 0
    if acc_loss is not None:
        opt.zero_grad()
        (acc_loss / acc_count).backward()
        opt.step()
        state = mechanism.detach(state)
    return {"losses": losses, "state": state}
