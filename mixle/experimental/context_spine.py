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
    """Raw per-layer KV caches and their explicit absolute positions."""

    cache_k: list[Any] = field(default_factory=list)  # per layer: (batch, cache_len<=window, n_head, head_dim) | None
    cache_v: list[Any] = field(default_factory=list)
    pos: int = 0
    cache_positions: list[Any] = field(default_factory=list)
    batch_size: int | None = None
    n_head: int | None = None
    head_dim: int | None = None
    device: Any | None = None


if _HAS_TORCH:

    def _rotate_half(x: Any) -> Any:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _rope_angles(positions: Any, head_dim: int, base: float = 10000.0) -> tuple[Any, Any]:
        """``(sin, cos)`` of shape ``(len(positions), head_dim)`` -- each half-pair shares one rotation frequency."""
        if positions.ndim != 1:
            raise ValueError("RoPE positions must be one-dimensional.")
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim must be a positive even integer.")
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device) / head_dim)
        )
        freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]  # (len, head_dim/2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # (len, head_dim)
        return freqs.sin(), freqs.cos()

    def _apply_rope(x: Any, sin: Any, cos: Any) -> Any:
        """``x``: ``(batch, T, n_head, head_dim)``; ``sin``/``cos``: ``(T, head_dim)``."""
        if x.ndim != 4 or sin.shape != (x.shape[1], x.shape[-1]) or cos.shape != sin.shape:
            raise ValueError("RoPE tensors do not match the expected (batch, time, head, head_dim) layout.")
        sin = sin.to(device=x.device, dtype=x.dtype)[None, :, None, :]
        cos = cos.to(device=x.device, dtype=x.dtype)[None, :, None, :]
        return x * cos + _rotate_half(x) * sin

    class SlidingWindowSpine(nn.Module):
        """E1 baseline: sliding-window exact attention with a stop-gradient carried KV cache (Transformer-XL style).

        ``window=None`` (or ``window >= `` any sequence length this mechanism will ever see) makes ``step``
        compute ordinary full causal self-attention with no truncation -- the "full-attention-equivalent"
        configuration ``notes/designs/E1.md`` uses as the acceptance baseline, computed by the exact same code
        path multi-chunk streaming uses (not a second, independently-written transformer).

        ``cp_size`` (E8, ``notes/designs/E8.md``): context-parallel window sharding. ``cp_size=1`` (the
        default) is the original single-device path above, completely unchanged -- byte-identical, not just
        numerically close, so E1's existing behavior and tests are untouched. ``cp_size > 1`` shards the
        current step's KV axis (``cache ++ chunk``) across ``cp_size`` simulated ranks via
        ``mixle.experimental.context_parallel_spine`` for the attention sub-step of every layer; nothing
        else (embeddings, LayerNorm, MLP, head, cache bookkeeping) changes, since those are all per-position
        and need no communication.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int | None = 64,
            cp_size: int = 1,
        ) -> None:
            super().__init__()
            integer_fields = {"vocab": vocab, "d_model": d_model, "n_layer": n_layer, "n_head": n_head}
            for field_name, field_value in integer_fields.items():
                if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
                    raise ValueError(f"{field_name} must be a positive integer.")
            if d_model % n_head != 0:
                raise ValueError("d_model must be divisible by n_head.")
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            if self.head_dim % 2:
                raise ValueError("d_model / n_head must be even for RoPE.")
            if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
                raise ValueError("window must be None or a positive integer.")
            self.window = window

            if isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size < 1:
                raise ValueError("cp_size must be a positive integer.")
            self.cp_size = cp_size
            if self.cp_size > 1:
                from mixle.experimental.context_parallel_spine import validate_cp_window_plan

                validate_cp_window_plan(self.cp_size, self.window)

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
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            state_device = torch.device(device)
            return SlidingWindowState(
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                cache_positions=[None] * self.n_layer,
                pos=0,
                batch_size=batch_size,
                n_head=self.n_head,
                head_dim=self.head_dim,
                device=state_device,
            )

        def detach(self, state: SlidingWindowState) -> SlidingWindowState:
            return SlidingWindowState(
                cache_k=[k.detach() if k is not None else None for k in state.cache_k],
                cache_v=[v.detach() if v is not None else None for v in state.cache_v],
                pos=state.pos,
                cache_positions=[p.detach() if p is not None else None for p in state.cache_positions],
                batch_size=state.batch_size,
                n_head=state.n_head,
                head_dim=state.head_dim,
                device=state.device,
            )

        def _validate_state(self, state: SlidingWindowState, batch_size: int, device: Any) -> None:
            if not isinstance(state, SlidingWindowState):
                raise TypeError("state must be a SlidingWindowState.")
            if isinstance(state.pos, bool) or not isinstance(state.pos, int) or state.pos < 0:
                raise ValueError("state.pos must be a non-negative integer.")
            if state.batch_size != batch_size or state.n_head != self.n_head or state.head_dim != self.head_dim:
                raise ValueError("state batch/head metadata does not match this model and chunk.")
            if torch.device(state.device) != device:
                raise ValueError("state device does not match the chunk device.")
            if not all(len(values) == self.n_layer for values in (state.cache_k, state.cache_v, state.cache_positions)):
                raise ValueError("state must contain one K, V, and position cache per layer.")
            presence = [
                cache_k is not None or cache_v is not None or positions is not None
                for cache_k, cache_v, positions in zip(state.cache_k, state.cache_v, state.cache_positions)
            ]
            if any(presence) and not all(presence):
                raise ValueError("all layer caches must be empty or populated together.")

            expected_positions = None
            for layer, (cache_k, cache_v, positions) in enumerate(
                zip(state.cache_k, state.cache_v, state.cache_positions)
            ):
                if cache_k is None or cache_v is None or positions is None:
                    if cache_k is not None or cache_v is not None or positions is not None:
                        raise ValueError(f"layer {layer} has a partial cache.")
                    continue
                if (
                    cache_k.ndim != 4
                    or cache_v.shape != cache_k.shape
                    or cache_k.shape[0] != batch_size
                    or cache_k.shape[2:] != (self.n_head, self.head_dim)
                ):
                    raise ValueError(f"layer {layer} cache has an invalid layout.")
                if cache_k.device != device or cache_v.device != device or positions.device != device:
                    raise ValueError(f"layer {layer} cache is on the wrong device.")
                if cache_k.dtype != self.tok.weight.dtype or cache_v.dtype != cache_k.dtype:
                    raise ValueError(f"layer {layer} cache has the wrong dtype.")
                cache_len = cache_k.shape[1]
                if positions.ndim != 1 or positions.shape[0] != cache_len or state.pos < cache_len:
                    raise ValueError(f"layer {layer} cache positions have an invalid layout.")
                if self.window is not None and cache_len > self.window:
                    raise ValueError(f"layer {layer} cache exceeds the configured window.")
                wanted = torch.arange(state.pos - cache_len, state.pos, device=device)
                if not torch.equal(positions, wanted):
                    raise ValueError(f"layer {layer} cache positions are not the expected absolute positions.")
                if expected_positions is None:
                    expected_positions = positions
                elif not torch.equal(positions, expected_positions):
                    raise ValueError("all layer caches must cover the same absolute positions.")

        def step(self, state: SlidingWindowState, chunk: tuple[Any, Any]) -> tuple[SlidingWindowState, Any]:
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise ValueError("chunk must be an (x, y) tuple.")
            x, y = chunk
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.ndim != 2 or y.shape != x.shape or x.shape[1] == 0:
                raise ValueError("x and y must be non-empty tensors with matching (batch, time) shape.")
            if x.dtype not in (torch.int32, torch.int64) or y.dtype != torch.int64:
                raise ValueError("x must contain integer token IDs and y must be torch.int64 targets.")
            if x.device != y.device:
                raise ValueError("x and y must be on the same device.")
            model_device = self.tok.weight.device
            if x.device != model_device:
                raise ValueError("chunk device must match the model device.")
            b, t = x.shape
            device = x.device
            self._validate_state(state, b, device)
            query_positions = torch.arange(state.pos, state.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_cache_positions: list[Any] = []
            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (b, t, n_head, head_dim)

                cache_k, cache_v = state.cache_k[layer], state.cache_v[layer]
                if cache_k is not None:
                    key_positions = torch.cat([state.cache_positions[layer], query_positions])
                    k_full = torch.cat([cache_k, k], dim=1)
                    v_full = torch.cat([cache_v, v], dim=1)
                else:
                    key_positions = query_positions
                    k_full, v_full = k, v

                if self.cp_size == 1:
                    # Original single-device path -- unchanged, byte-identical to E1.
                    sin_q, cos_q = _rope_angles(query_positions, self.head_dim)
                    sin_k, cos_k = _rope_angles(key_positions, self.head_dim)
                    q = _apply_rope(q, sin_q, cos_q)
                    attention_k = _apply_rope(k_full, sin_k, cos_k)

                    delta = query_positions[:, None] - key_positions[None, :]  # (t, len(keys))
                    allowed = (delta >= 0) & (delta < self.window) if self.window is not None else (delta >= 0)
                    mask = torch.zeros(t, key_positions.shape[0], device=device)
                    mask = mask.masked_fill(~allowed, float("-inf"))

                    qh = q.transpose(1, 2)  # (b, n_head, t, head_dim)
                    kh = attention_k.transpose(1, 2)  # (b, n_head, len(keys), head_dim)
                    vh = v_full.transpose(1, 2)
                    attn = (qh @ kh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, len(keys))
                    attn = attn + mask[None, None]
                    attn = attn.softmax(dim=-1)
                    out = (attn @ vh).transpose(1, 2).reshape(b, t, self.d_model)  # (b, t, d_model)
                else:
                    # E8: shard the KV axis (cache ++ chunk) across cp_size simulated ranks and reconstruct
                    # the dense attention output. k_full/v_full here are still raw (pre-RoPE) -- RoPE is
                    # applied per-shard inside cp_window_attention_forward, see notes/designs/E8.md.
                    from mixle.experimental.context_parallel_spine import cp_shard_kv, cp_window_attention_forward

                    shards = cp_shard_kv(k_full, v_full, key_positions, self.cp_size)
                    out = cp_window_attention_forward(
                        q, query_positions, shards, window=self.window, head_dim=self.head_dim
                    )
                h = h + self.proj[layer](out)
                h = h + self.mlp[layer](self.ln2[layer](h))

                keep = self.window if self.window is not None else k_full.shape[1]
                new_cache_k.append(k_full[:, -keep:])
                new_cache_v.append(v_full[:, -keep:])
                new_cache_positions.append(key_positions[-keep:])

            logits = self.head(self.ln_f(h))  # (b, t, vocab)
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            new_state = SlidingWindowState(
                cache_k=new_cache_k,
                cache_v=new_cache_v,
                cache_positions=new_cache_positions,
                pos=state.pos + t,
                batch_size=b,
                n_head=self.n_head,
                head_dim=self.head_dim,
                device=device,
            )
            return new_state, loss


def train_tbptt(
    mechanism: ContextMechanism,
    state: Any,
    chunks: Any,
    opt: Any,
    *,
    detach_horizon: int = 1,
    extra_loss: Any | None = None,
) -> dict[str, Any]:
    """Stream ``chunks`` through ``mechanism``, TBPTT-training with the given optimizer.

    Every ``detach_horizon`` chunks (or at end of stream, whichever comes first): backward the mean
    accumulated loss, step the optimizer, then ``mechanism.detach(state)`` to cut the graph before
    continuing. ``detach_horizon=1`` is literal per-chunk stop-gradient (Transformer-XL); a horizon
    spanning the whole stream means no mid-stream detach happens at all (see ``notes/designs/E1.md``).
    Returns ``{"losses": [float, ...], "state": final_state}`` -- one loss per chunk, detached telemetry.

    ``extra_loss`` is an optional live tensor added to the FIRST backward only -- a term earned outside
    ``chunks`` (e.g. an auxiliary regularizer from a warm-up prefix this call does not train the LM on).
    Folding it in here rather than backpropagating it separately keeps the optimizer-step count
    independent of whether the term is present, so an ablation on it is not confounded by step count.
    It is excluded from the returned ``losses`` telemetry, which stays one entry per chunk.
    """
    if isinstance(detach_horizon, bool) or not isinstance(detach_horizon, int) or detach_horizon <= 0:
        raise ValueError("detach_horizon must be a positive integer.")
    # Duck-typed on purpose: this function is defined outside the ``_HAS_TORCH`` guard and must not name torch.
    if extra_loss is not None and not (hasattr(extra_loss, "backward") and getattr(extra_loss, "ndim", None) == 0):
        raise TypeError("extra_loss must be a scalar torch tensor or None.")
    losses: list[float] = []
    acc_loss = None
    acc_count = 0
    pending_extra = extra_loss
    for chunk in chunks:
        state, loss = mechanism.step(state, chunk)
        losses.append(float(loss.detach()))
        acc_loss = loss if acc_loss is None else acc_loss + loss
        acc_count += 1
        if acc_count >= detach_horizon:
            opt.zero_grad()
            step_loss = acc_loss / acc_count
            if pending_extra is not None:
                step_loss, pending_extra = step_loss + pending_extra, None
            step_loss.backward()
            opt.step()
            state = mechanism.detach(state)
            acc_loss, acc_count = None, 0
    # ``pending_extra`` survives an empty or exactly-horizon-aligned stream: it must still be trained on.
    if acc_loss is not None or pending_extra is not None:
        opt.zero_grad()
        step_loss = None if acc_loss is None else acc_loss / acc_count
        if pending_extra is not None:
            step_loss = pending_extra if step_loss is None else step_loss + pending_extra
        step_loss.backward()
        opt.step()
        state = mechanism.detach(state)
    return {"losses": losses, "state": state}
