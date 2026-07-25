"""E8: context parallelism for :class:`~mixle.experimental.context_spine.SlidingWindowSpine`.

See ``notes/designs/E8.md`` for the full design (why torch-native CP doesn't suffice --
``SlidingWindowSpine.step`` computes attention by hand, there is no single ``scaled_dot_product_attention``
call for the native CP context manager to intercept -- and why F1's own CP
(``tensor_pipeline_context_parallel.py``) doesn't drop in as-is -- it shards a whole-block STATELESS
forward, not a carried-state streaming step).

This module reuses F1's proven PATTERN (chunk the KV axis, all-gather once per attention op, reconstruct
with an explicit offset mask, verify by exact match against a dense reference) rather than its entry
points, purpose-built for ``ContextMechanism.step``'s carried-state shape: each call shards the CURRENT
step's KV axis (``cache ++ chunk``, not a whole logical stream) across ``cp_size`` simulated ranks.

Same scope cut as F1's CP, for the same reason (this environment has <=1 real GPU): eager
chunked-all-gather CP, exact and testable at small scale, NOT incremental ring attention (no
rank-to-rank streaming overlap). See the design note's Risks section.

RoPE is applied PER SHARD, using each shard's own absolute ``key_positions_shard``, BEFORE the gather
(not once on the full gathered K after). This is deliberately closer to what a real distributed rank
would do (a rank never needs another rank's raw K to compute its own RoPE) at the cost of being only
mathematically -- not bit-exactly -- identical to the single-device code path (float non-associativity
of RoPE-per-slice-then-concat vs. RoPE-on-the-full-array). See ``notes/designs/E8.md``'s Risks section
for the fallback (gather-raw-K-then-RoPE-once) if this tolerance had not held in practice; the test suite
(``mixle/tests/context_parallel_spine_test.py``) measures this directly and it DOES hold, at low-1e-6
absolute divergence even under real multi-chunk SGD training, comfortably inside the documented
``rtol=1e-4`` tolerance (F1's own precedent).

Cache convention: ``SlidingWindowSpine.step`` caches raw K/V together with explicit absolute
positions. RoPE is applied once, after the current raw keys and raw cached keys are assembled for
attention. Both dense and context-parallel paths use this convention, preventing older keys from
being rotated again on every streamed chunk.

Unlike F1's TP (`n_head % tp_size == 0`) and CP (`block % cp_size == 0`), which both fail fast on
uneven division, this module's KV axis length (``cache_len + t``) varies chunk to chunk and is not
required to divide evenly by ``cp_size`` -- ``torch.chunk`` already degrades gracefully (size-balanced,
possibly-uneven chunks, or fewer than ``cp_size`` chunks if the axis is shorter than ``cp_size``) rather
than erroring, and this module relies on exactly that behavior instead of adopting F1's stricter
validate-and-reject posture. See the design note's Risks section for why this is a deliberate difference,
not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False


if _HAS_TORCH:

    def _rotate_half(x: Any) -> Any:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _rope_angles(positions: Any, head_dim: int, base: float = 10000.0) -> tuple[Any, Any]:
        """``(sin, cos)`` of shape ``(len(positions), head_dim)`` -- mirrors ``context_spine._rope_angles``
        exactly (duplicated rather than imported: this module is infrastructure over
        ``SlidingWindowSpine``, not a piece of its public surface, and should not create an import
        dependency between the two sibling ``mixle.experimental`` modules)."""
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

    @dataclass
    class CPWindowShard:
        """One (simulated) rank's contiguous slice of the current step's KV axis (``cache ++ chunk``).

        ``k_shard``/``v_shard`` are this rank's raw (pre-RoPE) K/V slice -- RoPE is applied later, per
        shard, using ``key_positions_shard``, so a real rank never needs another rank's raw K.
        """

        k_shard: Any  # (batch, len_shard, n_head, head_dim) -- pre-RoPE
        v_shard: Any  # (batch, len_shard, n_head, head_dim)
        key_positions_shard: Any  # (len_shard,) absolute positions

    def cp_shard_kv(k_full: Any, v_full: Any, key_positions: Any, cp_size: int) -> list[CPWindowShard]:
        """Split the current step's ``(cache ++ chunk)`` K/V/positions into ``cp_size`` contiguous
        position chunks (``torch.chunk`` along the position axis) -- the "exact window sharded across
        devices" the E8 card names.

        Deliberately does NOT require ``cp_size`` to divide ``k_full``'s length evenly: the KV axis
        length (``cache_len + t``) varies chunk to chunk near stream start/end, unlike F1's TP/CP shapes
        which are fixed and validated up front. ``torch.chunk`` degrades gracefully on its own (uneven
        trailing chunk, or fewer than ``cp_size`` chunks if the axis is shorter than ``cp_size``) -- both
        cases are exercised by ``context_parallel_spine_test.py``'s window-edge-case tests.
        """
        if isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size < 1:
            raise ValueError("cp_size must be a positive integer.")
        if (
            not torch.is_tensor(k_full)
            or not torch.is_tensor(v_full)
            or not torch.is_tensor(key_positions)
            or k_full.ndim != 4
            or v_full.shape != k_full.shape
            or key_positions.ndim != 1
            or key_positions.shape[0] != k_full.shape[1]
        ):
            raise ValueError("K, V, and positions must have aligned KV-cache layouts.")
        if k_full.shape[1] == 0:
            raise ValueError("the KV axis must be non-empty.")
        if k_full.device != v_full.device or k_full.device != key_positions.device:
            raise ValueError("K, V, and positions must be on the same device.")
        k_chunks = torch.chunk(k_full, cp_size, dim=1)
        v_chunks = torch.chunk(v_full, cp_size, dim=1)
        pos_chunks = torch.chunk(key_positions, cp_size, dim=0)
        return [
            CPWindowShard(k_shard=k, v_shard=v, key_positions_shard=p)
            for k, v, p in zip(k_chunks, v_chunks, pos_chunks)
        ]

    def cp_window_attention_forward(
        q: Any,
        query_positions: Any,
        shards: list[CPWindowShard],
        *,
        window: int | None,
        head_dim: int,
    ) -> Any:
        """Context-parallel sliding-window attention for one layer, one step.

        ``q``: ``(batch, t, n_head, head_dim)``, raw (pre-RoPE) queries for this chunk.
        ``query_positions``: ``(t,)`` absolute positions (never sharded -- only keys/values are, per the
        E8 card: "sharding the exact window," i.e. the KV axis, not the query/chunk axis).

        Per shard (rank): apply RoPE to its own K slice using its own absolute ``key_positions_shard``
        (step 3 of the design's algorithm -- no cross-rank RoPE dependency, so this generalizes to a real
        distributed setting where a rank never materializes another rank's raw K/V before the shard
        boundary). Then all-gather (concat in position order) the RoPE'd K and raw V across shards -- the
        one collective per attention call, matching F1 CP's "one collective per block" scope. Finally run
        the *unmodified* masked-matmul attention math ``SlidingWindowSpine.step`` already does (delta =
        query_pos - key_pos; allowed = 0 <= delta < window) against the gathered full K/V.

        Returns ``(batch, t, n_head * head_dim)`` -- the same shape/content
        ``SlidingWindowSpine.step``'s single-device attention sub-step produces, reconstructed exactly
        (mod float non-associativity of the per-shard-RoPE-then-gather order -- see the module
        docstring).
        """
        if (
            not torch.is_tensor(q)
            or not torch.is_tensor(query_positions)
            or q.ndim != 4
            or query_positions.ndim != 1
            or query_positions.shape[0] != q.shape[1]
            or q.shape[-1] != head_dim
        ):
            raise ValueError("queries and query positions have an invalid attention layout.")
        if not shards:
            raise ValueError("at least one KV shard is required.")
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("window must be None or a positive integer.")

        device = q.device
        for shard in shards:
            if (
                shard.k_shard.ndim != 4
                or shard.v_shard.shape != shard.k_shard.shape
                or shard.k_shard.shape[0] != q.shape[0]
                or shard.k_shard.shape[2:] != q.shape[2:]
                or shard.key_positions_shard.ndim != 1
                or shard.key_positions_shard.shape[0] != shard.k_shard.shape[1]
            ):
                raise ValueError("a KV shard has an invalid layout.")
            if (
                shard.k_shard.device != device
                or shard.v_shard.device != device
                or shard.key_positions_shard.device != device
            ):
                raise ValueError("queries and every KV shard must be on the same device.")
        sin_q, cos_q = _rope_angles(query_positions, head_dim)
        q = _apply_rope(q, sin_q, cos_q)

        roped_k_shards = []
        for shard in shards:
            sin_k, cos_k = _rope_angles(shard.key_positions_shard, head_dim)
            roped_k_shards.append(_apply_rope(shard.k_shard, sin_k, cos_k))

        full_k = torch.cat(roped_k_shards, dim=1)  # all-gather (RoPE'd K), concat in position order
        full_v = torch.cat([shard.v_shard for shard in shards], dim=1)  # all-gather (raw V)
        full_key_positions = torch.cat([shard.key_positions_shard for shard in shards], dim=0)
        if full_key_positions.numel() == 0 or (
            full_key_positions.numel() > 1 and not torch.all(full_key_positions[1:] > full_key_positions[:-1])
        ):
            raise ValueError("KV shard positions must form one strictly increasing sequence.")

        b, t, n_head, _ = q.shape
        delta = query_positions[:, None] - full_key_positions[None, :]  # (t, len(keys))
        allowed = (delta >= 0) & (delta < window) if window is not None else (delta >= 0)
        if not torch.all(torch.any(allowed, dim=1)):
            raise ValueError("every query must have at least one causal key.")
        mask = torch.zeros(t, full_key_positions.shape[0], device=device)
        mask = mask.masked_fill(~allowed, float("-inf"))

        qh = q.transpose(1, 2)  # (b, n_head, t, head_dim)
        kh = full_k.transpose(1, 2)  # (b, n_head, len(keys), head_dim)
        vh = full_v.transpose(1, 2)
        attn = (qh @ kh.transpose(-2, -1)) / (head_dim**0.5)
        attn = attn + mask[None, None]
        attn = attn.softmax(dim=-1)
        out = (attn @ vh).transpose(1, 2).reshape(b, t, n_head * head_dim)
        return out

    def validate_cp_window_plan(cp_size: int, window: int | None) -> None:
        """Fail-fast on a malformed ``cp_size``; warn (never error) on a degenerate ``window``/``cp_size``.

        Unlike F1's ``validate_tp_pp_cp_plan`` (which requires exact division of fixed model dimensions),
        E8's KV axis length has no fixed size -- ``cache_len + t`` varies chunk to chunk -- so there is no
        "``cp_size`` must divide the window evenly" check to make: correctness holds for any
        ``cp_size >= 1`` (``cp_shard_kv``'s ``torch.chunk`` degrades gracefully on uneven/short axes).
        The only thing worth flagging is a plan that is *valid but probably pointless*: if the window is
        so small relative to ``cp_size`` that each shard would carry on the order of one key or fewer on
        average, the communication/bookkeeping overhead of sharding is very unlikely to pay for itself --
        that is a performance heads-up, not a correctness problem, hence a warning.
        """
        if isinstance(cp_size, bool) or not isinstance(cp_size, int) or cp_size < 1:
            raise ValueError("cp_size must be a positive integer.")
        if window is not None and (isinstance(window, bool) or not isinstance(window, int) or window <= 0):
            raise ValueError("window must be None or a positive integer.")
        if window is not None and cp_size > 1:
            per_shard = window // cp_size
            if per_shard <= 1:
                import warnings

                warnings.warn(
                    "cp_size=%d with window=%d gives only %d key(s) per shard on average -- "
                    "this is still correct (KV-axis sharding does not require even division) but "
                    "unlikely to be worth the sharding overhead at this window size." % (cp_size, window, per_shard),
                    stacklevel=2,
                )


__all__ = [
    "CPWindowShard",
    "cp_shard_kv",
    "cp_window_attention_forward",
    "validate_cp_window_plan",
]
