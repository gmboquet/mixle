"""E10: group-structured quantized-key attention -- an ADAPTIVE far-field state whose compression is
exact by construction once keys are quantized.

Keys are product-quantized to a ``codes_per_block^n_blocks`` lattice (``n_blocks`` sub-vectors, one
learned codebook each; the per-block simplex perturbation group is isomorphic to additive logits, which
is what makes "cell" a group-theoretic object and not just a bucket). Attention weights are then
CELL-CONSTANT over the far field, so the softmax over every evicted token collapses exactly to

    sum_cells  count_c * exp(q . k_c / sqrt(d)) * vbar_c
    ------------------------------------------------------        (values stay continuous!)
    sum_cells  count_c * exp(q . k_c / sqrt(d))  + near mass

-- an identity, not an approximation (falsified 2026-07-12: max error 4e-17 numpy / 1e-9 on a trained
torch model; ``experiments/group_attention/RESULTS.md``). Per-query far-field cost is O(occupied cells),
independent of context length; the trained falsification model occupied 49 of 65,536 possible cells,
flat from 4k to 64k context.

Three deliverables live here:

- :class:`QuantizedKeyAttentionSpine` -- a :class:`~mixle.experimental.context_spine.ContextMechanism`:
  exact RoPE'd softmax over a sliding near window, plus a bounded far-field CELL STORE (integer counts,
  value sums, code vectors) that evicted tokens fold into. Near and far are normalized JOINTLY in one
  softmax (``log count`` enters as a logit offset), so the whole mechanism IS softmax attention over the
  quantized stream -- unlike the E3 sketches' two separately-normalized branches. Keys quantize with a
  VQ-VAE commitment loss (straight-through estimator), added to the training loss only when gradients
  are enabled so eval probes still return pure cross-entropy.
- :class:`SlidingCellWindow` -- streaming windowed aggregation where eviction is EXACT integer
  subtraction in the group Z^cells (counts can never drift; value sums are float64 running sums).
- :class:`CellCountTree` -- a Fenwick tree of sparse cell-count nodes: O(log n) updates, O(log n)-node
  prefix merges, and arbitrary-window counts via ``prefix(hi) - prefix(lo)`` using the group inverse
  (integer subtraction), never a float cancellation.

Positional honesty: the far field is position-free -- a bag of typed content. Retrieval across the far
field is by content (associative recall), not by position; tasks that need positional order beyond the
near window (e.g. long-range copy) are structurally out of reach, and the E7 referee is expected to show
exactly that. The near window keeps full RoPE'd positional attention.

Visibility is a pure function of (query position p, key position j): j is attended EXACTLY iff
``p - j < window``, and QUANTIZED (position-free) otherwise -- regardless of where chunk boundaries
fall. Aggregation into cells happens lazily at chunk ends, but tokens past the window horizon that
have not folded yet are attended through a "gap bank" of count-1 quantized keys, which by the collapse
identity is the same distribution their folded cells produce. Without the gap bank there is a
chunk-size-dependent blind spot (evicted but unfolded tokens invisible), which breaks
chunking-invariance for every layer past the first -- the streaming-equivalence test caught exactly
that failure before the gap bank existed. Streaming with any chunk size <= window therefore yields
identical integer cell contents after any prefix, and probe losses equal to float-addition-order
tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

if _HAS_TORCH:
    import torch.nn as nn

    from mixle.experimental.context_spine import _apply_rope, _rope_angles
    from mixle.experimental.sketch_state_attention import _transformer_block

__all__ = [
    "CellCountTree",
    "ProductQuantizer",
    "QuantizedKeyAttentionSpine",
    "QuantizedKeyCellState",
    "SlidingCellWindow",
]


# ---------------------------------------------------------------------------------------------------------
# Integer cell aggregation -- torch-free, usable by any consumer of quantized far fields.
# ---------------------------------------------------------------------------------------------------------


class SlidingCellWindow:
    """Sliding-window cell aggregation with EXACT group eviction.

    Per-cell integer counts live in Z^cells, a true group under addition: pushing a token adds 1 to its
    cell, evicting the token that scrolls out subtracts 1 -- exact integer arithmetic, so the window's
    counts after any number of operations are identical to recomputing from scratch (no drift, ever).
    Value sums are float64 running sums maintained by the same add/subtract; those are floats, so they
    carry ~1e-12-scale addition-order drift relative to recomputation -- the group-exactness guarantee
    is about the counts. Cells whose count reaches zero are deleted, so ``counts`` always enumerates
    exactly the occupied cells.
    """

    def __init__(self, window: int, value_dim: int) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)
        self.value_dim = int(value_dim)
        self._ring: list[tuple[int, np.ndarray]] = []
        self._start = 0  # ring is a list-backed FIFO; _start avoids O(n) pops
        self.counts: dict[int, int] = {}
        self.value_sums: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._ring) - self._start

    def push(self, cell_id: int, value: Any) -> tuple[int, np.ndarray] | None:
        """Add one token; return the ``(cell_id, value)`` it evicted, or ``None`` if the window isn't full."""
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.value_dim,):
            raise ValueError(f"value must have shape ({self.value_dim},), got {value.shape}")
        cell_id = int(cell_id)
        self._ring.append((cell_id, value))
        self.counts[cell_id] = self.counts.get(cell_id, 0) + 1
        self.value_sums[cell_id] = self.value_sums.get(cell_id, np.zeros(self.value_dim)) + value

        if len(self) <= self.window:
            return None
        old_id, old_value = self._ring[self._start]
        self._start += 1
        if self._start > 2 * self.window:  # amortized compaction
            self._ring = self._ring[self._start :]
            self._start = 0
        remaining = self.counts[old_id] - 1  # the group inverse: exact integer subtraction
        if remaining == 0:
            del self.counts[old_id]
            del self.value_sums[old_id]
        else:
            self.counts[old_id] = remaining
            self.value_sums[old_id] = self.value_sums[old_id] - old_value
        return old_id, old_value

    def totals(self) -> tuple[dict[int, int], dict[int, np.ndarray]]:
        """Snapshot of ``(counts, value_sums)`` over occupied cells (copies; safe to mutate)."""
        return dict(self.counts), {k: v.copy() for k, v in self.value_sums.items()}


class CellCountTree:
    """Fenwick (binary indexed) tree over stream positions holding sparse per-cell integer counts.

    ``add(pos, cell_id)`` touches O(log capacity) nodes; ``prefix(pos)`` merges O(log capacity) sparse
    dicts; ``range_counts(lo, hi)`` is ``prefix(hi) - prefix(lo)`` -- the Z^cells group inverse, an exact
    integer subtraction whose zero entries genuinely cancel (they are removed, not left as float dust).
    This answers arbitrary-window cell-count queries in O(log n) node merges, where the plain
    :class:`SlidingCellWindow` handles only the single fixed trailing window. ``last_touches`` receipts
    the node count of the most recent operation so tests can PROVE the O(log n) claim rather than assert
    it from theory.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._nodes: list[dict[int, int]] = [{} for _ in range(self.capacity + 1)]
        self.last_touches = 0

    def add(self, pos: int, cell_id: int, count: int = 1) -> None:
        """Record ``count`` tokens of ``cell_id`` at stream position ``pos`` (0-based, < capacity)."""
        if not 0 <= pos < self.capacity:
            raise ValueError(f"pos must be in [0, {self.capacity}), got {pos}")
        cell_id = int(cell_id)
        touches = 0
        i = pos + 1
        while i <= self.capacity:
            node = self._nodes[i]
            node[cell_id] = node.get(cell_id, 0) + int(count)
            touches += 1
            i += i & (-i)
        self.last_touches = touches

    def prefix(self, pos: int) -> dict[int, int]:
        """Cell counts over positions ``[0, pos)``."""
        if not 0 <= pos <= self.capacity:
            raise ValueError(f"pos must be in [0, {self.capacity}], got {pos}")
        out: dict[int, int] = {}
        touches = 0
        i = pos
        while i > 0:
            for cell_id, count in self._nodes[i].items():
                out[cell_id] = out.get(cell_id, 0) + count
            touches += 1
            i -= i & (-i)
        self.last_touches = touches
        return out

    def range_counts(self, lo: int, hi: int) -> dict[int, int]:
        """Cell counts over positions ``[lo, hi)`` via the group inverse: ``prefix(hi) - prefix(lo)``."""
        if lo > hi:
            raise ValueError(f"need lo <= hi, got [{lo}, {hi})")
        upper = self.prefix(hi)
        upper_touches = self.last_touches
        lower = self.prefix(lo)
        self.last_touches += upper_touches
        for cell_id, count in lower.items():
            remaining = upper.get(cell_id, 0) - count  # exact integer subtraction
            if remaining == 0:
                upper.pop(cell_id, None)
            else:
                upper[cell_id] = remaining
        return upper


# ---------------------------------------------------------------------------------------------------------
# The E10 mechanism -- torch-gated, mirroring the E3 spines' scaffolding.
# ---------------------------------------------------------------------------------------------------------


@dataclass
class QuantizedKeyCellState:
    """Near window (stop-gradient KV cache, as E1) + far-field cell store, per layer.

    The cell store is a fixed-capacity slot table shared across nothing -- one table per (batch, head):
    ``codes[layer][b, h, c]`` is slot ``c``'s lattice code vector, ``counts[layer][b, h, c]`` its integer
    token count (0 = free slot), ``vsum[layer][b, h, c]`` the float value sum. Fixed capacity means the
    carried state is bounded regardless of context length -- the property the E7 state-byte budget
    measures. ``drops`` counts tokens discarded because every slot was occupied (the honesty receipt:
    zero on every run whose results you trust for exactness claims).
    """

    codes: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, max_cells, n_blocks) long
    counts: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, max_cells) long
    vsum: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, max_cells, head_dim) float
    cache_k: list[Any] = field(default_factory=list)  # per layer: (batch, cache_len<=window, n_head, head_dim) | None
    cache_v: list[Any] = field(default_factory=list)
    pos: int = 0
    drops: int = 0


if _HAS_TORCH:

    class ProductQuantizer(nn.Module):
        """Per-block learned codebooks: keys snap to the nearest code per sub-vector (VQ-VAE style).

        ``forward`` returns the straight-through quantized keys, the integer code vectors, and the
        commitment loss ``||k - sg(q)||^2 + beta ||q - sg(k)||^2`` that trains encoder toward codebook
        and codebook toward encoder. ``encode``/``reconstruct`` are the integer <-> vector halves the
        cell store uses: ``reconstruct`` is a live embedding lookup, so far-field attention scores
        backpropagate into the codebooks directly.

        ``codebook_update="ema"`` switches the codebook half to exponential-moving-average cluster
        updates (van den Oord et al.'s VQ-VAE-2 recipe) with dead-code reseeding: codes track the
        decayed mean of the sub-vectors assigned to them instead of descending the commitment
        gradient, and a code whose decayed occupancy falls below ``reseed_threshold`` is re-seeded
        to a random sub-vector from the current batch. This is the standard remedy for the VQ
        optimization friction the falsification measured (the quantized arm needing ~2x the dense
        arm's step budget -- ``experiments/group_attention/RESULTS.md``); the acceptance experiment
        ``experiments/group_attention/ema_friction.py`` quantifies the reduction. In EMA mode the
        commitment loss keeps only the encoder-side term (the codebook no longer needs gradients)
        and the codebooks stop receiving far-field score gradients (they are buffers in all but
        name) -- the encoder path is unchanged either way.
        """

        def __init__(
            self,
            head_dim: int,
            *,
            n_blocks: int,
            codes_per_block: int,
            beta: float = 0.25,
            codebook_update: str = "gradient",
            ema_decay: float = 0.99,
            ema_eps: float = 1e-5,
            reseed_threshold: float = 0.1,
        ) -> None:
            super().__init__()
            if head_dim % n_blocks != 0:
                raise ValueError(f"head_dim={head_dim} must divide into n_blocks={n_blocks}")
            if codebook_update not in ("gradient", "ema"):
                raise ValueError(f"codebook_update must be 'gradient' or 'ema', got {codebook_update!r}")
            self.head_dim = int(head_dim)
            self.n_blocks = int(n_blocks)
            self.codes_per_block = int(codes_per_block)
            self.sub_dim = head_dim // n_blocks
            self.beta = float(beta)
            self.codebook_update = codebook_update
            self.ema_decay = float(ema_decay)
            self.ema_eps = float(ema_eps)
            self.reseed_threshold = float(reseed_threshold)
            init = torch.randn(self.n_blocks, self.codes_per_block, self.sub_dim) / (head_dim**0.5)
            self.codebooks = nn.Parameter(init, requires_grad=codebook_update == "gradient")
            if codebook_update == "ema":
                self.register_buffer("_ema_cluster_size", torch.ones(self.n_blocks, self.codes_per_block))
                self.register_buffer("_ema_embed_avg", init.clone())
                self.register_buffer("_ema_initialized", torch.zeros((), dtype=torch.bool))

        def _ema_update(self, k: Any, codes: Any) -> None:
            """One EMA cluster step from a batch of (detached) keys and their code assignments."""
            with torch.no_grad():
                kb = k.detach().reshape(-1, self.n_blocks, self.sub_dim)
                if not bool(self._ema_initialized) and kb.shape[0] > 0:
                    # data-dependent init: seed every code from ACTUAL first-batch sub-vectors. The
                    # random N(0, 1/sqrt(d)) init has the wrong scale relative to the encoder's real
                    # key activations, so early assignments collapse onto a few codes and every
                    # update scheme spends its first phase waiting for the encoder to reorganize --
                    # the decay-insensitive ~1.6x friction floor measured in ema_friction.py.
                    for b_idx in range(self.n_blocks):
                        picks = torch.randperm(kb.shape[0])[: self.codes_per_block]
                        if len(picks) < self.codes_per_block:
                            picks = torch.randint(0, kb.shape[0], (self.codes_per_block,))
                        self.codebooks.data[b_idx] = kb[picks, b_idx]
                        self._ema_embed_avg[b_idx] = kb[picks, b_idx]
                        self._ema_cluster_size[b_idx].fill_(1.0)
                    self._ema_initialized.fill_(True)
                    codes = self.encode(k)  # re-assign against the seeded codebooks
                flat = codes.detach().reshape(-1, self.n_blocks)
                for b_idx in range(self.n_blocks):
                    assign = flat[:, b_idx]
                    n = torch.bincount(assign, minlength=self.codes_per_block).to(kb.dtype)
                    sums = torch.zeros(self.codes_per_block, self.sub_dim, dtype=kb.dtype, device=kb.device)
                    sums.index_add_(0, assign, kb[:, b_idx])
                    self._ema_cluster_size[b_idx].mul_(self.ema_decay).add_(n, alpha=1.0 - self.ema_decay)
                    self._ema_embed_avg[b_idx].mul_(self.ema_decay).add_(sums, alpha=1.0 - self.ema_decay)
                    # Laplace-smoothed normalization keeps rarely-hit codes finite
                    size = self._ema_cluster_size[b_idx]
                    total = size.sum()
                    smoothed = (size + self.ema_eps) / (total + self.codes_per_block * self.ema_eps) * total
                    self.codebooks.data[b_idx] = self._ema_embed_avg[b_idx] / smoothed[:, None]
                    # dead-code reseed: a code nobody uses re-enters at a random batch sub-vector,
                    # so early collapse (everything mapping to 2 cells) self-repairs
                    dead = size < self.reseed_threshold
                    if bool(dead.any()) and kb.shape[0] > 0:
                        picks = torch.randint(0, kb.shape[0], (int(dead.sum()),), device=kb.device)
                        self.codebooks.data[b_idx][dead] = kb[picks, b_idx]
                        self._ema_embed_avg[b_idx][dead] = kb[picks, b_idx]
                        self._ema_cluster_size[b_idx][dead] = 1.0

        def encode(self, k: Any) -> Any:
            """``(..., head_dim) -> (..., n_blocks)`` nearest-code indices (no gradient; argmin is discrete)."""
            blocks = k.detach().reshape(*k.shape[:-1], self.n_blocks, self.sub_dim)
            dist = ((blocks.unsqueeze(-2) - self.codebooks) ** 2).sum(-1)  # (..., n_blocks, codes_per_block)
            return dist.argmin(-1)

        def reconstruct(self, codes: Any) -> Any:
            """``(..., n_blocks) -> (..., head_dim)`` codebook lookup; differentiable w.r.t. the codebooks."""
            gathered = [self.codebooks[b_idx][codes[..., b_idx]] for b_idx in range(self.n_blocks)]
            return torch.cat(gathered, dim=-1)

        def forward(self, k: Any) -> tuple[Any, Any, Any]:
            codes = self.encode(k)
            quant = self.reconstruct(codes)
            blocks_flat = k  # commitment in the full-key metric == sum of per-block metrics
            if self.codebook_update == "ema":
                # encoder-side pull only; the codebook is trained by the EMA cluster step (applied
                # AFTER this batch used the current codes -- the standard VQ-VAE-EMA ordering, so
                # the returned codes and quantized keys stay mutually consistent)
                commit = self.beta * F.mse_loss(blocks_flat, quant.detach())
                if self.training and torch.is_grad_enabled():
                    self._ema_update(k, codes)
            else:
                commit = F.mse_loss(blocks_flat, quant.detach()) + self.beta * F.mse_loss(quant, blocks_flat.detach())
            k_q = k + (quant - k).detach()  # straight-through
            return k_q, codes, commit

    def _windowed_logits(
        q_raw: Any,
        k_raw: Any,
        v_raw: Any,
        cache_k_raw: Any,
        cache_v_raw: Any,
        *,
        window: int,
        head_dim: int,
        pos: int,
        pq: ProductQuantizer,
    ) -> tuple[Any, Any, Any, Any, Any, Any | None, Any | None]:
        """``sketch_state_attention._local_window_step``'s exact construction with the softmax DEFERRED
        and a quantized GAP bank added: returns ``(near_logits, gap_logits, values, new_cache_k,
        new_cache_v, evicted_k_raw, evicted_v_raw)``, logits of shape ``(b, n_head, t, L)`` over the
        ``L = cache + chunk`` unfolded tokens and values ``(b, n_head, L, head_dim)``.

        ``near_logits[.., p, j]`` is the RoPE'd exact score where ``0 <= p - j < window``, ``-inf``
        elsewhere. ``gap_logits[.., p, j]`` is the position-free QUANTIZED score (raw query against the
        token's snapped key -- a count-1 cell) where ``p - j >= window``: tokens past the window horizon
        that have not yet folded into the far store. By the collapse identity, attending them this way
        is exactly what their folded cells will produce, so visibility never depends on where a chunk
        boundary happens to fall. The caller normalizes far cells + gap + near in ONE softmax -- the
        collapse identity is about a single distribution over {tokens}, not separately-normalized
        branches."""
        b, t, n_head, _ = q_raw.shape
        device = q_raw.device
        query_positions = torch.arange(pos, pos + t, device=device)

        if cache_k_raw is not None:
            cache_len = cache_k_raw.shape[1]
            key_positions = torch.arange(pos - cache_len, pos + t, device=device)
            k_full_raw = torch.cat([cache_k_raw, k_raw], dim=1)
            v_full_raw = torch.cat([cache_v_raw, v_raw], dim=1)
        else:
            key_positions = query_positions
            k_full_raw, v_full_raw = k_raw, v_raw

        sin_q, cos_q = _rope_angles(query_positions, head_dim)
        sin_k, cos_k = _rope_angles(key_positions, head_dim)
        q = _apply_rope(q_raw, sin_q, cos_q)
        k_full = _apply_rope(k_full_raw, sin_k, cos_k)

        delta = query_positions[:, None] - key_positions[None, :]
        near_allowed = (delta >= 0) & (delta < window)
        near_mask = torch.zeros(t, key_positions.shape[0], device=device)
        near_mask = near_mask.masked_fill(~near_allowed, float("-inf"))

        qh = q.transpose(1, 2)  # (b, n_head, t, head_dim)
        kh = k_full.transpose(1, 2)
        vh = v_full_raw.transpose(1, 2)  # values are not rotated (RoPE only orients the QK dot product)
        near_logits = (qh @ kh.transpose(-2, -1)) / (head_dim**0.5) + near_mask[None, None]

        gap_mask = torch.zeros(t, key_positions.shape[0], device=device)
        gap_mask = gap_mask.masked_fill(delta < window, float("-inf"))  # gap = strictly beyond the window
        # Each gap token is a count-1 cell (log 1 = 0 offset), attended through its SNAPPED key in
        # straight-through form: numerically the codebook vector (the collapse identity needs exactly
        # that), but gradient-wise `k + (quant - k).detach()`, so far-field retrieval error reaches the
        # KEY ENCODER. This is the falsification experiment's training path; without it the encoder only
        # ever feels the commitment pull, and its keys collapse into one or two cells (measured: 2/64
        # occupied on the E7 referee) -- a far field that has organized nothing.
        quant = pq.reconstruct(pq.encode(k_full_raw))
        k_snap = k_full_raw + (quant - k_full_raw).detach()
        qh_raw = q_raw.transpose(1, 2)
        gap_logits = (qh_raw @ k_snap.transpose(1, 2).transpose(-2, -1)) / (head_dim**0.5) + gap_mask[None, None]

        total_len = k_full_raw.shape[1]
        if total_len > window:
            n_evict = total_len - window
            evicted_k_raw = k_full_raw[:, :n_evict]
            evicted_v_raw = v_full_raw[:, :n_evict]
        else:
            evicted_k_raw = evicted_v_raw = None
        new_cache_k = k_full_raw[:, -window:]
        new_cache_v = v_full_raw[:, -window:]
        return near_logits, gap_logits, vh, new_cache_k, new_cache_v, evicted_k_raw, evicted_v_raw

    def quantized_softmax_weights(logits: Any, *, bits: int, span: float = 24.0) -> Any:
        """Unnormalized softmax weights through the Q-LSE grid: ONE ``2^bits`` exp table, no real exp.

        ``mixle.engines.qlut.quantized_logsumexp``'s exact semantics, elementwise: shift by the
        row max, round to the ``span / 2^bits`` grid, clamp scores more than ``span`` below the max
        into the bottom bin, and read ``exp`` off the precomputed table (``-inf`` rows get weight
        0, matching softmax's treatment of masked slots). Dividing by the row sum gives softmax
        weights within ``exp(+-lse_error_bound(bits, span)) - 1`` relative of exact -- the same
        grid half-step bound, now carried through the attention READOUT rather than just the
        scorer. The histogram+dot evaluation of these same numbers is the O(2^bits) form; here the
        table gather is per element because the readout needs the per-slot weights against values.
        """
        levels = 1 << bits
        delta = span / levels
        finite = torch.isfinite(logits)
        m = logits.masked_fill(~finite, float("-inf")).amax(dim=-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))  # all-masked rows: any shift works
        idx = torch.clamp(torch.round((logits - m) / delta).long() + levels - 1, min=0, max=levels - 1)
        table = torch.exp((torch.arange(levels, device=logits.device, dtype=torch.float64) - (levels - 1)) * delta).to(
            logits.dtype
        )
        return table[idx] * finite.to(logits.dtype)

    def _far_bank(codes: Any, counts: Any, vsum: Any, q_raw: Any, pq: ProductQuantizer) -> tuple[Any, Any]:
        """Far-field logits and mean values from the cell store.

        ``logits[b, h, t, c] = q . k_c / sqrt(d) + log(count_c)`` for occupied slots, ``-inf`` for free
        ones -- the ``log count`` offset is exactly what makes one joint softmax over cells reproduce
        softmax over the underlying tokens (cell-constant weights sum to ``count * exp(score)``).
        Queries enter RAW (no RoPE): the far field is position-free by design. Returns
        ``(logits (b, n_head, t, C), vbar (b, n_head, C, head_dim))``.
        """
        occupied = counts > 0  # (b, n_head, C)
        k_cells = pq.reconstruct(codes)  # (b, n_head, C, head_dim); live codebook lookup
        qh = q_raw.transpose(1, 2)  # (b, n_head, t, head_dim)
        scores = (qh @ k_cells.transpose(-2, -1)) / (pq.head_dim**0.5)  # (b, n_head, t, C)
        log_count = counts.clamp(min=1).to(scores.dtype).log()
        logits = scores + log_count[:, :, None, :]
        logits = logits.masked_fill(~occupied[:, :, None, :], float("-inf"))
        vbar = vsum / counts.clamp(min=1).to(vsum.dtype)[..., None]
        return logits, vbar

    def _fold_evictions(
        codes: Any,
        counts: Any,
        vsum: Any,
        evicted_codes: Any,
        evicted_v: Any,
        *,
        codes_per_block: int,
    ) -> tuple[Any, Any, Any, int]:
        """Merge evicted tokens into the cell store; returns ``(codes, counts, vsum, dropped)``.

        Counts/codes are integer bookkeeping (no autograd); value sums are combined with OUT-OF-PLACE
        ``index_put(accumulate=True)`` so gradients flow from evicted values into the store within the
        TBPTT horizon (mirroring how the FD spine's sketch update carries the chunk's graph until
        ``detach``). Tokens whose cell has no slot and no free slot remain are DROPPED and counted --
        never silently.
        """
        b, n_evict, n_head, n_blocks = evicted_codes.shape
        device = evicted_codes.device
        radix = codes_per_block ** torch.arange(n_blocks, device=device, dtype=torch.long)

        slot_flat = (codes * radix).sum(-1)  # (b, n_head, C); free slots hold stale ids, masked below
        new_codes = codes.clone()
        dropped = 0

        put_b: list[int] = []
        put_h: list[int] = []
        put_slot: list[int] = []
        put_count: list[int] = []
        put_vs: list[Any] = []

        for bi in range(b):
            for hi in range(n_head):
                ids = (evicted_codes[bi, :, hi] * radix).sum(-1)  # (n_evict,)
                uniq, inv = torch.unique(ids, return_inverse=True)
                cell_counts = torch.bincount(inv, minlength=len(uniq))
                v_bh = evicted_v[bi, :, hi]  # (n_evict, head_dim)
                cell_vs = torch.zeros(len(uniq), v_bh.shape[-1], dtype=v_bh.dtype, device=device)
                cell_vs = cell_vs.index_add(0, inv, v_bh)  # out-of-place: keeps the autograd graph

                occ = counts[bi, hi] > 0
                row_flat = slot_flat[bi, hi]
                free = (~occ).nonzero().flatten().tolist()
                pending: dict[int, int] = {}
                for j in range(len(uniq)):
                    cid = int(uniq[j])
                    match = ((row_flat == cid) & occ).nonzero().flatten()
                    if len(match) > 0:
                        slot = int(match[0])
                    elif cid in pending:
                        slot = pending[cid]
                    elif free:
                        slot = free.pop(0)
                        pending[cid] = slot
                        first = int((inv == j).nonzero()[0])
                        new_codes[bi, hi, slot] = evicted_codes[bi, first, hi]
                    else:
                        dropped += int(cell_counts[j])
                        continue
                    put_b.append(bi)
                    put_h.append(hi)
                    put_slot.append(slot)
                    put_count.append(int(cell_counts[j]))
                    put_vs.append(cell_vs[j])

        if not put_b:
            return new_codes, counts, vsum, dropped
        idx = (
            torch.tensor(put_b, device=device),
            torch.tensor(put_h, device=device),
            torch.tensor(put_slot, device=device),
        )
        new_counts = counts.index_put(idx, torch.tensor(put_count, device=device, dtype=counts.dtype), accumulate=True)
        new_vsum = vsum.index_put(idx, torch.stack(put_vs), accumulate=True)
        return new_codes, new_counts, new_vsum, dropped

    class QuantizedKeyAttentionSpine(nn.Module):
        """E10: exact near-window attention + quantized-key far-field cell store, one joint softmax.

        Same pre-norm residual scaffolding as every Track-E spine (``_transformer_block``); what varies
        is the far field: evicted tokens quantize (one shared :class:`ProductQuantizer` per layer, across
        heads, as in the falsification experiment) and fold into a bounded slot table of
        ``(code vector, integer count, value sum)``. A query attends over {far cells with ``log count``
        logit offsets} + {gap tokens past the window horizon but not yet folded, as count-1 quantized
        cells} + {RoPE'd exact near window}, normalized together in ONE softmax -- by the collapse
        identity this equals softmax attention over the entire stream in which every key more than
        ``window`` positions back is quantized and position-free, independent of chunk boundaries. The
        commitment loss trains keys toward the lattice; it joins the returned loss only while gradients
        are enabled, so ``torch.no_grad()`` probes (the E7 referee's measurement mode) see pure
        cross-entropy.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int = 64,
            n_blocks: int = 2,
            codes_per_block: int = 16,
            max_cells: int = 128,
            commit_weight: float = 1.0,
            codebook_update: str = "gradient",
            lse_bits: int | None = None,
            lse_span: float = 24.0,
        ) -> None:
            super().__init__()
            assert d_model % n_head == 0
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = int(window)
            self.n_blocks = int(n_blocks)
            self.codes_per_block = int(codes_per_block)
            self.max_cells = int(max_cells)
            self.commit_weight = float(commit_weight)
            # Q-LSE readout: with lse_bits set, INFERENCE steps (no grad) normalize the joint
            # softmax through the quantized-exp table (quantized_softmax_weights); training steps
            # always use exact exp -- the same exact-when-learning discipline as the fused E-steps.
            self.lse_bits = None if lse_bits is None else int(lse_bits)
            self.lse_span = float(lse_span)
            (self.tok, self.qkv, self.proj, self.ln1, self.ln2, self.mlp, self.ln_f, self.head) = _transformer_block(
                vocab, d_model, n_layer, n_head
            )
            self.pq = nn.ModuleList(
                [
                    ProductQuantizer(
                        self.head_dim,
                        n_blocks=n_blocks,
                        codes_per_block=codes_per_block,
                        codebook_update=codebook_update,
                    )
                    for _ in range(n_layer)
                ]
            )

        def init_state(self, batch_size: int, *, device: str = "cpu") -> QuantizedKeyCellState:
            shape = (batch_size, self.n_head, self.max_cells)
            return QuantizedKeyCellState(
                codes=[
                    torch.zeros(*shape, self.n_blocks, dtype=torch.long, device=device) for _ in range(self.n_layer)
                ],
                counts=[torch.zeros(*shape, dtype=torch.long, device=device) for _ in range(self.n_layer)],
                vsum=[torch.zeros(*shape, self.head_dim, device=device) for _ in range(self.n_layer)],
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                pos=0,
                drops=0,
            )

        def detach(self, state: QuantizedKeyCellState) -> QuantizedKeyCellState:
            return QuantizedKeyCellState(
                codes=list(state.codes),  # integer tensors carry no graph
                counts=list(state.counts),
                vsum=[v.detach() for v in state.vsum],
                cache_k=[k.detach() if k is not None else None for k in state.cache_k],
                cache_v=[v.detach() if v is not None else None for v in state.cache_v],
                pos=state.pos,
                drops=state.drops,
            )

        def occupancy_receipt(self, state: QuantizedKeyCellState) -> dict[str, Any]:
            """How full the far field is -- the claim ``O(occupied cells)`` is only honest with this attached."""
            occupied = [int((c > 0).sum(dim=-1).max()) for c in state.counts]
            return {
                "occupied_cells_per_layer": occupied,
                "capacity": self.max_cells,
                "possible_cells": self.codes_per_block**self.n_blocks,
                "dropped_tokens": state.drops,
            }

        def step(self, state: QuantizedKeyCellState, chunk: tuple[Any, Any]) -> tuple[QuantizedKeyCellState, Any]:
            x, y = chunk
            b, t = x.shape

            h = self.tok(x)
            new_codes: list[Any] = []
            new_counts: list[Any] = []
            new_vsum: list[Any] = []
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            drops = state.drops
            commit_total = h.new_zeros(())
            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q_raw, k_raw, v_raw = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

                _, _, commit = self.pq[layer](k_raw)  # commitment on the whole chunk's keys, every step
                commit_total = commit_total + commit

                near_logits, gap_logits, values, cache_k, cache_v, evicted_k, evicted_v = _windowed_logits(
                    q_raw,
                    k_raw,
                    v_raw,
                    state.cache_k[layer],
                    state.cache_v[layer],
                    window=self.window,
                    head_dim=self.head_dim,
                    pos=state.pos,
                    pq=self.pq[layer],
                )
                far_logits, vbar = _far_bank(
                    state.codes[layer], state.counts[layer], state.vsum[layer], q_raw, self.pq[layer]
                )

                # One softmax over {far cells} + {gap tokens, quantized} + {near tokens, exact}: softmax
                # attention over the whole stream with far keys quantized -- the collapse identity's LHS.
                joint_logits = torch.cat([far_logits, gap_logits, near_logits], dim=-1)
                if self.lse_bits is not None and not torch.is_grad_enabled():
                    w = quantized_softmax_weights(joint_logits, bits=self.lse_bits, span=self.lse_span)
                    joint = w / w.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(w.dtype).tiny)
                else:
                    joint = joint_logits.softmax(dim=-1)
                n_cells = far_logits.shape[-1]
                n_unfolded = gap_logits.shape[-1]
                out = (
                    joint[..., :n_cells] @ vbar
                    + joint[..., n_cells : n_cells + n_unfolded] @ values
                    + joint[..., n_cells + n_unfolded :] @ values
                )  # (b, n_head, t, head_dim)
                out = out.transpose(1, 2).reshape(b, t, self.d_model)

                h = h + self.proj[layer](out)
                h = h + self.mlp[layer](self.ln2[layer](h))

                codes_l, counts_l, vsum_l = state.codes[layer], state.counts[layer], state.vsum[layer]
                if evicted_k is not None:
                    evicted_codes = self.pq[layer].encode(evicted_k)  # (b, n_evict, n_head, n_blocks)
                    codes_l, counts_l, vsum_l, dropped = _fold_evictions(
                        codes_l,
                        counts_l,
                        vsum_l,
                        evicted_codes,
                        evicted_v,
                        codes_per_block=self.codes_per_block,
                    )
                    drops += dropped
                new_codes.append(codes_l)
                new_counts.append(counts_l)
                new_vsum.append(vsum_l)
                new_cache_k.append(cache_k)
                new_cache_v.append(cache_v)

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))
            if torch.is_grad_enabled():
                loss = loss + self.commit_weight * commit_total / self.n_layer

            new_state = QuantizedKeyCellState(
                codes=new_codes,
                counts=new_counts,
                vsum=new_vsum,
                cache_k=new_cache_k,
                cache_v=new_cache_v,
                pos=state.pos + t,
                drops=drops,
            )
            return new_state, loss
