"""E3: sketch-state attention -- oblivious (data-independent update rule) far-field states with a
*provable* approximation guarantee, contrasted with E2's adaptive/learned far-field state. See
``notes/designs/E3.md`` for the full design (citations, the augmented-row cross-covariance trick, the
tensor-sketch FFT derivation, and the honestly-flagged tension between FD's SVD shrink and the card's
"All linear => exact gradients" framing).

Three mechanisms, all implementing :class:`~mixle.experimental.context_spine.ContextMechanism`, differing
only in how the stream of per-token key/value pairs ``(phi(k_t), v_t)`` is compressed into carried state:

- **(a) `LinearAttentionSpine`** -- exact, unbounded-rank running sum ``S = sum phi(k_t) v_t^T``,
  ``Z = sum phi(k_t)`` (Katharopoulos et al. 2020 kernel trick, ``phi = elu(x) + 1``). No local window: the
  whole stream is the linear-attention prefix, chunked as a running cumulative sum (bit-identical to a
  single non-chunked pass, since carrying ``S``/``Z`` across chunk boundaries IS the prefix sum's carry).
  This is the fixed-byte-size reference point (b)/(c) approximate.
- **(b) `FrequentDirectionsSpine`** -- a small exact local window (`SlidingWindowSpine`-style stop-gradient
  cache) plus a Frequent Directions sketch (Liberty, KDD 2013) of the augmented rows
  ``[phi(k_t) ; v_t]`` for every token once it scrolls out of the local window. ``B`` is literally
  ``ell x (d_phi + d_v)`` with genuine zero rows between shrinks (Liberty's Algorithm 1, not a
  rank-compacted variant) -- this is what makes the deterministic Theorem 1.1 bound test meaningful. The
  normalizer ``Z = sum phi(k_t)`` is tracked exactly alongside the sketch (cheap, O(d_phi) per step; the
  Proposed API's illustrative dataclass didn't spell this field out, but the design note's own Algorithm
  section requires it -- there is no valid FD readout without it).
- **(c) `TensorSketchSpine`** -- same local-window split, but the far-field accumulator is a Count-Sketch +
  FFT-circular-convolution tensor sketch (Pham & Pagh, KDD 2013) of ``phi(k_t)``, capturing degree-``p``
  polynomial-kernel interactions FD's/`(a)`'s linear rows cannot represent, at the cost of an in-expectation
  (not worst-case) guarantee.

**Local window vs pure prefix.** (a)'s own Algorithm section describes no local softmax component at all --
it degenerates the WHOLE stream to a linear-attention kernel that carries exact state, matching its
constructor (no ``window`` parameter). (b)/(c) each keep a small local exact-softmax window (their
constructors take ``window=64``) plus the sketch as an additive far-field term -- "local half unchanged from
`SlidingWindowSpine`, far-field half is what varies" (design note's Proposed API section, and the "Do NOT
fold the near-field/far-field split into one undifferentiated block" rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

if _HAS_TORCH:
    from mixle.experimental.context_spine import _apply_rope, _rope_angles

# The three *Spine classes subclass nn.Module and are defined only under `if _HAS_TORCH`, so
# advertising them unconditionally made __all__ describe a module that does not exist without the
# optional torch extra: `from ... import *` raised AttributeError, and a direct import of any Spine
# raised ImportError with no indication that a missing optional dependency was the reason. __all__ is
# a promise about what this module exports, so it has to be conditioned on the same thing the
# definitions are.
__all__ = [
    "LinearAttentionState",
    "FrequentDirectionsState",
    "TensorSketchState",
    "frequent_directions_update",
    "frequent_directions_error_bound",
    "tensor_sketch_project",
    "make_tensor_sketch_hashes",
    "fd_misfit_receipt",
    "tensor_sketch_misfit_receipt",
    "E3_UNAVAILABLE_COMPARISONS",
]

if _HAS_TORCH:
    __all__ += ["LinearAttentionSpine", "FrequentDirectionsSpine", "TensorSketchSpine"]

# Retained for compatibility with the roadmap-era API. All formerly missing comparison pieces now exist.
E3_UNAVAILABLE_COMPARISONS: dict[str, str] = {}


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("mixle.experimental.sketch_state_attention requires torch.")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive exact integer, got {value}")
    return int(value)


# ---------------------------------------------------------------------------------------------------------
# Frequent Directions -- Liberty (KDD 2013), Algorithm 1, literal ell x d shape with genuine zero rows.
# ---------------------------------------------------------------------------------------------------------

if _HAS_TORCH:

    def _fd_insert_row(B: Any, row: Any, ell: int) -> Any:
        """Insert one row (batched over arbitrary leading dims) into ``B`` (``*batch, ell, d``), matching
        Liberty's Algorithm 1 steps 2a/2b **in their literal order**: (2a) insert ``row`` into any all-zero
        row of ``B`` first (a zero row is guaranteed to exist on entry -- ``B`` starts all-zero and this
        invariant is maintained by step 2b below); (2b) only THEN check whether ``B`` -- now containing the
        just-inserted row -- has no all-zero row left, and if so, shrink. Doing the zero-row check before
        insertion (as an earlier version of this function did) is a subtle deviation: it tests "was B full
        before this row arrived" rather than the design note's literal "if B now [after inserting a_t] has
        no all-zero row", and -- more importantly for testability -- it fuses the shrink and the following
        insert into one atomic step so the freed zero row is never externally observable. This order leaves
        a genuine all-zero row in the returned ``B`` whenever a shrink just fired, persisting until the next
        row arrives -- exactly the freed-row invariant ``notes/designs/E3.md``'s Test plan #1 companion test
        checks. Generalized to operate over an arbitrary batch of independent streams at once (each stream's
        fill/shrink timing is data-independent -- only insertion COUNT determines it -- but this re-derives
        the zero-row mask from ``B`` itself every call rather than assuming batch-wide synchronization, so it
        is correct even if that invariant is ever violated, e.g. by a future caller feeding streams out of
        lock-step)."""
        zero_mask = B.abs().sum(dim=-1) == 0  # (*batch, ell), before this row's insertion
        idx = zero_mask.float().argmax(dim=-1)  # first all-zero row index, (*batch,) -- guaranteed to exist
        idx_exp = idx[..., None, None].expand(*idx.shape, 1, B.shape[-1])
        B = B.scatter(dim=-2, index=idx_exp, src=row.unsqueeze(-2))  # 2a: insert a_t

        zero_mask_after = B.abs().sum(dim=-1) == 0  # (*batch, ell), after this row's insertion
        has_zero_after = zero_mask_after.any(dim=-1)  # (*batch,)
        if not bool(has_zero_after.all()):
            U, S, Vt = torch.linalg.svd(B, full_matrices=False)  # batched SVD over the last two dims
            delta = S[..., -1:] ** 2  # sigma_ell^2, the smallest singular value's square
            shrunk_s = torch.clamp(S**2 - delta, min=0.0).sqrt()
            b_shrunk = shrunk_s.unsqueeze(-1) * Vt  # exactly one zero row (the smallest-sigma one), per Liberty
            need_shrink = (~has_zero_after)[..., None, None]
            B = torch.where(need_shrink, b_shrunk, B)  # 2b: shrink -- leaves >=1 genuine zero row in B
        return B

    def frequent_directions_update(B: Any, rows: Any, ell: int) -> Any:
        """One FD ingest-and-shrink pass (Liberty 2013, Algorithm 1). ``B``: ``(ell, d)``, ``rows``:
        ``(m, d)`` new rows, inserted one at a time (insert into a zero row; shrink whenever none remains).
        Returns the updated ``(ell, d)`` ``B`` -- unbatched, matching the design note's Proposed API
        signature exactly (the batched spine-internal use reuses the same ``_fd_insert_row`` primitive)."""
        ell = _positive_integer(ell, "ell")
        if not torch.is_tensor(B) or not torch.is_tensor(rows) or B.ndim != 2 or rows.ndim != 2:
            raise TypeError("B and rows must be real floating-point matrices")
        if not B.is_floating_point() or not rows.is_floating_point():
            raise TypeError("B and rows must be real floating-point matrices")
        if B.shape[0] != ell or B.shape[1] == 0 or ell > B.shape[1]:
            raise ValueError("B must have shape (ell, d) with 1 <= ell <= d")
        if rows.shape[1] != B.shape[1] or rows.device != B.device or rows.dtype != B.dtype:
            raise ValueError("rows must match B's feature dimension, device, and dtype")
        if not bool(torch.isfinite(B).all().item()) or not bool(torch.isfinite(rows).all().item()):
            raise ValueError("B and rows must contain only finite values")
        if not bool((B.abs().sum(dim=-1) == 0).any().item()):
            raise ValueError("B must be a valid FD state containing at least one exact zero insertion row")
        B = B.clone()
        for t in range(rows.shape[0]):
            B = _fd_insert_row(B, rows[t], B.shape[0])
        return B

    def frequent_directions_error_bound(A: Any, B: Any, ell: int, k: int) -> float:
        """RHS of Liberty's Theorem 1.1: ``||A - A_k||_F^2 / (ell - k)`` -- ``A_k`` is ``A``'s best
        rank-``k`` approximation (Eckart-Young). Depends only on ``A``, ``ell``, ``k`` (not on ``B`` -- the
        theorem's guarantee is that ANY ``B`` produced by streaming ``A``'s rows through FD satisfies
        ``||A^T A - B^T B||_2 <= `` this quantity); ``B`` is accepted to match the design note's Proposed API
        signature and to allow a caller to sanity-check ``B.shape[0] == ell``."""
        ell = _positive_integer(ell, "ell")
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= int(k) < ell:
            raise ValueError(f"k must be an exact integer in [0, ell), got {k}")
        k = int(k)
        if not torch.is_tensor(A) or A.ndim != 2 or not A.is_floating_point() or A.shape[1] == 0:
            raise TypeError("A must be a real floating-point matrix with a non-empty feature dimension")
        if ell > A.shape[1]:
            raise ValueError(f"ell={ell} must not exceed A's feature dimension {A.shape[1]}")
        if not bool(torch.isfinite(A).all().item()):
            raise ValueError("A must contain only finite values")
        if B is not None:
            if (
                not torch.is_tensor(B)
                or B.ndim != 2
                or B.shape != (ell, A.shape[1])
                or B.device != A.device
                or B.dtype != A.dtype
            ):
                raise ValueError(f"B must have shape {(ell, A.shape[1])} on A's device and dtype")
            if not bool(torch.isfinite(B).all().item()):
                raise ValueError("B must contain only finite values")
        if k == 0:
            resid = torch.linalg.norm(A, ord="fro") ** 2
        else:
            s = torch.linalg.svdvals(A)
            resid = torch.sum(s[k:] ** 2)
        return float(resid / (ell - k))

    # -----------------------------------------------------------------------------------------------------
    # Tensor sketch -- Pham & Pagh (KDD 2013): Count Sketch + FFT circular convolution.
    # -----------------------------------------------------------------------------------------------------

    def make_tensor_sketch_hashes(
        d: int, *, sketch_dim: int, degree: int, seed: int, device: str = "cpu"
    ) -> tuple[list[Any], list[Any]]:
        """``degree`` independent ``(hash, sign)`` pairs, fixed at construction (the "oblivious" part --
        the hash/sign choice does not depend on the data). ``hash_i: [d] -> [sketch_dim]``,
        ``sign_i: [d] -> {-1, +1}``."""
        d = _positive_integer(d, "d")
        sketch_dim = _positive_integer(sketch_dim, "sketch_dim")
        degree = _positive_integer(degree, "degree")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError(f"seed must be an exact integer, got {seed}")
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        hashes, signs = [], []
        for _ in range(degree):
            h = torch.randint(0, sketch_dim, (d,), generator=gen).to(device)
            s = (torch.randint(0, 2, (d,), generator=gen).to(device).float() * 2 - 1).to(device)
            hashes.append(h)
            signs.append(s)
        return hashes, signs

    def _count_sketch(x: Any, h: Any, s: Any, sketch_dim: int) -> Any:
        """``CS(x)[j] = sum_{r: h(r)=j} s(r) x_r`` -- a linear map of ``x``'s last dimension into
        ``sketch_dim``, broadcast over any leading batch dims."""
        contrib = x * s
        out_shape = x.shape[:-1] + (sketch_dim,)
        cs = torch.zeros(out_shape, device=x.device, dtype=x.dtype)
        idx = h.expand(x.shape[:-1] + h.shape)
        cs.scatter_add_(-1, idx, contrib)
        return cs

    def tensor_sketch_project(x: Any, hashes: list[Any], signs: list[Any], sketch_dim: int) -> Any:
        """Degree-``len(hashes)`` tensor sketch of ``x`` (last dim is the feature dim) via count-sketch +
        FFT circular convolution (Pham & Pagh 2013): ``TS(x) = IFFT(prod_i FFT(CS_i(x)))``. The defining
        property this implements: ``TS(x)^T TS(y)`` is an unbiased estimator of ``(x^T y)^p`` for
        ``p = len(hashes)``, with variance ``O(1 / sketch_dim)``."""
        sketch_dim = _positive_integer(sketch_dim, "sketch_dim")
        if not torch.is_tensor(x) or not x.is_floating_point() or x.ndim < 1 or x.shape[-1] == 0:
            raise TypeError("x must be a real floating-point tensor with a non-empty final dimension")
        if not bool(torch.isfinite(x).all().item()):
            raise ValueError("x must contain only finite values")
        if not hashes or len(hashes) != len(signs):
            raise ValueError("hashes and signs must have equal non-zero degree")
        for index, (hash_values, sign_values) in enumerate(zip(hashes, signs)):
            if (
                not torch.is_tensor(hash_values)
                or hash_values.dtype != torch.long
                or hash_values.shape != (x.shape[-1],)
                or hash_values.device != x.device
            ):
                raise ValueError(f"hashes[{index}] must be a torch.long vector on x's device")
            if (
                not torch.is_tensor(sign_values)
                or sign_values.shape != (x.shape[-1],)
                or sign_values.device != x.device
            ):
                raise ValueError(f"signs[{index}] must be a vector on x's device")
            if bool(((hash_values < 0) | (hash_values >= sketch_dim)).any().item()):
                raise ValueError(f"hashes[{index}] contains an index outside [0, sketch_dim)")
            if not bool(((sign_values == -1) | (sign_values == 1)).all().item()):
                raise ValueError(f"signs[{index}] must contain only -1 and +1")
        prod = None
        for h, s in zip(hashes, signs):
            cs = _count_sketch(x, h, s, sketch_dim)
            spec = torch.fft.fft(cs.to(torch.float64))
            prod = spec if prod is None else prod * spec
        ts = torch.fft.ifft(prod).real
        return ts.to(x.dtype)

    # -----------------------------------------------------------------------------------------------------
    # Shared local-window near-field block (mirrors SlidingWindowSpine's shape exactly for (b)/(c);
    # (a) has no near field at all -- see module docstring).
    # -----------------------------------------------------------------------------------------------------

    def _phi(x: Any) -> Any:
        """``elu(x) + 1`` -- the standard linear-attention feature map (Katharopoulos et al. 2020), kept
        non-negative so the far-field normalizer ``phi(q) . Z`` cannot be zero or negative."""
        return F.elu(x) + 1.0

    def _local_window_step(
        q_raw: Any,
        k_raw: Any,
        v_raw: Any,
        cache_k_raw: Any,
        cache_v_raw: Any,
        *,
        window: int,
        head_dim: int,
        pos: int,
    ) -> tuple[Any, Any, Any, Any | None, Any | None]:
        """Windowed exact causal softmax attention over ``cache + chunk`` (RoPE'd, same construction as
        ``SlidingWindowSpine.step``), returning ``(out, new_cache_k_raw, new_cache_v_raw, evicted_k_raw,
        evicted_v_raw)``. ``evicted_*`` are the PRE-RoPE raw keys/values that scrolled out of the window
        this step (``None`` if nothing was evicted) -- exactly the tokens whose ``(phi(k_t), v_t)`` the
        caller folds into far-field state next."""
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
        allowed = (delta >= 0) & (delta < window)
        mask = torch.zeros(t, key_positions.shape[0], device=device)
        mask = mask.masked_fill(~allowed, float("-inf"))

        qh = q.transpose(1, 2)
        kh = k_full.transpose(1, 2)
        vh = v_full_raw.transpose(1, 2)  # values are not rotated (RoPE only orients the QK dot product)
        attn = (qh @ kh.transpose(-2, -1)) / (head_dim**0.5)
        attn = attn + mask[None, None]
        attn = attn.softmax(dim=-1)
        out = (attn @ vh).transpose(1, 2)  # (b, t, n_head, head_dim)

        total_len = k_full_raw.shape[1]
        if total_len > window:
            n_evict = total_len - window
            evicted_k_raw = k_full_raw[:, :n_evict]
            evicted_v_raw = v_full_raw[:, :n_evict]
        else:
            evicted_k_raw = evicted_v_raw = None
        new_cache_k = k_full_raw[:, -window:]
        new_cache_v = v_full_raw[:, -window:]
        return out, new_cache_k, new_cache_v, evicted_k_raw, evicted_v_raw

    def _transformer_block(
        vocab: int, d_model: int, n_layer: int, n_head: int
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        """The pre-norm residual scaffolding shared by every spine in this module -- embedding, per-layer
        QKV/output projections, LayerNorms, MLPs, final norm, tied output head. Identical to
        ``SlidingWindowSpine``'s construction so the only thing E3's mechanisms vary is far-field state."""
        tok = nn.Embedding(vocab, d_model)
        qkv = nn.ModuleList([nn.Linear(d_model, 3 * d_model) for _ in range(n_layer)])
        proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layer)])
        ln1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        ln2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layer)])
        mlp = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
                for _ in range(n_layer)
            ]
        )
        ln_f = nn.LayerNorm(d_model)
        head = nn.Linear(d_model, vocab, bias=False)
        head.weight = tok.weight
        return tok, qkv, proj, ln1, ln2, mlp, ln_f, head


# ---------------------------------------------------------------------------------------------------------
# (a) Linear-attention prefix state -- exact, unbounded-rank, chunked scan.
# ---------------------------------------------------------------------------------------------------------


def _validated_state(state: Any, groups: tuple[tuple[str, list], ...], *, name: str) -> None:
    """Establish the invariants a carried attention state must satisfy before ``step`` trusts it.

    Three separate holes, all reachable through the public dataclasses (MXR-080-1879):

    * ``pos`` was an unchecked int. A negative position makes ``far_count_before = pos - cache_len``
      negative, and a fractional one reaches tensor indexing as a float.
    * the per-layer lists were independent, so a state could carry three numerator layers and two
      normalizer layers; ``step`` then indexed past the shorter one, or worse, silently paired layer
      i's numerator with a different layer's normalizer.
    * a non-finite accumulator poisons every subsequent readout while looking like ordinary state,
      and the readout's own finiteness check runs after the arithmetic rather than before it.

    Finiteness is checked only on entry, and that limit is deliberate and stated rather than implied:
    these lists hold live tensors the caller can mutate afterwards, so this establishes the invariant
    at construction, not for all time.
    """
    if isinstance(state.pos, bool) or not isinstance(state.pos, int):
        raise TypeError(f"{name} pos must be an exact integer, got {state.pos!r}")
    if state.pos < 0:
        raise ValueError(f"{name} pos must be non-negative, got {state.pos}")
    lengths = {label: len(values) for label, values in groups}
    distinct = {n for n in lengths.values() if n}
    if len(distinct) > 1:
        raise ValueError(
            f"{name} carries per-layer lists of differing lengths {lengths}; layer i's tensors would "
            "be paired with another layer's, or indexed past the end of the shorter list."
        )
    if not _HAS_TORCH:
        return
    for label, values in groups:
        for index, tensor in enumerate(values):
            if tensor is None or not hasattr(tensor, "shape"):
                continue
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(
                    f"{name} {label}[{index}] holds a non-finite value; every readout built from it "
                    "would be non-finite while the state still looks ordinary."
                )


@dataclass
class LinearAttentionState:
    S: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, head_dim, head_dim), sum phi(k) v^T
    Z: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, head_dim), sum phi(k)
    pos: int = 0

    def __post_init__(self) -> None:
        _validated_state(self, (("S", self.S), ("Z", self.Z)), name="LinearAttentionState")


if _HAS_TORCH:

    class LinearAttentionSpine(nn.Module):
        """(a) Exact linear-attention prefix state (Katharopoulos et al. 2020), chunked-scan trained.

        No local window: the whole stream is the linear-attention kernel (see module docstring for why --
        this mechanism's own Algorithm section in ``notes/designs/E3.md`` has no local softmax term at all,
        unlike (b)/(c)). RoPE is applied to the raw ``q``/``k`` projections before the ``phi = elu + 1``
        feature map, so positional information survives into the kernel while ``S``/``Z`` stay simple running
        sums -- carrying them across chunk boundaries reproduces the exact same cumulative sum a single
        non-chunked pass over the whole prefix would compute (the "chunked scan" streaming-equivalence
        invariant the test suite checks directly).
        """

        def __init__(self, vocab: int, *, d_model: int = 32, n_layer: int = 2, n_head: int = 2) -> None:
            super().__init__()
            # A public constructor argument check, so not an assert: `python -O` strips asserts, and
            # this one gates a real architectural invariant -- with d_model not divisible by n_head
            # the head split is lossy and the model silently builds wrong (MXR-080-1861).
            # Every constructor argument, not only the two an assert used to cover: a fractional
            # n_head passes a `d_model % n_head` test whenever the remainder happens to be zero and
            # then makes head_dim a float, and zero layers builds a model with nothing in it
            # (MXR-080-1863).
            vocab = _positive_integer(vocab, "LinearAttentionSpine vocab")
            d_model = _positive_integer(d_model, "LinearAttentionSpine d_model")
            n_layer = _positive_integer(n_layer, "LinearAttentionSpine n_layer")
            n_head = _positive_integer(n_head, "LinearAttentionSpine n_head")
            if d_model % n_head != 0:
                raise ValueError(
                    f"LinearAttentionSpine requires d_model divisible by a positive n_head, got "
                    f"d_model={d_model}, n_head={n_head}."
                )
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            (self.tok, self.qkv, self.proj, self.ln1, self.ln2, self.mlp, self.ln_f, self.head) = _transformer_block(
                vocab, d_model, n_layer, n_head
            )

        def init_state(self, batch_size: int, *, device: str = "cpu") -> LinearAttentionState:
            S = [
                torch.zeros(batch_size, self.n_head, self.head_dim, self.head_dim, device=device)
                for _ in range(self.n_layer)
            ]
            Z = [torch.zeros(batch_size, self.n_head, self.head_dim, device=device) for _ in range(self.n_layer)]
            return LinearAttentionState(S=S, Z=Z, pos=0)

        def detach(self, state: LinearAttentionState) -> LinearAttentionState:
            return LinearAttentionState(S=[s.detach() for s in state.S], Z=[z.detach() for z in state.Z], pos=state.pos)

        def step(self, state: LinearAttentionState, chunk: tuple[Any, Any]) -> tuple[LinearAttentionState, Any]:
            x, y = chunk
            b, t = x.shape
            device = x.device
            positions = torch.arange(state.pos, state.pos + t, device=device)
            sin, cos = _rope_angles(positions, self.head_dim)

            h = self.tok(x)
            new_S: list[Any] = []
            new_Z: list[Any] = []
            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q_raw, k_raw, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                q = _apply_rope(q_raw, sin, cos)
                k = _apply_rope(k_raw, sin, cos)
                phi_q = _phi(q)
                phi_k = _phi(k)

                outer = torch.einsum("bthd,bthe->bthde", phi_k, v)  # (b, t, n_head, head_dim, head_dim)
                cum_outer = torch.cumsum(outer, dim=1) + state.S[layer][:, None]
                cum_z = torch.cumsum(phi_k, dim=1) + state.Z[layer][:, None]  # (b, t, n_head, head_dim)

                num = torch.einsum("bthd,bthde->bthe", phi_q, cum_outer)
                den = torch.einsum("bthd,bthd->bth", phi_q, cum_z).clamp(min=1e-6)
                out = num / den.unsqueeze(-1)  # (b, t, n_head, head_dim)

                h = h + self.proj[layer](out.reshape(b, t, self.d_model))
                h = h + self.mlp[layer](self.ln2[layer](h))

                new_S.append(cum_outer[:, -1])
                new_Z.append(cum_z[:, -1])

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))
            return LinearAttentionState(S=new_S, Z=new_Z, pos=state.pos + t), loss


# ---------------------------------------------------------------------------------------------------------
# (b) Frequent Directions sketch of the KV outer-product stream.
# ---------------------------------------------------------------------------------------------------------


@dataclass
class FrequentDirectionsState:
    B: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, ell, d_phi + d_v)
    Z: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, d_phi) -- exact normalizer
    cache_k: list[Any] = field(default_factory=list)  # per layer: (batch, cache_len<=window, n_head, head_dim) | None
    cache_v: list[Any] = field(default_factory=list)
    pos: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validated_state(
            self,
            (("B", self.B), ("Z", self.Z), ("cache_k", self.cache_k), ("cache_v", self.cache_v)),
            name="FrequentDirectionsState",
        )


if _HAS_TORCH:

    def _frequent_directions_far_scan(
        q_raw: Any,
        B: Any,
        Z: Any,
        evicted_k: Any,
        evicted_v: Any,
        *,
        cache_len: int,
        window: int,
        ell: int,
        head_dim: int,
        far_count_before: int,
    ) -> tuple[Any, Any, Any, float | None]:
        """Advance FD state at each query's exact window crossing and return query-aligned far outputs."""
        t = q_raw.shape[1]
        n_evict = 0 if evicted_k is None else evicted_k.shape[1]
        processed = 0
        outputs: list[Any] = []
        min_denominator: float | None = None
        phi_q = _phi(q_raw)
        for query_index in range(t):
            target = max(0, cache_len + query_index + 1 - window)
            if target > n_evict:
                raise RuntimeError("local-window eviction schedule is inconsistent with FD far-state scan")
            while processed < target:
                phi_key = _phi(evicted_k[:, processed])
                row = torch.cat([phi_key, evicted_v[:, processed]], dim=-1)
                B = _fd_insert_row(B, row, ell)
                Z = Z + phi_key
                processed += 1
            if far_count_before + processed == 0:
                outputs.append(q_raw.new_zeros(q_raw.shape[0], q_raw.shape[2], head_dim))
                continue
            B_K = B[..., :head_dim]
            B_V = B[..., head_dim:]
            S_approx = torch.einsum("bnld,bnle->bnde", B_K, B_V)
            query = phi_q[:, query_index]
            numerator = torch.einsum("bhd,bhde->bhe", query, S_approx)
            denominator = torch.einsum("bhd,bhd->bh", query, Z)
            if not bool(torch.isfinite(numerator).all().item()) or not bool(torch.isfinite(denominator).all().item()):
                raise ValueError("Frequent-Directions far readout produced non-finite values")
            if bool((denominator <= 0).any().item()):
                raise ValueError(
                    "Frequent-Directions far normalizer must be strictly positive when far state is non-empty"
                )
            current_min = float(denominator.min().item())
            min_denominator = current_min if min_denominator is None else min(min_denominator, current_min)
            outputs.append(numerator / denominator.unsqueeze(-1))
        if processed != n_evict:
            raise RuntimeError("FD far-state scan did not consume every token that left the local window")
        return B, Z, torch.stack(outputs, dim=1), min_denominator

    class FrequentDirectionsSpine(nn.Module):
        """(b) FD sketch of the KV outer-product stream, exact per Liberty (2013).

        Local half: a small ``SlidingWindowSpine``-shaped exact-softmax window. Far-field half: once a token
        scrolls out of the window, its augmented row ``[phi(k_t) ; v_t]`` is streamed into a Frequent
        Directions sketch ``B`` (literal ``ell x (d_phi + d_v)`` shape with genuine zero rows -- see
        ``frequent_directions_update``/``_fd_insert_row``), and the normalizer ``Z = sum phi(k_t)`` is
        tracked exactly alongside it. A query reads the far field back as
        ``phi(q)^T (B_K^T B_V) / (phi(q)^T Z)``, ``B_K``/``B_V`` being ``B``'s two column blocks split at
        ``d_phi`` -- an FD-bounded approximation of the exact cross term (a) tracks exactly.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int = 64,
            ell: int = 16,
        ) -> None:
            super().__init__()
            self.vocab = _positive_integer(vocab, "vocab")
            self.d_model = _positive_integer(d_model, "d_model")
            self.n_layer = _positive_integer(n_layer, "n_layer")
            self.n_head = _positive_integer(n_head, "n_head")
            self.window = _positive_integer(window, "window")
            self.ell = _positive_integer(ell, "ell")
            if self.d_model % self.n_head != 0:
                raise ValueError(f"d_model={self.d_model} must be divisible by n_head={self.n_head}")
            self.head_dim = d_model // n_head
            self.d_row = 2 * self.head_dim
            if self.ell > self.d_row:
                raise ValueError(f"ell={self.ell} must not exceed augmented row dimension {self.d_row}")
            (self.tok, self.qkv, self.proj, self.ln1, self.ln2, self.mlp, self.ln_f, self.head) = _transformer_block(
                self.vocab, self.d_model, self.n_layer, self.n_head
            )

        def init_state(self, batch_size: int, *, device: str = "cpu") -> FrequentDirectionsState:
            batch_size = _positive_integer(batch_size, "batch_size")
            dtype = self.tok.weight.dtype
            B = [
                torch.zeros(batch_size, self.n_head, self.ell, self.d_row, device=device, dtype=dtype)
                for _ in range(self.n_layer)
            ]
            Z = [
                torch.zeros(batch_size, self.n_head, self.head_dim, device=device, dtype=dtype)
                for _ in range(self.n_layer)
            ]
            return FrequentDirectionsState(
                B=B, Z=Z, cache_k=[None] * self.n_layer, cache_v=[None] * self.n_layer, pos=0
            )

        def detach(self, state: FrequentDirectionsState) -> FrequentDirectionsState:
            return FrequentDirectionsState(
                B=[b.detach() for b in state.B],
                Z=[z.detach() for z in state.Z],
                cache_k=[k.detach() if k is not None else None for k in state.cache_k],
                cache_v=[v.detach() if v is not None else None for v in state.cache_v],
                pos=state.pos,
                receipt=dict(state.receipt),
            )

        def step(self, state: FrequentDirectionsState, chunk: tuple[Any, Any]) -> tuple[FrequentDirectionsState, Any]:
            x, y = chunk
            b, t = x.shape

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_B: list[Any] = []
            new_Z: list[Any] = []
            denominator_minima: list[float | None] = []
            far_counts: list[int] = []
            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q_raw, k_raw, v_raw = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

                local_out, cache_k, cache_v, evicted_k, evicted_v = _local_window_step(
                    q_raw,
                    k_raw,
                    v_raw,
                    state.cache_k[layer],
                    state.cache_v[layer],
                    window=self.window,
                    head_dim=self.head_dim,
                    pos=state.pos,
                )

                cache_len = 0 if state.cache_k[layer] is None else state.cache_k[layer].shape[1]
                far_count_before = state.pos - cache_len
                if far_count_before < 0:
                    raise ValueError("state position cannot be smaller than its cache length")
                B_layer, Z_layer, far_out, denominator_minimum = _frequent_directions_far_scan(
                    q_raw,
                    state.B[layer],
                    state.Z[layer],
                    evicted_k,
                    evicted_v,
                    cache_len=cache_len,
                    window=self.window,
                    ell=self.ell,
                    head_dim=self.head_dim,
                    far_count_before=far_count_before,
                )
                denominator_minima.append(denominator_minimum)
                far_counts.append(far_count_before + (0 if evicted_k is None else evicted_k.shape[1]))

                out = local_out + far_out
                h = h + self.proj[layer](out.reshape(b, t, self.d_model))
                h = h + self.mlp[layer](self.ln2[layer](h))

                new_B.append(B_layer)
                new_Z.append(Z_layer)
                new_cache_k.append(cache_k)
                new_cache_v.append(cache_v)

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))
            new_state = FrequentDirectionsState(
                B=new_B,
                Z=new_Z,
                cache_k=new_cache_k,
                cache_v=new_cache_v,
                pos=state.pos + t,
                receipt={
                    "far_tokens_per_layer": far_counts,
                    "minimum_positive_denominator_per_layer": denominator_minima,
                    "chunk_boundary_invariant_update": True,
                },
            )
            return new_state, loss


# ---------------------------------------------------------------------------------------------------------
# (c) Tensor sketch -- higher-order (degree-p) key features.
# ---------------------------------------------------------------------------------------------------------


@dataclass
class TensorSketchState:
    C: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, sketch_dim, d_v)
    # per layer: (batch, n_head, d_phi) -- the normalizer sum phi(k_t) tracked EXACTLY, as (b) does.
    # Sketching it too would be both unnecessary (d_phi == head_dim, O(d_phi) per step) and unsound:
    # TensorSketch applies random signs, so a sketched normalizer is only an unbiased estimate and can
    # come out zero or negative, which is what _phi's non-negativity is supposed to rule out.
    Z: list[Any] = field(default_factory=list)
    # per layer: (batch, n_head, sketch_dim) -- sum TS(phi(k_t)), the DEGREE-MATCHED normalizer used
    # when degree > 1. The exact Z above is degree one: it estimates sum <phi(q), phi(k_t)>, while a
    # degree-p numerator estimates sum <phi(q), phi(k_t)>^p. Dividing one by the other left a stray
    # <phi(q), phi(k)>^(p-1) factor in the readout, so a single-key degree-2 scan returned 5 where a
    # normalized readout must return 1 (MXR-080-1853). Only the sketch can supply the degree-p sum,
    # so degree > 1 uses this and degree 1 keeps the exact Z, which already matches.
    Z_ts: list[Any] = field(default_factory=list)
    cache_k: list[Any] = field(default_factory=list)
    cache_v: list[Any] = field(default_factory=list)
    pos: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a state carrying numerator history without its degree-matched denominator.

        ``Z_ts`` was added to an existing dataclass with a ``list`` default, so a state built before
        it -- or by any caller that does not know about it -- arrived with ``C`` and ``Z`` populated
        and ``Z_ts == []``. ``step`` read that as "start the denominator from zero" and kept
        accumulating the numerator, so the readout divided a full history by a partial one: a
        deterministic two-key probe returned ``4.0`` where a normalized readout must return ``2.0``
        (MXR-080-1870). The repair for MXR-080-1853 fixed the arithmetic and left the state's own
        shape unchecked, which is a quieter version of the same wrong answer.

        There is no migration to offer above degree one. ``Z_ts = sum_t TS(phi(k_t))`` and TensorSketch
        is a polynomial feature map, not a linear one, so the degree-p sum cannot be recovered from
        the exact degree-one ``Z`` or from ``C`` without the values ``v_t`` that were folded into it.
        Refusing is the only answer that is not a guess.
        """
        _validated_state(
            self,
            (("C", self.C), ("Z", self.Z), ("Z_ts", self.Z_ts), ("cache_k", self.cache_k), ("cache_v", self.cache_v)),
            name="TensorSketchState",
        )
        if not self.C:
            return
        if len(self.Z) != len(self.C) or len(self.Z_ts) != len(self.C):
            raise ValueError(
                f"TensorSketchState carries {len(self.C)} numerator layer(s) but {len(self.Z)} exact "
                f"normalizer(s) and {len(self.Z_ts)} degree-matched normalizer(s). A state written "
                "before the degree-matched normalizer existed (MXR-080-1853) cannot be migrated: "
                "sum_t TS(phi(k_t)) is not recoverable from C or from the exact Z, because "
                "TensorSketch is not linear above degree one. Re-run the scan from init_state."
            )


if _HAS_TORCH:

    def _tensor_sketch_far_scan(
        q_raw: Any,
        C: Any,
        Z: Any,
        Z_ts: Any,
        evicted_k: Any,
        evicted_v: Any,
        hashes: list[Any],
        signs: list[Any],
        *,
        cache_len: int,
        window: int,
        sketch_dim: int,
        head_dim: int,
        far_count_before: int,
    ) -> tuple[Any, Any, Any, Any, float | None]:
        """Advance TensorSketch numerator and denominator at every query's exact window crossing."""
        t = q_raw.shape[1]
        n_evict = 0 if evicted_k is None else evicted_k.shape[1]
        processed = 0
        outputs: list[Any] = []
        min_denominator: float | None = None
        phi_q = _phi(q_raw)
        ts_q = tensor_sketch_project(phi_q, hashes, signs, sketch_dim)
        degree = len(hashes)
        if degree > 1 and Z_ts is None and far_count_before > 0:
            # ``C`` already holds ``far_count_before`` tokens, so starting the degree-matched
            # denominator from this chunk's first key divides a full numerator by a partial
            # normalizer -- the readout is then wrong by the ratio of the two histories rather than
            # simply unnormalized (MXR-080-1870).
            raise ValueError(
                "TensorSketch far state carries %d evicted token(s) of numerator history but no "
                "degree-matched normalizer; a degree-%d readout cannot be normalized from it. See "
                "TensorSketchState.Z_ts." % (far_count_before, degree)
            )
        for query_index in range(t):
            target = max(0, cache_len + query_index + 1 - window)
            if target > n_evict:
                raise RuntimeError("local-window eviction schedule is inconsistent with TensorSketch far-state scan")
            while processed < target:
                phi_key = _phi(evicted_k[:, processed])
                ts_key = tensor_sketch_project(phi_key, hashes, signs, sketch_dim)
                C = C + torch.einsum("bhm,bhe->bhme", ts_key, evicted_v[:, processed])
                Z = Z + phi_key  # exact, degree one: correct normalizer when degree == 1
                Z_ts = ts_key if Z_ts is None else Z_ts + ts_key  # degree-p normalizer, see the state
                processed += 1
            if far_count_before + processed == 0:
                outputs.append(q_raw.new_zeros(q_raw.shape[0], q_raw.shape[2], head_dim))
                continue
            query = ts_q[:, query_index]
            numerator = torch.einsum("bhm,bhme->bhe", query, C)
            # phi is elu + 1, so both factors are strictly positive and this is exact: the sum cannot be
            # zero or negative once any token has left the window. Only the NUMERATOR is sketched, which
            # is the standard exact-normalizer form and is what (b) already does.
            if degree == 1:
                # Degree one is Count Sketch: the numerator estimates sum <phi(q), phi(k_t)>, which the
                # exact Z matches. phi is elu + 1 so both factors are strictly positive and a
                # non-positive value here really does mean corrupted state.
                denominator = torch.einsum("bhd,bhd->bh", phi_q[:, query_index], Z)
            else:
                denominator = torch.einsum("bhm,bhm->bh", query, Z_ts)
            if not bool(torch.isfinite(numerator).all().item()) or not bool(torch.isfinite(denominator).all().item()):
                raise ValueError("TensorSketch far readout produced non-finite values")
            if bool((denominator <= 0).any().item()):
                raise ValueError(
                    "TensorSketch kernel normalizer must be strictly positive when far state is "
                    "non-empty; got a non-positive value at degree %d. At degree one phi is elu + 1 so "
                    "this indicates corrupted state; above degree one the normalizer is a sketched "
                    "estimate whose random signs admit a non-positive draw, and the readout is "
                    "undefined for it -- raise sketch_dim or reseed the hashes." % degree
                )
            current_min = float(denominator.min().item())
            min_denominator = current_min if min_denominator is None else min(min_denominator, current_min)
            outputs.append(numerator / denominator.unsqueeze(-1))
        if processed != n_evict:
            raise RuntimeError("TensorSketch far-state scan did not consume every token that left the local window")
        return C, Z, Z_ts, torch.stack(outputs, dim=1), min_denominator

    class TensorSketchSpine(nn.Module):
        """(c) Tensor sketch (Count Sketch + circular convolution) of degree-``p`` key features (Pham &
        Pagh 2013). Local half identical in shape to (b); far-field half accumulates
        ``C_t = C_{t-1} + TS(phi(k_t)) v_t^T``, ``Z_t = Z_{t-1} + phi(k_t)``, and
        ``Zts_t = Zts_{t-1} + TS(phi(k_t))`` for evicted tokens.

        **The normalizer matches the numerator's degree.** ``TS(phi(q))^T TS(phi(k))`` estimates
        ``<phi(q), phi(k)>**p``, so a degree-``p`` numerator needs ``sum_t <phi(q), phi(k_t)>**p``
        beneath it. An earlier revision divided it by the exact ``phi(q)^T Z``, which is degree ONE:
        the ratio kept a stray ``<phi(q), phi(k)>**(p-1)`` factor, and a single key with value 1 at
        degree 2 read out ``5`` where a normalized readout must be ``1`` (MXR-080-1853). The readout
        is therefore ``TS(phi(q))^T C / phi(q)^T Z`` at degree 1 and
        ``TS(phi(q))^T C / <TS(phi(q)), Zts>`` above it.

        Degree 1 deliberately keeps the exact normalizer. There it is already the right degree, and it
        avoids the sketched estimator's one real hazard: TensorSketch applies random signs, so a
        sketched denominator is an unbiased estimate that can come out zero or negative however
        non-negative ``phi`` is, and the probability of that is not monotone in ``sketch_dim``
        (measured on the E7 bake-off: ``sketch_dim`` 16 and 32 succeeded where 12 and 24 failed).
        Above degree 1 that hazard is unavoidable -- only the sketch can express the degree-``p``
        sum -- so a non-positive draw raises and names itself as sketch variance rather than
        corrupted state. Being occasionally undefined is the honest cost of computing the right
        quantity; the alternative was computing the wrong one deterministically.
        ``phi(q)^T Z > 0`` a theorem again rather than a hope.

        This is still a normalized estimator of polynomial-kernel attention, not a probability
        distribution: the sketched NUMERATOR's per-token contributions can be signed, so non-negative
        per-token weights are not claimed.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int = 64,
            sketch_dim: int = 64,
            degree: int = 2,
            seed: int = 0,
        ) -> None:
            super().__init__()
            self.vocab = _positive_integer(vocab, "vocab")
            self.d_model = _positive_integer(d_model, "d_model")
            self.n_layer = _positive_integer(n_layer, "n_layer")
            self.n_head = _positive_integer(n_head, "n_head")
            self.window = _positive_integer(window, "window")
            self.sketch_dim = _positive_integer(sketch_dim, "sketch_dim")
            self.degree = _positive_integer(degree, "degree")
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise ValueError(f"seed must be an exact integer, got {seed}")
            if self.d_model % self.n_head != 0:
                raise ValueError(f"d_model={self.d_model} must be divisible by n_head={self.n_head}")
            self.head_dim = d_model // n_head
            (self.tok, self.qkv, self.proj, self.ln1, self.ln2, self.mlp, self.ln_f, self.head) = _transformer_block(
                self.vocab, self.d_model, self.n_layer, self.n_head
            )
            self._hash_names: list[list[str]] = []
            self._sign_names: list[list[str]] = []
            for layer in range(self.n_layer):
                hashes, signs = make_tensor_sketch_hashes(
                    self.head_dim,
                    sketch_dim=self.sketch_dim,
                    degree=self.degree,
                    seed=int(seed) + layer,
                )
                layer_hash_names: list[str] = []
                layer_sign_names: list[str] = []
                for degree_index, (hash_values, sign_values) in enumerate(zip(hashes, signs)):
                    hash_name = f"_tensor_hash_{layer}_{degree_index}"
                    sign_name = f"_tensor_sign_{layer}_{degree_index}"
                    self.register_buffer(hash_name, hash_values)
                    self.register_buffer(sign_name, sign_values)
                    layer_hash_names.append(hash_name)
                    layer_sign_names.append(sign_name)
                self._hash_names.append(layer_hash_names)
                self._sign_names.append(layer_sign_names)

        @property
        def _hashes(self) -> list[list[Any]]:
            """Registered hash maps, grouped by layer (compatibility view)."""
            return [[getattr(self, name) for name in names] for names in self._hash_names]

        @property
        def _signs(self) -> list[list[Any]]:
            """Registered sign maps, grouped by layer (compatibility view)."""
            return [[getattr(self, name) for name in names] for names in self._sign_names]

        def init_state(self, batch_size: int, *, device: str = "cpu") -> TensorSketchState:
            batch_size = _positive_integer(batch_size, "batch_size")
            dtype = self.tok.weight.dtype
            C = [
                torch.zeros(
                    batch_size,
                    self.n_head,
                    self.sketch_dim,
                    self.head_dim,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.n_layer)
            ]
            Z = [
                torch.zeros(batch_size, self.n_head, self.head_dim, device=device, dtype=dtype)
                for _ in range(self.n_layer)
            ]
            return TensorSketchState(
                C=C,
                Z=Z,
                Z_ts=[
                    torch.zeros(batch_size, self.n_head, self.sketch_dim, device=device, dtype=dtype)
                    for _ in range(self.n_layer)
                ],
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                pos=0,
            )

        def detach(self, state: TensorSketchState) -> TensorSketchState:
            return TensorSketchState(
                C=[c.detach() for c in state.C],
                Z=[z.detach() for z in state.Z],
                Z_ts=[z.detach() if z is not None else None for z in state.Z_ts],
                cache_k=[k.detach() if k is not None else None for k in state.cache_k],
                cache_v=[v.detach() if v is not None else None for v in state.cache_v],
                pos=state.pos,
                receipt=dict(state.receipt),
            )

        def step(self, state: TensorSketchState, chunk: tuple[Any, Any]) -> tuple[TensorSketchState, Any]:
            x, y = chunk
            b, t = x.shape

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_C: list[Any] = []
            new_Z: list[Any] = []
            new_Z_ts: list[Any] = []
            denominator_minima: list[float | None] = []
            far_counts: list[int] = []
            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q_raw, k_raw, v_raw = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

                local_out, cache_k, cache_v, evicted_k, evicted_v = _local_window_step(
                    q_raw,
                    k_raw,
                    v_raw,
                    state.cache_k[layer],
                    state.cache_v[layer],
                    window=self.window,
                    head_dim=self.head_dim,
                    pos=state.pos,
                )

                hashes, signs = self._hashes[layer], self._signs[layer]
                cache_len = 0 if state.cache_k[layer] is None else state.cache_k[layer].shape[1]
                far_count_before = state.pos - cache_len
                if far_count_before < 0:
                    raise ValueError("state position cannot be smaller than its cache length")
                C_layer, Z_layer, Z_ts_layer, far_out, denominator_minimum = _tensor_sketch_far_scan(
                    q_raw,
                    state.C[layer],
                    state.Z[layer],
                    # Indexed unconditionally: ``TensorSketchState`` refuses a state whose Z_ts does
                    # not cover its C, so `if state.Z_ts else None` could only ever have silently
                    # restarted the denominator against an already-accumulated numerator
                    # (MXR-080-1870).
                    state.Z_ts[layer],
                    evicted_k,
                    evicted_v,
                    hashes,
                    signs,
                    cache_len=cache_len,
                    window=self.window,
                    sketch_dim=self.sketch_dim,
                    head_dim=self.head_dim,
                    far_count_before=far_count_before,
                )
                denominator_minima.append(denominator_minimum)
                far_counts.append(far_count_before + (0 if evicted_k is None else evicted_k.shape[1]))

                out = local_out + far_out
                h = h + self.proj[layer](out.reshape(b, t, self.d_model))
                h = h + self.mlp[layer](self.ln2[layer](h))

                new_C.append(C_layer)
                new_Z.append(Z_layer)
                new_Z_ts.append(Z_ts_layer)
                new_cache_k.append(cache_k)
                new_cache_v.append(cache_v)

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))
            new_state = TensorSketchState(
                C=new_C,
                Z=new_Z,
                Z_ts=new_Z_ts,
                cache_k=new_cache_k,
                cache_v=new_cache_v,
                pos=state.pos + t,
                receipt={
                    "far_tokens_per_layer": far_counts,
                    "minimum_positive_denominator_per_layer": denominator_minima,
                    "chunk_boundary_invariant_update": True,
                    # True at every degree now that the normalizer matches the numerator's degree
                    # (MXR-080-1853). The numerator estimates sum_t <phi(q), phi(k_t)>**degree; the
                    # denominator used to be phi(q)^T sum_t phi(k_t), a degree-ONE kernel, so the
                    # ratio kept a stray <phi(q), phi(k)>**(degree - 1) factor -- one key, value 1,
                    # degree 2, q = k = [2, 1] returned 5 where a normalized readout must return 1.
                    # Relabelling that in the receipt did not repair the computation, so the state now
                    # carries Z_ts = sum_t TS(phi(k_t)) and degree > 1 divides by <TS(phi(q)), Z_ts>,
                    # which estimates the degree-p sum directly. Degree 1 keeps the exact
                    # normalizer -- already the right degree, and free of the sketched estimator's
                    # sign risk. Verified: the single-key readout is exactly 1.0 at degrees 1, 2, 3.
                    "normalized_kernel_estimator": True,
                    "nonnegative_attention_weights_guaranteed": False,
                },
            )
            return new_state, loss


# ---------------------------------------------------------------------------------------------------------
# Misfit receipts (graduation.py bookkeeping -- see mixle/experimental/graduation.py's docstring, which
# already names "sketch collision rate" as the worked example this module fills in).
# ---------------------------------------------------------------------------------------------------------

if _HAS_TORCH:

    def fd_misfit_receipt(A: Any, ell: int, *, k: int = 0) -> dict[str, float]:
        """Stream ``A``'s rows through FD, then report the realized ``||A^T A - B^T B||_2`` against
        Liberty's Theorem 1.1 bound -- "how tight is the guarantee in practice", the (b) misfit receipt."""
        d = A.shape[1]
        B0 = torch.zeros(ell, d, dtype=A.dtype, device=A.device)
        B = frequent_directions_update(B0, A, ell)
        realized = float(torch.linalg.matrix_norm(A.T @ A - B.T @ B, ord=2))
        bound = frequent_directions_error_bound(A, B, ell, k)
        return {
            "realized_error": realized,
            "bound": bound,
            "tightness_ratio": realized / bound if bound > 0 else float("nan"),
        }

    def tensor_sketch_misfit_receipt(
        *, d: int, sketch_dim: int, degree: int, seed: int = 0, trials: int = 200
    ) -> dict[str, float]:
        """Empirical collision/variance rate of the tensor sketch inner-product estimator: sample
        ``TS(x)^T TS(y)`` over many fresh random ``(x, y)`` pairs (same hash/sign, per the "oblivious"
        contract) and report the empirical bias and variance against the true ``(x^T y)^p`` -- the (c)
        misfit receipt (`graduation.py`'s "sketch collision rate" worked example)."""
        hashes, signs = make_tensor_sketch_hashes(d, sketch_dim=sketch_dim, degree=degree, seed=seed)
        rng = torch.Generator().manual_seed(seed + 999)
        errors = []
        for _ in range(trials):
            x = torch.randn(d, generator=rng)
            y = torch.randn(d, generator=rng)
            true_val = float((x @ y) ** degree)
            ts_x = tensor_sketch_project(x, hashes, signs, sketch_dim)
            ts_y = tensor_sketch_project(y, hashes, signs, sketch_dim)
            est = float(ts_x @ ts_y)
            errors.append(est - true_val)
        errors_t = torch.tensor(errors)
        return {
            "mean_bias": float(errors_t.mean()),
            "empirical_variance": float(errors_t.var(unbiased=True)),
            "trials": float(trials),
        }
