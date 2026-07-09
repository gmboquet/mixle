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

__all__ = [
    "LinearAttentionState",
    "LinearAttentionSpine",
    "FrequentDirectionsState",
    "FrequentDirectionsSpine",
    "TensorSketchState",
    "TensorSketchSpine",
    "frequent_directions_update",
    "frequent_directions_error_bound",
    "tensor_sketch_project",
    "make_tensor_sketch_hashes",
    "fd_misfit_receipt",
    "tensor_sketch_misfit_receipt",
    "E3_UNAVAILABLE_COMPARISONS",
]

E3_UNAVAILABLE_COMPARISONS: dict[str, str] = {
    "E2": (
        "moment-closure attention (roadmap E2) has not been implemented anywhere reachable from this "
        "worktree as of 2026-07-09; moment-closure-attention exists only as an unpushed local "
        "branch/worktree sitting at this worktree's own base commit (b5928139, the E7 tip) -- i.e. zero "
        "E2 work has landed anywhere reachable. Per this project's convention for documenting unreachable "
        "dependencies honestly instead of blocking or fabricating a result (mixle.task.pilot_ladder's "
        "PILOT_LADDER_UNAVAILABLE_PIECES pattern), the E7 comparison table below is E1 vs the three E3 "
        "sketches only, with this string standing in for the E2 column rather than a fabricated row."
    ),
}


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("mixle.experimental.sketch_state_attention requires torch.")


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
        del ell  # ell is implied by B.shape[0]; kept in the signature to match the design note's API.
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
        if B is not None and int(B.shape[0]) != int(ell):
            raise ValueError(f"B has {B.shape[0]} rows, expected ell={ell}.")
        if k <= 0:
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


@dataclass
class LinearAttentionState:
    S: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, head_dim, head_dim), sum phi(k) v^T
    Z: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, head_dim), sum phi(k)
    pos: int = 0


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
            assert d_model % n_head == 0
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


if _HAS_TORCH:

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
            assert d_model % n_head == 0
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = int(window)
            self.ell = int(ell)
            self.d_row = 2 * self.head_dim
            (self.tok, self.qkv, self.proj, self.ln1, self.ln2, self.mlp, self.ln_f, self.head) = _transformer_block(
                vocab, d_model, n_layer, n_head
            )

        def init_state(self, batch_size: int, *, device: str = "cpu") -> FrequentDirectionsState:
            B = [torch.zeros(batch_size, self.n_head, self.ell, self.d_row, device=device) for _ in range(self.n_layer)]
            Z = [torch.zeros(batch_size, self.n_head, self.head_dim, device=device) for _ in range(self.n_layer)]
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
            )

        def step(self, state: FrequentDirectionsState, chunk: tuple[Any, Any]) -> tuple[FrequentDirectionsState, Any]:
            x, y = chunk
            b, t = x.shape

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_B: list[Any] = []
            new_Z: list[Any] = []
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

                B_layer, Z_layer = state.B[layer], state.Z[layer]
                B_K = B_layer[..., : self.head_dim]  # (b, n_head, ell, head_dim)
                B_V = B_layer[..., self.head_dim :]
                S_approx = torch.einsum("bnld,bnle->bnde", B_K, B_V)  # (b, n_head, head_dim, head_dim)

                phi_q = _phi(q_raw)
                far_num = torch.einsum("bthd,bnde->bthe", phi_q, S_approx)
                far_den = torch.einsum("bthd,bnd->bth", phi_q, Z_layer).clamp(min=1e-6)
                far_out = far_num / far_den.unsqueeze(-1)

                out = local_out + far_out
                h = h + self.proj[layer](out.reshape(b, t, self.d_model))
                h = h + self.mlp[layer](self.ln2[layer](h))

                if evicted_k is not None:
                    phi_evicted = _phi(evicted_k)  # (b, n_evict, n_head, head_dim)
                    rows = torch.cat([phi_evicted, evicted_v], dim=-1)  # (b, n_evict, n_head, d_row)
                    rows = rows.permute(0, 2, 1, 3)  # (b, n_head, n_evict, d_row)
                    B_layer = B_layer.clone()
                    for i in range(rows.shape[2]):
                        B_layer = _fd_insert_row(B_layer, rows[:, :, i], self.ell)
                    Z_layer = Z_layer + phi_evicted.sum(dim=1)  # sum over n_evict -> (b, n_head, head_dim)

                new_B.append(B_layer)
                new_Z.append(Z_layer)
                new_cache_k.append(cache_k)
                new_cache_v.append(cache_v)

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))
            new_state = FrequentDirectionsState(
                B=new_B, Z=new_Z, cache_k=new_cache_k, cache_v=new_cache_v, pos=state.pos + t
            )
            return new_state, loss


# ---------------------------------------------------------------------------------------------------------
# (c) Tensor sketch -- higher-order (degree-p) key features.
# ---------------------------------------------------------------------------------------------------------


@dataclass
class TensorSketchState:
    C: list[Any] = field(default_factory=list)  # per layer: (batch, n_head, sketch_dim, d_v)
    cache_k: list[Any] = field(default_factory=list)
    cache_v: list[Any] = field(default_factory=list)
    pos: int = 0


if _HAS_TORCH:

    class TensorSketchSpine(nn.Module):
        """(c) Tensor sketch (Count Sketch + circular convolution) of degree-``p`` key features (Pham &
        Pagh 2013). Local half identical in shape to (b); far-field half accumulates
        ``C_t = C_{t-1} + TS(phi(k_t)) v_t^T`` for evicted tokens and reads it back as
        ``TS(phi(q))^T C`` (no normalizer -- the design note's Algorithm section for (c) doesn't specify
        one, unlike (a)/(b); this mirrors that exactly rather than inventing an extra division).
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
            assert d_model % n_head == 0
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = int(window)
            self.sketch_dim = int(sketch_dim)
            self.degree = int(degree)
            (self.tok, self.qkv, self.proj, self.ln1, self.ln2, self.mlp, self.ln_f, self.head) = _transformer_block(
                vocab, d_model, n_layer, n_head
            )
            self._hashes: list[list[Any]] = []
            self._signs: list[list[Any]] = []
            for layer in range(n_layer):
                hashes, signs = make_tensor_sketch_hashes(
                    self.head_dim, sketch_dim=self.sketch_dim, degree=self.degree, seed=seed + layer
                )
                self._hashes.append(hashes)
                self._signs.append(signs)

        def init_state(self, batch_size: int, *, device: str = "cpu") -> TensorSketchState:
            C = [
                torch.zeros(batch_size, self.n_head, self.sketch_dim, self.head_dim, device=device)
                for _ in range(self.n_layer)
            ]
            return TensorSketchState(C=C, cache_k=[None] * self.n_layer, cache_v=[None] * self.n_layer, pos=0)

        def detach(self, state: TensorSketchState) -> TensorSketchState:
            return TensorSketchState(
                C=[c.detach() for c in state.C],
                cache_k=[k.detach() if k is not None else None for k in state.cache_k],
                cache_v=[v.detach() if v is not None else None for v in state.cache_v],
                pos=state.pos,
            )

        def step(self, state: TensorSketchState, chunk: tuple[Any, Any]) -> tuple[TensorSketchState, Any]:
            x, y = chunk
            b, t = x.shape

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_C: list[Any] = []
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
                phi_q = _phi(q_raw)
                ts_q = tensor_sketch_project(phi_q, hashes, signs, self.sketch_dim)  # (b, t, n_head, sketch_dim)
                # C is (b, n_head, sketch_dim, head_dim):
                far_out = torch.einsum("bthm,bnme->bthe", ts_q, state.C[layer])

                out = local_out + far_out
                h = h + self.proj[layer](out.reshape(b, t, self.d_model))
                h = h + self.mlp[layer](self.ln2[layer](h))

                C_layer = state.C[layer]
                if evicted_k is not None:
                    phi_evicted = _phi(evicted_k)  # (b, n_evict, n_head, head_dim)
                    ts_k = tensor_sketch_project(phi_evicted, hashes, signs, self.sketch_dim)
                    contrib = torch.einsum("bnhm,bnhe->bnhme", ts_k, evicted_v)  # (b, n_evict, n_head, m, head_dim)
                    contrib = contrib.sum(dim=1)  # (b, n_head, sketch_dim, head_dim)
                    C_layer = C_layer + contrib

                new_C.append(C_layer)
                new_cache_k.append(cache_k)
                new_cache_v.append(cache_v)

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))
            new_state = TensorSketchState(C=new_C, cache_k=new_cache_k, cache_v=new_cache_v, pos=state.pos + t)
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
        B0 = torch.zeros(ell, d, dtype=A.dtype)
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
