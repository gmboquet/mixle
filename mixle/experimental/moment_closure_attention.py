"""E2: moment-closure (mixture-state) attention -- see ``notes/designs/E2.md`` (APPROVED) for the full
derivation this module implements section-by-section (that note's section numbers are cited throughout this
module's docstrings so the two stay easy to cross-reference).

**What this is.** E1's :class:`~mixle.experimental.context_spine.SlidingWindowSpine` keeps an exact but
bounded KV window; everything older than ``window`` tokens is gone. E2 additionally keeps a *far-field*
summary of everything outside that window as a streaming Gaussian mixture over ``(key, value)`` pairs (one
:class:`ClusterBank` per layer, covering all heads) and answers queries against it in closed form via the
MGF identity :func:`mixle.models.moment_propagation.attention_law` already proved for a single stationary
population (E2.md section 3.2 extends that identity to ``K`` clusters). Per query, per layer: near-field
exact attention (E1's window) and far-field mixture attention are combined by ONE joint softmax spanning
both (E2.md section 3.3) -- not two independently-normalized attentions blended by a gate.

**Cost (E2.md section 3.5, stated honestly, not claimed away).** The far-field forward is
``O(K * d_head)`` for the linear (mean) and diagonal-quadratic terms, but ``O(K * d_head^2)`` overall
because ``Sigma_vk`` is a full (not diagonal) ``d_head x d_head`` cross-covariance and its matvec against
``q`` is the dominant per-cluster cost. This is still independent of stream length -- the ``O(B)``-per-token
property the roadmap card wants (bounded state, cost independent of how much history has been summarized)
-- it just isn't literally ``O(K * d_head)`` as an early draft of the design note claimed before
self-correcting.

**Gradient path (E2.md section 3.4).** No custom backward function anywhere in this module. Responsibilities
``r_ik`` are themselves differentiable softmax outputs of the same MGF logits (evaluated with the token's
own key playing the role of a one-token query), and the running cluster statistics are literal weighted sums
/ divisions of ``r_ik`` and the token's own ``k``/``v`` (themselves outputs of the model's ``qkv``
projection) -- ordinary autograd carries gradients from the eventual loss back through both "how much
responsibility did token t get" and "what did the qkv projection produce for token t", the same way E1's KV
cache concat is differentiable with no detach except at ``mechanism.detach()``.

Former E2 roadmap dependencies are now present. :data:`E2_UNAVAILABLE_PIECES` remains as an empty
compatibility registry so callers can determine that no declared dependency is currently missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is optional
    _HAS_TORCH = False

from mixle.experimental.context_spine import SlidingWindowState

if _HAS_TORCH:
    from mixle.experimental.context_spine import _apply_rope, _rope_angles

__all__ = [
    "E2_UNAVAILABLE_PIECES",
    "ClusterBank",
    "MomentClosureState",
    "MomentClosureAttention",
    "mgf_cluster_attention",
    "cluster_responsibilities",
    "update_cluster_bank",
    "birth_and_merge",
    "ingest_cluster_batch",
]

#: Compatibility registry for unresolved E2 dependencies. E3 sketch-state attention and I2/G4 clustered
#: KV quantization are present in this release-preparation tree, so the registry is intentionally empty.
E2_UNAVAILABLE_PIECES: dict[str, str] = {}


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("mixle.experimental.moment_closure_attention requires torch.")


@dataclass
class ClusterBank:
    """Per-layer (all heads) sufficient statistics for the far-field Gaussian-mixture KV store.

    All fields are torch tensors so gradients flow through them (E2.md section 3.4); ``n_clusters`` is the
    live cluster count (``<= max_clusters``), shared across heads for simple slot bookkeeping (birth/merge
    runs once per chunk, at TBPTT granularity, per E2.md section 4 -- not once per head). The rest are
    pre-allocated to ``max_clusters``; inactive slots carry ``count == 0`` and all-zero statistics until
    :func:`birth_and_merge` seeds them.

    Shapes carry an explicit leading ``n_head`` axis (E2.md section 3.1: "per-(layer, head)"; a single bank
    with no head axis could not hold independent clusters per head's own K/V subspace, so the head axis is
    made explicit here even though the design note's illustrative shape comments omit it).
    """

    count: Any  # (n_head, max_clusters)                     soft token count (sum of responsibilities)
    mu_k: Any  # (n_head, max_clusters, d_head)                running mean key
    mu_v: Any  # (n_head, max_clusters, d_head)                running mean value
    sigma_kk: Any  # (n_head, max_clusters, d_head)            running DIAGONAL key covariance (v1 restriction)
    sigma_vk: Any  # (n_head, max_clusters, d_head, d_head)    running cross-covariance Cov(v, k) -- full, not diag
    n_clusters: int  # live prefix; birth/merge only ever touches this many slots
    max_clusters: int

    def detach(self) -> ClusterBank:
        return replace(
            self,
            count=self.count.detach(),
            mu_k=self.mu_k.detach(),
            mu_v=self.mu_v.detach(),
            sigma_kk=self.sigma_kk.detach(),
            sigma_vk=self.sigma_vk.detach(),
        )


def _empty_cluster_bank(n_head: int, max_clusters: int, d_head: int, *, device: Any, dtype: Any) -> ClusterBank:
    z = lambda *shape: torch.zeros(*shape, device=device, dtype=dtype)  # noqa: E731
    return ClusterBank(
        count=z(n_head, max_clusters),
        mu_k=z(n_head, max_clusters, d_head),
        mu_v=z(n_head, max_clusters, d_head),
        sigma_kk=z(n_head, max_clusters, d_head),
        sigma_vk=z(n_head, max_clusters, d_head, d_head),
        n_clusters=0,
        max_clusters=max_clusters,
    )


@dataclass
class MomentClosureState:
    """``ContextMechanism`` carried state: E1's exact near-field cache plus one far-field bank per layer."""

    near: SlidingWindowState  # E1's exact near-field state, reused verbatim (E2.md section 3.3)
    banks: list  # one ClusterBank per layer (per-head handled inside each ClusterBank's own head axis)
    batch_size: int = 1
    detach_horizon_clusters: bool = True  # whether ClusterBank stats are stop-gradiented at mechanism.detach()


def _bank_total_per_head(bank: ClusterBank) -> Any:
    if bank.n_clusters == 0:
        return bank.count.new_zeros(bank.count.shape[0])
    return bank.count[:, : bank.n_clusters].sum(dim=1)


def _representation_receipt(
    bank: ClusterBank,
    *,
    batch_size: int,
    positions_seen: int,
    near_tokens_per_stream: int,
    evicted_tokens_per_stream: int,
) -> dict[str, Any]:
    expected_far = batch_size * (positions_seen - near_tokens_per_stream)
    actual_far = _bank_total_per_head(bank)
    expected = actual_far.new_full(actual_far.shape, float(expected_far))
    conserved = bool(torch.allclose(actual_far, expected, atol=1e-4, rtol=1e-5))
    receipt = {
        "batch_size": batch_size,
        "positions_seen_per_stream": positions_seen,
        "near_tokens_per_stream": near_tokens_per_stream,
        "evicted_tokens_per_stream": evicted_tokens_per_stream,
        "far_tokens_per_head": [float(value) for value in actual_far.detach().cpu()],
        "expected_far_tokens_per_head": expected_far,
        "overlap_tokens": 0,
        "conserved": conserved,
    }
    if not conserved:
        raise RuntimeError(
            "moment-closure state violates one-token/one-representation accounting: "
            f"far={receipt['far_tokens_per_head']}, expected={expected_far}."
        )
    return receipt


# -------------------------------------------------------------------------------------------------------
# Pure-math core (E2.md section 3.2): the K-cluster MGF identity, unit-testable without a model.
# -------------------------------------------------------------------------------------------------------


def _mgf_core(q: Any, mu_k: Any, mu_v: Any, sigma_kk: Any, sigma_vk: Any, count: Any) -> tuple[Any, Any]:
    """Shared core of the MGF identity, evaluated against an explicit ``(mu_k, mu_v, sigma_kk, sigma_vk,
    count)`` tuple already sliced to the live clusters (no ``ClusterBank`` dependency, so it doubles as the
    engine for both :func:`mgf_cluster_attention` (query = a real query) and
    :func:`cluster_responsibilities` (query = the token's own key, per E2.md section 3.4).

    ``q``: ``(b, t, n_head, d_head)``. ``mu_k``/``mu_v``/``sigma_kk``: ``(n_head, n_clusters, d_head)``.
    ``sigma_vk``: ``(n_head, n_clusters, d_head, d_head)``. ``count``: ``(n_head, n_clusters)``.
    Returns ``(out, logits)`` with ``out``: ``(b, t, n_head, n_clusters, d_head)``, ``logits``:
    ``(b, t, n_head, n_clusters)`` -- the per-cluster affine map and its MGF log-partition (E2.md eq. in
    section 3.2: ``pi_k(q) = softmax_k[q^T mu_k/sqrt(d) + 0.5 q^T Sigma_kk q / d + log count_k]``).
    """
    d_head = q.shape[-1]
    scale = 1.0 / math.sqrt(d_head)

    linear = torch.einsum("bthd,hcd->bthc", q, mu_k) * scale
    quad = 0.5 * (scale**2) * torch.einsum("bthd,hcd->bthc", q * q, sigma_kk)
    log_count = torch.log(count.clamp_min(1e-8))[None, None, :, :]
    logits = linear + quad + log_count  # (b, t, h, c)

    lin_out = torch.einsum("hcij,bthj->bthci", sigma_vk, q) * scale  # (b, t, h, c, d)
    out = lin_out + mu_v[None, None, :, :, :]  # (b, t, h, c, d)
    return out, logits


def mgf_cluster_attention(q: Any, bank: ClusterBank) -> tuple[Any, Any]:
    """Pure function (E2.md section 2/3.2): ``(b, t, n_head, d_head)`` query, :class:`ClusterBank` ->
    ``(per-cluster affine output (b, t, n_clusters, n_head, d_head), per-cluster log-partition
    (b, t, n_clusters, n_head))``, restricted to the bank's live ``n_clusters`` (inactive slots are
    excluded entirely, not eps-suppressed, so a bank with exactly one live cluster reduces EXACTLY -- to
    float tolerance, not approximately -- to :func:`mixle.models.moment_propagation.attention_law`'s
    single-population formula; see ``mixle/tests/moment_closure_attention_test.py``).
    """
    _require_torch()
    n = bank.n_clusters
    b, t, h, d = q.shape
    if n == 0:
        return q.new_zeros(b, t, 0, h, d), q.new_zeros(b, t, 0, h)
    out, logits = _mgf_core(
        q,
        bank.mu_k[:, :n],
        bank.mu_v[:, :n],
        bank.sigma_kk[:, :n],
        bank.sigma_vk[:, :n],
        bank.count[:, :n],
    )
    return out.transpose(2, 3), logits.transpose(2, 3)  # (b, t, c, h, d) / (b, t, c, h)


def cluster_responsibilities(k: Any, bank: ClusterBank) -> Any:
    """Per-token soft cluster assignment ``r_ik`` (E2.md section 3.4): the token's own key plays the role
    of a one-token query into the same MGF logits :func:`mgf_cluster_attention` uses, softmaxed over the
    live clusters and zero-padded (exactly, not eps-suppressed) out to ``max_clusters`` so it can be fed
    straight into :func:`update_cluster_bank` without the caller tracking ``n_clusters`` separately.
    Returns ``(b, t, n_head, max_clusters)``.
    """
    _require_torch()
    n = bank.n_clusters
    b, t, h, d = k.shape
    if n == 0:
        return k.new_zeros(b, t, h, bank.max_clusters)
    _, logits = _mgf_core(
        k,
        bank.mu_k[:, :n],
        bank.mu_v[:, :n],
        bank.sigma_kk[:, :n],
        bank.sigma_vk[:, :n],
        bank.count[:, :n],
    )  # (b, t, h, n)
    r = F.softmax(logits, dim=-1)
    if n < bank.max_clusters:
        r = F.pad(r, (0, bank.max_clusters - n))
    return r


# -------------------------------------------------------------------------------------------------------
# Sufficient-statistic update (E2.md section 3.4): Welford/Chan-style parallel combination, fully
# differentiable (ordinary weighted sums and divisions -- no custom backward anywhere in this module).
# -------------------------------------------------------------------------------------------------------


def update_cluster_bank(bank: ClusterBank, k: Any, v: Any, responsibilities: Any) -> ClusterBank:
    """Soft, differentiable sufficient-statistic update (E2.md section 3.4).

    ``k``/``v``: ``(b, t, n_head, d_head)``; ``responsibilities``: ``(b, t, n_head, max_clusters)`` (as
    returned by :func:`cluster_responsibilities` -- exactly zero for inactive/unassigned slots). Uses
    Chan et al.'s parallel-variance-combination identity (the same "combine two mini-batches' running
    statistics" shape E2.md section 4's merge rule also reuses) to combine the bank's existing
    ``(count, mean, M2)`` with this chunk's batch statistics -- this naturally handles ``count == 0``
    (inactive slots, or ``n1 == 0`` slots nobody was responsible for this chunk) without a special case:
    when the existing count is zero the combination reduces to exactly the batch's own statistics; when the
    batch's responsibility-weighted count is zero, the bank is returned unchanged for that slot.
    """
    _require_torch()
    r = responsibilities
    n1 = r.sum(dim=(0, 1))  # (h, c)
    n1_safe = n1.clamp_min(1e-8)

    mean_k1 = torch.einsum("bthc,bthd->hcd", r, k) / n1_safe[..., None]
    mean_v1 = torch.einsum("bthc,bthd->hcd", r, v) / n1_safe[..., None]
    dk = k[:, :, :, None, :] - mean_k1[None, None]  # (b, t, h, c, d)
    dv = v[:, :, :, None, :] - mean_v1[None, None]
    m2_k1 = torch.einsum("bthc,bthcd->hcd", r, dk * dk)  # (h, c, d)
    c2_1 = torch.einsum("bthc,bthci,bthcj->hcij", r, dv, dk)  # (h, c, d_v, d_k)

    n0 = bank.count
    mean_k0, mean_v0 = bank.mu_k, bank.mu_v
    m2_k0 = bank.sigma_kk * n0[..., None]
    c2_0 = bank.sigma_vk * n0[..., None, None]

    n_new = n0 + n1
    n_new_safe = n_new.clamp_min(1e-8)
    delta_k = mean_k1 - mean_k0
    delta_v = mean_v1 - mean_v0
    frac = (n1 / n_new_safe)[..., None]
    mean_k_new = mean_k0 + delta_k * frac
    mean_v_new = mean_v0 + delta_v * frac

    cross_n = n0 * n1 / n_new_safe
    m2_k_new = m2_k0 + m2_k1 + delta_k * delta_k * cross_n[..., None]
    c2_new = c2_0 + c2_1 + delta_v[..., :, None] * delta_k[..., None, :] * cross_n[..., None, None]

    sigma_kk_new = m2_k_new / n_new_safe[..., None]
    sigma_vk_new = c2_new / n_new_safe[..., None, None]

    return replace(bank, count=n_new, mu_k=mean_k_new, mu_v=mean_v_new, sigma_kk=sigma_kk_new, sigma_vk=sigma_vk_new)


def _pooled_combine(
    count_a: Any,
    mu_k_a: Any,
    mu_v_a: Any,
    sigma_kk_a: Any,
    sigma_vk_a: Any,
    count_b: Any,
    mu_k_b: Any,
    mu_v_b: Any,
    sigma_kk_b: Any,
    sigma_vk_b: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Same Chan parallel-combination identity as :func:`update_cluster_bank`, applied to two EXISTING
    sufficient-statistic blocks (rather than a bank + a raw batch) -- the "pooled-variance identity" E2.md
    section 4's merge rule calls for, reused instead of re-derived."""
    n_new = count_a + count_b
    n_new_safe = n_new.clamp_min(1e-8)
    delta_k = mu_k_b - mu_k_a
    delta_v = mu_v_b - mu_v_a
    frac = (count_b / n_new_safe)[..., None]
    mu_k_new = mu_k_a + delta_k * frac
    mu_v_new = mu_v_a + delta_v * frac
    cross_n = count_a * count_b / n_new_safe
    m2_new = sigma_kk_a * count_a[..., None] + sigma_kk_b * count_b[..., None] + delta_k * delta_k * cross_n[..., None]
    c2_new = (
        sigma_vk_a * count_a[..., None, None]
        + sigma_vk_b * count_b[..., None, None]
        + delta_v[..., :, None] * delta_k[..., None, :] * cross_n[..., None, None]
    )
    return n_new, mu_k_new, mu_v_new, m2_new / n_new_safe[..., None], c2_new / n_new_safe[..., None, None]


# -------------------------------------------------------------------------------------------------------
# Birth / merge (E2.md section 4): DPM-style, evaluated once per chunk (not per token).
# -------------------------------------------------------------------------------------------------------


def _cluster_mean_distance(bank: ClusterBank, cluster_a: int, cluster_b: int) -> float:
    """RMS diagonal-Mahalanobis distance between two multivariate key means across heads."""
    delta = bank.mu_k[:, cluster_a] - bank.mu_k[:, cluster_b]
    pooled_variance = 0.5 * (bank.sigma_kk[:, cluster_a] + bank.sigma_kk[:, cluster_b])
    standardized = delta.square() / pooled_variance.clamp_min(1e-6)
    distance = torch.sqrt(standardized.mean())
    if not bool(torch.isfinite(distance)):
        raise RuntimeError("cluster mean distance is non-finite.")
    return float(distance.detach())


def birth_and_merge(
    bank: ClusterBank,
    k: Any,
    v: Any,
    *,
    birth_threshold: float,
    merge_threshold: float | None = None,
    outlier_top_k: int = 4,
) -> tuple[ClusterBank, dict]:
    """DPM-style birth/merge (E2.md section 4), evaluated once per chunk on the chunk's raw ``(k, v)``
    (``(b, t, n_head, d_head)``). Discrete structural decisions (which slot is born, which pair merges) are
    made from detached statistics -- birth/merge changes the STATE'S SHAPE, which cannot itself carry a
    gradient; the ongoing per-token responsibility path (:func:`update_cluster_bank`) is where E2.md section
    3.4's gradient flow actually lives.

    Returns ``(new_bank, receipt)``. ``receipt`` includes ``"birthed"`` (bool), ``"merged"`` (list of
    ``(i, j)`` pairs merged), ``"misfit"`` (per-active-cluster mean residual norm, E2.md section 4's misfit
    receipt), and ``"per_cluster_outlier_tokens"`` (consumed by
    :func:`mixle.experimental.kv_cache_quant.quantize_cluster_outliers`). A true
    ``"birth_incorporated_batch"`` means the birth initialization already counted this batch; callers
    should normally use :func:`ingest_cluster_batch`, which enforces the correct single-update path.
    """
    _require_torch()
    if isinstance(outlier_top_k, bool) or not isinstance(outlier_top_k, int) or outlier_top_k < 0:
        raise ValueError("outlier_top_k must be a non-negative exact integer.")
    n_head, max_clusters, d_head = bank.mu_k.shape
    device = bank.mu_k.device
    dtype = bank.mu_k.dtype
    if (
        not torch.is_tensor(k)
        or not torch.is_tensor(v)
        or k.ndim != 4
        or v.shape != k.shape
        or k.shape[0] <= 0
        or k.shape[1] <= 0
    ):
        raise ValueError("k and v must be non-empty tensors with matching (batch, time, head, dim) shape.")
    if k.device != device or v.device != device or k.dtype != dtype or v.dtype != dtype:
        raise ValueError("k and v must match the cluster bank device and dtype.")
    if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):
        raise ValueError("k and v must contain only finite values.")
    b, t, h, d = k.shape
    if h != n_head or d != d_head:
        raise ValueError("k and v head dimensions must match the cluster bank.")

    receipt: dict[str, Any] = {
        "birthed": False,
        "birth_incorporated_batch": False,
        "merged": [],
        "misfit": {},
        "per_cluster_outlier_tokens": {},
    }

    # --- birth --------------------------------------------------------------------------------------
    k_bar = k.mean(dim=(0, 1))  # (h, d) chunk pooled mean, per E2.md section 4
    v_bar = v.mean(dim=(0, 1))
    if bank.n_clusters == 0:
        best_score = torch.full((n_head,), float("-inf"), device=device, dtype=dtype)
    else:
        n = bank.n_clusters
        _, logits = _mgf_core(
            k_bar[None, None],
            bank.mu_k[:, :n],
            bank.mu_v[:, :n],
            bank.sigma_kk[:, :n],
            bank.sigma_vk[:, :n],
            bank.count[:, :n],
        )
        best_score = logits[0, 0].max(dim=-1).values  # (h,)

    if bool((best_score < birth_threshold).all()) and bank.n_clusters < bank.max_clusters:
        slot = bank.n_clusters
        n_tok = float(b * t)
        var_k = ((k - k_bar[None, None]) ** 2).mean(dim=(0, 1))  # (h, d)
        dk = (k - k_bar[None, None]).reshape(b * t, n_head, d_head).permute(1, 0, 2)  # (h, bt, d)
        dv = (v - v_bar[None, None]).reshape(b * t, n_head, d_head).permute(1, 0, 2)
        cross_kv = torch.einsum("hti,htj->hij", dv, dk) / n_tok  # (h, d_v, d_k)

        new_count = bank.count.clone()
        new_mu_k = bank.mu_k.clone()
        new_mu_v = bank.mu_v.clone()
        new_sigma_kk = bank.sigma_kk.clone()
        new_sigma_vk = bank.sigma_vk.clone()
        new_count[:, slot] = n_tok
        new_mu_k[:, slot] = k_bar
        new_mu_v[:, slot] = v_bar
        new_sigma_kk[:, slot] = var_k.clamp_min(1e-6)
        new_sigma_vk[:, slot] = cross_kv
        bank = replace(
            bank,
            count=new_count,
            mu_k=new_mu_k,
            mu_v=new_mu_v,
            sigma_kk=new_sigma_kk,
            sigma_vk=new_sigma_vk,
            n_clusters=bank.n_clusters + 1,
        )
        receipt["birthed"] = True
        receipt["birth_incorporated_batch"] = True
    # else: birth skipped (bank full, or an existing cluster already fits well). Per E2.md section 4 the
    # token should "fall through to the least-recently-updated cluster instead"; this implementation lets
    # the ordinary softmax responsibility path (cluster_responsibilities / update_cluster_bank) do that
    # fall-through -- the softmax always assigns every token to *some* cluster (its best-scoring one under
    # the same log-partition the birth check itself used), which is a real fallback, not a fabricated LRU
    # tracker this implementation doesn't build. Documented here rather than silently claimed as LRU.

    # --- merge ----------------------------------------------------------------------------------------
    n = bank.n_clusters
    merged_pairs: list[tuple[int, int]] = []
    if n >= 2:
        alive = list(range(n))
        i = 0
        while i < len(alive):
            j = i + 1
            merged_here = False
            while j < len(alive):
                ci, cj = alive[i], alive[j]
                distance = _cluster_mean_distance(bank, ci, cj)
                n_small = float(min(float(bank.count[:, ci].min().detach()), float(bank.count[:, cj].min().detach())))
                threshold = (
                    merge_threshold if merge_threshold is not None else 2.65 + 6.0 / math.sqrt(max(n_small, 1.0))
                )
                if distance < threshold:
                    n_new, mu_k_new, mu_v_new, sigma_kk_new, sigma_vk_new = _pooled_combine(
                        bank.count[:, ci],
                        bank.mu_k[:, ci],
                        bank.mu_v[:, ci],
                        bank.sigma_kk[:, ci],
                        bank.sigma_vk[:, ci],
                        bank.count[:, cj],
                        bank.mu_k[:, cj],
                        bank.mu_v[:, cj],
                        bank.sigma_kk[:, cj],
                        bank.sigma_vk[:, cj],
                    )
                    new_count = bank.count.clone()
                    new_mu_k = bank.mu_k.clone()
                    new_mu_v = bank.mu_v.clone()
                    new_sigma_kk = bank.sigma_kk.clone()
                    new_sigma_vk = bank.sigma_vk.clone()
                    new_count[:, ci] = n_new
                    new_mu_k[:, ci] = mu_k_new
                    new_mu_v[:, ci] = mu_v_new
                    new_sigma_kk[:, ci] = sigma_kk_new
                    new_sigma_vk[:, ci] = sigma_vk_new
                    # compact: move the last live slot into cj's now-vacant position (unless cj was last)
                    last = n - 1
                    if cj != last:
                        new_count[:, cj] = bank.count[:, last]
                        new_mu_k[:, cj] = bank.mu_k[:, last]
                        new_mu_v[:, cj] = bank.mu_v[:, last]
                        new_sigma_kk[:, cj] = bank.sigma_kk[:, last]
                        new_sigma_vk[:, cj] = bank.sigma_vk[:, last]
                    new_count[:, last] = 0.0
                    new_mu_k[:, last] = 0.0
                    new_mu_v[:, last] = 0.0
                    new_sigma_kk[:, last] = 0.0
                    new_sigma_vk[:, last] = 0.0
                    bank = replace(
                        bank,
                        count=new_count,
                        mu_k=new_mu_k,
                        mu_v=new_mu_v,
                        sigma_kk=new_sigma_kk,
                        sigma_vk=new_sigma_vk,
                        n_clusters=n - 1,
                    )
                    merged_pairs.append((ci, cj))
                    receipt.setdefault("merge_distances", []).append(distance)
                    n -= 1
                    alive = list(range(n))
                    merged_here = True
                    break
                j += 1
            if not merged_here:
                i += 1
    receipt["merged"] = merged_pairs

    # --- misfit receipt (E2.md section 4) --------------------------------------------------------------
    n = bank.n_clusters
    if n > 0:
        with torch.no_grad():
            r = cluster_responsibilities(k, bank)[..., :n]  # (b, t, h, n)
            # One hard token-level membership is required by the storage seam. Head-specific threshold
            # masks overlap and therefore cannot conserve tokens across cluster tails.
            token_membership = r.mean(dim=2).argmax(dim=-1).reshape(-1)  # (b*t,)
            mu_k = bank.mu_k[:, :n]
            mu_v = bank.mu_v[:, :n]
            sigma_kk = bank.sigma_kk[:, :n].clamp_min(1e-6)
            sigma_vk = bank.sigma_vk[:, :n]
            dk = k[:, :, :, None, :] - mu_k[None, None]  # (b, t, h, c, d)
            whitened = dk / sigma_kk[None, None]
            predicted_v = mu_v[None, None] + torch.einsum("hcij,bthcj->bthci", sigma_vk, whitened)
            resid = v[:, :, :, None, :] - predicted_v  # (b, t, h, c, d)
            resid_norm = resid.norm(dim=-1)  # (b, t, h, c)

            misfit_by_cluster: dict[int, float] = {}
            outliers: dict[int, dict[str, Any]] = {}
            flat_k = k.reshape(b * t, n_head, d_head)
            flat_v = v.reshape(b * t, n_head, d_head)
            for c in range(n):
                member_indices = torch.nonzero(token_membership == c, as_tuple=False).flatten()
                if member_indices.numel():
                    flat_resid_bt = resid_norm[:, :, :, c].mean(dim=2).reshape(-1)
                    misfit_by_cluster[c] = float(flat_resid_bt[member_indices].mean())
                    # per-token (head-averaged) residual so the top-k indices align 1:1 with flat_k/flat_v's
                    # (b*t) token axis -- an outlier is a TOKEN (its full multi-head k/v), not a (token, head)
                    # pair, matching what a real ProfileQuantized consumer (I2/G4) would want to store.
                    n_outliers = min(outlier_top_k, int(member_indices.numel()))
                    local_top = torch.topk(flat_resid_bt[member_indices], k=n_outliers).indices
                    top = member_indices[local_top]
                    outliers[c] = {
                        "k": flat_k[top].detach(),
                        "v": flat_v[top].detach(),
                        "indices": top.detach(),
                        "member_indices": member_indices.detach(),
                    }
                else:
                    misfit_by_cluster[c] = 0.0
            receipt["misfit"] = misfit_by_cluster
            receipt["per_cluster_outlier_tokens"] = outliers
            weights = torch.tensor(
                [max(misfit_by_cluster.get(c, 0.0), 0.0) for c in range(n)], device=device, dtype=dtype
            )
            counts = bank.count[:, :n].mean(dim=0).clamp_min(1e-8)
            receipt["misfit_scalar"] = float((weights * counts).sum() / counts.sum()) if n > 0 else 0.0

    return bank, receipt


def ingest_cluster_batch(
    bank: ClusterBank,
    k: Any,
    v: Any,
    *,
    birth_threshold: float,
    merge_threshold: float | None = None,
    outlier_top_k: int = 4,
) -> tuple[ClusterBank, dict[str, Any]]:
    """Ingest one non-overlapping far-field batch exactly once.

    A birth seeds its new component with the batch and therefore must not be followed by a second
    responsibility update for the same observations. If no component is born, the ordinary soft update
    incorporates the batch. The returned receipt verifies that every head's total count increased by
    exactly ``batch * time`` despite any merge.
    """
    before = _bank_total_per_head(bank)
    bank, receipt = birth_and_merge(
        bank,
        k.detach(),
        v.detach(),
        birth_threshold=birth_threshold,
        merge_threshold=merge_threshold,
        outlier_top_k=outlier_top_k,
    )
    if bank.n_clusters > 0 and not receipt["birth_incorporated_batch"]:
        responsibilities = cluster_responsibilities(k, bank)
        bank = update_cluster_bank(bank, k, v, responsibilities)
        receipt["soft_update_applied"] = True
    else:
        receipt["soft_update_applied"] = False

    batch_tokens = int(k.shape[0] * k.shape[1])
    delta = _bank_total_per_head(bank) - before
    expected = delta.new_full(delta.shape, float(batch_tokens))
    conserved = bool(torch.allclose(delta, expected, atol=1e-4, rtol=1e-5))
    receipt["ingested_tokens_per_head"] = [float(value) for value in delta.detach().cpu()]
    receipt["expected_ingested_tokens_per_head"] = batch_tokens
    receipt["count_conserved"] = conserved
    if not conserved:
        raise RuntimeError(
            "cluster ingestion did not count every observation exactly once: "
            f"delta={receipt['ingested_tokens_per_head']}, expected={batch_tokens}."
        )
    return bank, receipt


# -------------------------------------------------------------------------------------------------------
# The mechanism itself (E2.md section 2): ContextMechanism protocol, near+far combined-softmax attention.
# -------------------------------------------------------------------------------------------------------

if _HAS_TORCH:

    class MomentClosureAttention(nn.Module):
        """``ContextMechanism`` (E1 protocol): near field = E1's exact windowed attention; far field =
        attention against a per-layer :class:`ClusterBank` via the MGF identity; combined per query by a
        SINGLE joint softmax over both (E2.md section 3.3), not two independently-normalized attentions
        blended by a gate.
        """

        def __init__(
            self,
            vocab: int,
            *,
            d_model: int = 32,
            n_layer: int = 2,
            n_head: int = 2,
            window: int = 16,
            max_clusters: int = 4,
            birth_threshold: float = -2.0,
            merge_threshold: float | None = None,
        ) -> None:
            super().__init__()
            for name, value in {
                "vocab": vocab,
                "d_model": d_model,
                "n_layer": n_layer,
                "n_head": n_head,
                "window": window,
                "max_clusters": max_clusters,
            }.items():
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive exact integer.")
            if d_model % n_head:
                raise ValueError("d_model must be divisible by n_head.")
            if (d_model // n_head) % 2:
                raise ValueError("d_model / n_head must be even for RoPE.")
            if not math.isfinite(float(birth_threshold)):
                raise ValueError("birth_threshold must be finite.")
            if merge_threshold is not None and not math.isfinite(float(merge_threshold)):
                raise ValueError("merge_threshold must be finite when provided.")
            self.vocab = int(vocab)
            self.d_model = int(d_model)
            self.n_layer = int(n_layer)
            self.n_head = int(n_head)
            self.head_dim = d_model // n_head
            self.window = int(window)
            self.max_clusters = int(max_clusters)
            self.birth_threshold = float(birth_threshold)
            self.merge_threshold = merge_threshold

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
            self.head.weight = self.tok.weight

            self.last_misfit: float = 0.0  # self-reported per-step signal, mean over layers (E2.md section 4)
            self.last_receipts: list[dict] = []

        def init_state(self, batch_size: int, *, device: str = "cpu") -> MomentClosureState:
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
                raise ValueError("batch_size must be a positive integer.")
            dev = torch.device(device)
            banks = [
                _empty_cluster_bank(
                    self.n_head,
                    self.max_clusters,
                    self.head_dim,
                    device=dev,
                    dtype=self.tok.weight.dtype,
                )
                for _ in range(self.n_layer)
            ]
            near = SlidingWindowState(
                cache_k=[None] * self.n_layer,
                cache_v=[None] * self.n_layer,
                pos=0,
                batch_size=batch_size,
                n_head=self.n_head,
                head_dim=self.head_dim,
                device=dev,
            )
            return MomentClosureState(near=near, banks=banks, batch_size=batch_size)

        def detach(self, state: MomentClosureState) -> MomentClosureState:
            near = SlidingWindowState(
                cache_k=[t.detach() if t is not None else None for t in state.near.cache_k],
                cache_v=[t.detach() if t is not None else None for t in state.near.cache_v],
                pos=state.near.pos,
            )
            if state.detach_horizon_clusters:
                banks = [b.detach() for b in state.banks]
            else:
                banks = list(state.banks)
            return MomentClosureState(
                near=near,
                banks=banks,
                batch_size=state.batch_size,
                detach_horizon_clusters=state.detach_horizon_clusters,
            )

        def step(self, state: MomentClosureState, chunk: tuple[Any, Any]) -> tuple[MomentClosureState, Any]:
            if not isinstance(state, MomentClosureState):
                raise TypeError("state must be a MomentClosureState.")
            if not isinstance(chunk, tuple) or len(chunk) != 2:
                raise ValueError("chunk must be an (x, y) tuple.")
            x, y = chunk
            if not torch.is_tensor(x) or not torch.is_tensor(y) or x.ndim != 2 or y.shape != x.shape or x.shape[1] <= 0:
                raise ValueError("x and y must be non-empty tensors with matching (batch, time) shape.")
            b, t = x.shape
            if b != state.batch_size:
                raise ValueError("chunk batch size does not match the initialized state.")
            if len(state.banks) != self.n_layer:
                raise ValueError("state must contain one cluster bank per layer.")
            device = x.device
            query_positions = torch.arange(state.near.pos, state.near.pos + t, device=device)

            h = self.tok(x)
            new_cache_k: list[Any] = []
            new_cache_v: list[Any] = []
            new_banks: list[ClusterBank] = []
            receipts: list[dict] = []
            misfits: list[float] = []

            for layer in range(self.n_layer):
                hn = self.ln1[layer](h)
                qkv = self.qkv[layer](hn).reshape(b, t, 3, self.n_head, self.head_dim)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (b, t, n_head, head_dim)

                cache_k, cache_v = state.near.cache_k[layer], state.near.cache_v[layer]
                if cache_k is not None:
                    if cache_v is None or cache_v.shape != cache_k.shape or cache_k.shape[0] != b:
                        raise ValueError(f"layer {layer} near K/V cache has an invalid layout.")
                    cache_len = cache_k.shape[1]
                    _representation_receipt(
                        state.banks[layer],
                        batch_size=b,
                        positions_seen=state.near.pos,
                        near_tokens_per_stream=cache_len,
                        evicted_tokens_per_stream=0,
                    )
                    key_positions = torch.arange(state.near.pos - cache_len, state.near.pos + t, device=device)
                    k_full = torch.cat([cache_k, k], dim=1)
                    v_full = torch.cat([cache_v, v], dim=1)
                else:
                    if cache_v is not None:
                        raise ValueError(f"layer {layer} has a partial near cache.")
                    _representation_receipt(
                        state.banks[layer],
                        batch_size=b,
                        positions_seen=state.near.pos,
                        near_tokens_per_stream=0,
                        evicted_tokens_per_stream=0,
                    )
                    key_positions = query_positions
                    k_full, v_full = k, v

                sin_q, cos_q = _rope_angles(query_positions, self.head_dim)
                sin_k, cos_k = _rope_angles(key_positions, self.head_dim)
                q_rope = _apply_rope(q, sin_q, cos_q)
                k_full_rope = _apply_rope(k_full, sin_k, cos_k)

                delta = query_positions[:, None] - key_positions[None, :]  # (t, len(keys))
                allowed = (delta >= 0) & (delta < self.window)
                near_mask = torch.zeros(t, key_positions.shape[0], device=device)
                near_mask = near_mask.masked_fill(~allowed, float("-inf"))

                qh = q_rope.transpose(1, 2)  # (b, n_head, t, head_dim)
                kh = k_full_rope.transpose(1, 2)
                vh = v_full.transpose(1, 2)
                near_logits = (qh @ kh.transpose(-2, -1)) / (self.head_dim**0.5)  # (b, n_head, t, len(keys))
                near_logits = near_logits + near_mask[None, None]

                # far field: MGF mixture attention against this layer's ClusterBank (E2.md section 3.2).
                # Queries do NOT get RoPE applied for the far-field path -- the cluster bank summarizes
                # positions across the whole (position-mixed) history, so there is no single relative
                # offset to rotate by; this mirrors the population-stationarity assumption
                # moment_propagation.attention_law already documents for the single-cluster case.
                bank = state.banks[layer]
                far_out, far_logits = mgf_cluster_attention(q, bank)  # (b,t,c,h,d) / (b,t,c,h)
                n_c = bank.n_clusters

                if n_c > 0:
                    far_logits_bh = far_logits.permute(0, 3, 1, 2)  # (b, n_head, t, c)
                    combined = torch.cat([near_logits, far_logits_bh], dim=-1)  # (b, n_head, t, len(keys)+c)
                    weights = combined.softmax(dim=-1)
                    near_w = weights[..., : key_positions.shape[0]]  # (b, n_head, t, len(keys))
                    far_w = weights[..., key_positions.shape[0] :]  # (b, n_head, t, c)
                    near_out = near_w @ vh  # (b, n_head, t, head_dim)
                    far_out_bh = far_out.permute(0, 3, 1, 2, 4)  # (b, n_head, t, c, d)
                    far_contrib = torch.einsum("bhtc,bhtcd->bhtd", far_w, far_out_bh)
                    out = (near_out + far_contrib).transpose(1, 2).reshape(b, t, self.d_model)
                else:
                    weights = near_logits.softmax(dim=-1)
                    out = (weights @ vh).transpose(1, 2).reshape(b, t, self.d_model)

                h = h + self.proj[layer](out)
                h = h + self.mlp[layer](self.ln2[layer](h))

                keep = min(self.window, k_full.shape[1])
                evicted = k_full.shape[1] - keep
                new_cache_k.append(k_full[:, -keep:])
                new_cache_v.append(v_full[:, -keep:])

                # The far bank receives only the exact tokens evicted from the near cache. Current/retained
                # tokens remain represented solely by the exact window.
                if evicted > 0:
                    bank, receipt = ingest_cluster_batch(
                        bank,
                        k_full[:, :evicted],
                        v_full[:, :evicted],
                        birth_threshold=self.birth_threshold,
                        merge_threshold=self.merge_threshold,
                    )
                else:
                    receipt = {
                        "birthed": False,
                        "birth_incorporated_batch": False,
                        "merged": [],
                        "misfit": {},
                        "per_cluster_outlier_tokens": {},
                        "soft_update_applied": False,
                        "ingested_tokens_per_head": [0.0] * self.n_head,
                        "expected_ingested_tokens_per_head": 0,
                        "count_conserved": True,
                    }
                receipt["accounting"] = _representation_receipt(
                    bank,
                    batch_size=b,
                    positions_seen=state.near.pos + t,
                    near_tokens_per_stream=keep,
                    evicted_tokens_per_stream=evicted,
                )
                new_banks.append(bank)
                receipts.append(receipt)
                misfits.append(float(receipt.get("misfit_scalar", 0.0)))

            logits = self.head(self.ln_f(h))
            loss = F.cross_entropy(logits.reshape(b * t, self.vocab), y.reshape(b * t))

            new_near = SlidingWindowState(cache_k=new_cache_k, cache_v=new_cache_v, pos=state.near.pos + t)
            new_state = MomentClosureState(
                near=new_near,
                banks=new_banks,
                batch_size=state.batch_size,
                detach_horizon_clusters=state.detach_horizon_clusters,
            )
            self.last_receipts = receipts
            self.last_misfit = float(sum(misfits) / len(misfits)) if misfits else 0.0
            return new_state, loss
