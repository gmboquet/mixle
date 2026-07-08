"""Distributed / out-of-core construction of the direct layout: map-reduce over data shards.

Everything :func:`mixle.utils.hvis.direct.model_map` learns from data is ADDITIVE over shards --
posterior masses and overlaps (``sum z``, ``sqrt(z)^T sqrt(z)``, ``z^T z``, triple products for the
nerve), per-(field, component) weighted moments behind the whiteners, and per-component weighted
moments of the fiber coordinates behind the chart PCA. The one order statistic (the occlusion
radius, a 90th percentile) is a cheap gather of one float per dominated point. So the whole layout
is a handful of map-reduce passes with EXACT combines -- the same shape as mixle's
estimator/accumulator machinery (``seq_estimate``'s Spark path, the multiprocessing/MPI encoded-
data handles), just for the visualization geometry instead of the model parameters:

1. :func:`fiber_stats` per shard, combined with ``+``  -> whiteners, vertices, nerve;
2. :func:`score_stats` per shard, combined with ``+``  -> chart loadings, spreads, scale
   (quadratic charts whose base dimension exceeds the lift cap need this pass twice: once for the
   pre-PCA, once for the lifted moments);
3. :func:`radius_stats` per shard (only when occlusion must resolve overlaps) -> radii percentile;
4. :meth:`~mixle.utils.hvis.direct.ModelMap.place` per shard -> coordinates (closed-form,
   embarrassingly parallel).

:func:`distributed_model_map` orchestrates the passes over an in-memory list of shards through any
``map``-shaped ``mapper`` (builtin ``map``, ``multiprocessing.Pool.map``, an MPI executor map, ...);
every stage function is a pure picklable callable and every stats object combines with ``+``, so on
Spark the same sequence is ``rdd.mapPartitions(lambda it: [fiber_stats(model, list(it))]).reduce(
lambda a, b: a + b)`` per pass with the small plan objects broadcast between passes, and coordinates
come from broadcasting the finished (geometry-only) map and calling ``.place`` per partition.

What does NOT distribute: the t-SNE/UMAP ``refine`` pass and embedding goals -- global sequential
optimizations by construction. At scale, run those on a driver-sized subsample and place the rest
(:mod:`mixle.utils.hvis.stream` is the incremental version of that recipe). Model FITTING is the
already-solved distributed problem (``seq_estimate``); this module assumes ``mix_model`` arrives
fitted and requires it explicitly.

The contract with the single-machine path is equality: ``distributed_model_map(chunks, model)``
reproduces ``model_map(concat(chunks), model)`` to floating-point summation order
(``hvis_distributed_test`` pins it), for any chunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial, reduce

import numpy as np

from mixle.utils.hvis.affinity import (
    _component_map_core,
    _field_log_density_features,
    _posteriors_and_loglikes,
)
from mixle.utils.hvis.direct import (
    _MAX_QUAD_BASE,
    ModelMap,
    _canonical_signs,
    _lift_quadratic,
    _quad_labels,
    _resolve_overlap,
    _sqrt_psd,
)

__all__ = [
    "FiberStats",
    "ScoreStats",
    "distributed_model_map",
    "fiber_stats",
    "fuzzy_nerve_from_stats",
    "radius_stats",
    "score_stats",
]


# ---------------------------------------------------------------------------------------------
# pass 1: posterior overlaps + raw per-field moments
# ---------------------------------------------------------------------------------------------


@dataclass
class _FieldMoments:
    """Additive raw moments for one coordinate-bearing field (pre-whitening).

    Moments cover only fully-finite rows; rows with any non-finite coordinate are carried verbatim
    in ``deferred`` (with their field posteriors) because their fill values -- the global finite
    column means -- are only known after the reduce. ``finite_col_sum/cnt`` cover finite ENTRIES of
    every row, matching ``_component_inv_covariances``' nanmean fill.
    """

    dim: int
    native: bool
    finite_col_sum: np.ndarray
    finite_col_cnt: np.ndarray
    n_rows: float
    g_sum: np.ndarray
    g_sum2: np.ndarray
    zf_sw: np.ndarray
    zf_sx: np.ndarray
    zf_sxx: np.ndarray
    deferred_x: list = field(default_factory=list)
    deferred_zf: list = field(default_factory=list)

    def __add__(self, other: _FieldMoments) -> _FieldMoments:
        if self.dim != other.dim or self.native != other.native:
            raise ValueError("cannot combine moments from different field layouts.")
        return _FieldMoments(
            dim=self.dim,
            native=self.native,
            finite_col_sum=self.finite_col_sum + other.finite_col_sum,
            finite_col_cnt=self.finite_col_cnt + other.finite_col_cnt,
            n_rows=self.n_rows + other.n_rows,
            g_sum=self.g_sum + other.g_sum,
            g_sum2=self.g_sum2 + other.g_sum2,
            zf_sw=self.zf_sw + other.zf_sw,
            zf_sx=self.zf_sx + other.zf_sx,
            zf_sxx=self.zf_sxx + other.zf_sxx,
            deferred_x=self.deferred_x + other.deferred_x,
            deferred_zf=self.deferred_zf + other.deferred_zf,
        )


@dataclass
class FiberStats:
    """Pass-1 additive statistics: everything the layout's GLOBAL geometry needs from data."""

    n: int
    mass: np.ndarray  # sum_i z_ik                      (K,)
    sqrt_overlap: np.ndarray  # sqrt(z)^T sqrt(z)       (K, K) -- Bhattacharyya numerator
    co: np.ndarray  # z^T z                             (K, K) -- co-claimed mass
    fields: list  # per leaf field: _FieldMoments, or None for coordinate-free fields
    triple: np.ndarray | None = None  # sum_i z_ia z_ib z_ic (K, K, K), for the distributed nerve

    def __add__(self, other: FiberStats) -> FiberStats:
        if len(self.fields) != len(other.fields):
            raise ValueError("cannot combine stats from different field layouts.")
        fields = []
        for a, b in zip(self.fields, other.fields):
            if (a is None) != (b is None):
                raise ValueError("cannot combine stats from different field layouts.")
            fields.append(None if a is None else a + b)
        triple = None
        if self.triple is not None and other.triple is not None:
            triple = self.triple + other.triple
        return FiberStats(
            n=self.n + other.n,
            mass=self.mass + other.mass,
            sqrt_overlap=self.sqrt_overlap + other.sqrt_overlap,
            co=self.co + other.co,
            fields=fields,
            triple=triple,
        )


def _field_posteriors(l_f: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    """The field-restricted posterior exactly as :func:`local_factors` computes it."""
    z_f = np.asarray(l_f, dtype=np.float64) + log_w
    finite_rows = np.isfinite(z_f).any(axis=1)
    if not np.all(finite_rows):
        z_f[~finite_rows] = log_w
    z_f -= z_f.max(axis=1, keepdims=True)
    np.exp(z_f, out=z_f)
    z_f /= z_f.sum(axis=1, keepdims=True)
    return z_f


def fiber_stats(mix_model, chunk, *, nerve_triple: bool = False) -> FiberStats:
    """Pass 1 over one shard. Combine shard results with ``+``; order never matters.

    ``nerve_triple=True`` also accumulates the (K, K, K) triple-overlap tensor so
    :func:`fuzzy_nerve_from_stats` can report triangles (an extra factor-K of work in this pass).
    """
    chunk = list(chunk)
    if not chunk:
        raise ValueError("fiber_stats needs a non-empty shard.")
    z, _ = _posteriors_and_loglikes(mix_model, data=chunk)
    k_count = z.shape[1]
    sq = np.sqrt(z)
    log_w = np.asarray(mix_model.log_w, dtype=np.float64).reshape(1, -1)

    triple = None
    if nerve_triple:
        triple = np.empty((k_count, k_count, k_count))
        for a in range(k_count):
            triple[a] = (z * z[:, a : a + 1]).T @ z

    fields: list = []
    for l_f, x_f, native_f in _field_log_density_features(list(mix_model.components), chunk):
        if x_f is None or np.asarray(x_f).ndim != 2 or np.asarray(x_f).shape[1] == 0:
            fields.append(None)
            continue
        x = np.asarray(x_f, dtype=np.float64)
        z_f = _field_posteriors(l_f, log_w)
        dim = x.shape[1]
        finite = np.isfinite(x)
        row_ok = finite.all(axis=1)
        xr, zr = x[row_ok], z_f[row_ok]
        fields.append(
            _FieldMoments(
                dim=dim,
                native=bool(native_f),
                finite_col_sum=np.where(finite, x, 0.0).sum(axis=0),
                finite_col_cnt=finite.sum(axis=0).astype(np.float64),
                n_rows=float(row_ok.sum()),
                g_sum=xr.sum(axis=0),
                g_sum2=xr.T @ xr,
                zf_sw=zr.sum(axis=0),
                zf_sx=zr.T @ xr,
                zf_sxx=np.einsum("ik,id,ie->kde", zr, xr, xr, optimize=True),
                deferred_x=[x[~row_ok]] if not row_ok.all() else [],
                deferred_zf=[z_f[~row_ok]] if not row_ok.all() else [],
            )
        )
    return FiberStats(
        n=len(chunk), mass=z.sum(axis=0), sqrt_overlap=sq.T @ sq, co=z.T @ z, fields=fields, triple=triple
    )


def fuzzy_nerve_from_stats(
    stats: FiberStats, *, edge_threshold: float = 0.02, triangle_threshold: float = 0.02
) -> dict:
    """:func:`mixle.utils.hvis.topology.fuzzy_nerve` from combined pass-1 statistics -- the
    distributed nerve. Triangles need ``fiber_stats(..., nerve_triple=True)``; without the tensor
    only the 1-skeleton is reported (and :func:`~mixle.utils.hvis.topology.nerve_report` would
    over-count holes, so it raises instead)."""
    k_count = stats.co.shape[0]
    safe = np.maximum(stats.mass, 1.0e-12)
    edges: dict[tuple[int, int], float] = {}
    for a in range(k_count):
        for b in range(a + 1, k_count):
            w = float(stats.co[a, b] / min(safe[a], safe[b]))
            if w > 0.0:
                edges[(a, b)] = w
    if stats.triple is None:
        raise ValueError("triangles need fiber_stats(..., nerve_triple=True); combine those stats and retry.")
    triangles: dict[tuple[int, int, int], float] = {}
    strong_pairs = {e for e, w in edges.items() if w >= edge_threshold}
    for a in range(k_count):
        for b in range(a + 1, k_count):
            if (a, b) not in strong_pairs:
                continue
            for c in range(b + 1, k_count):
                if (a, c) not in strong_pairs or (b, c) not in strong_pairs:
                    continue
                w = float(stats.triple[a, b, c] / min(safe[a], safe[b], safe[c]))
                if w > 0.0:
                    triangles[(a, b, c)] = w
    return {
        "masses": stats.mass,
        "edges": edges,
        "triangles": triangles,
        "edge_threshold": float(edge_threshold),
        "triangle_threshold": float(triangle_threshold),
    }


# ---------------------------------------------------------------------------------------------
# driver finalization A: whiteners + vertices from pass-1 stats
# ---------------------------------------------------------------------------------------------


def _inv_covs_from_moments(m: _FieldMoments, ridge: float = 1.0e-4) -> np.ndarray:
    """`_component_inv_covariances` finalization from additive moments (same fills, ridges, and
    small-mass fallback; deferred non-finite rows folded in with the now-known global fills)."""
    dim = m.dim
    k_count = m.zf_sw.shape[0]
    fill = np.where(m.finite_col_cnt > 0, m.finite_col_sum / np.maximum(m.finite_col_cnt, 1.0), 0.0)

    n = m.n_rows
    g_sum, g_sum2 = m.g_sum.copy(), m.g_sum2.copy()
    zf_sw, zf_sx, zf_sxx = m.zf_sw.copy(), m.zf_sx.copy(), m.zf_sxx.copy()
    for xd, zd in zip(m.deferred_x, m.deferred_zf):
        xf = np.where(np.isfinite(xd), xd, fill[None, :])
        n += xd.shape[0]
        g_sum += xf.sum(axis=0)
        g_sum2 += xf.T @ xf
        zf_sw += zd.sum(axis=0)
        zf_sx += zd.T @ xf
        zf_sxx += np.einsum("ik,id,ie->kde", zd, xf, xf, optimize=True)

    mean = g_sum / max(n, 1.0)
    global_cov = (g_sum2 - n * np.outer(mean, mean)) / max(n - 1, 1)
    scale = float(np.trace(global_cov) / max(dim, 1))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    global_cov = global_cov + np.eye(dim) * (ridge * scale + 1.0e-8)

    inv_covs = np.empty((k_count, dim, dim), dtype=np.float64)
    for k in range(k_count):
        sw = float(zf_sw[k])
        if sw <= dim + 1.0e-8:
            cov = global_cov
        else:
            mu = zf_sx[k] / sw
            cov = zf_sxx[k] / sw - np.outer(mu, mu)
            local_scale = float(np.trace(cov) / max(dim, 1))
            if not np.isfinite(local_scale) or local_scale <= 0.0:
                local_scale = scale
            cov = cov + np.eye(dim) * (ridge * local_scale + 1.0e-8)
        inv_covs[k] = np.linalg.pinv(cov)
    return inv_covs


def _strong_edges(co: np.ndarray, mass: np.ndarray, edge_threshold: float) -> set:
    safe = np.maximum(mass, 1.0e-12)
    k_count = co.shape[0]
    return {
        (a, b)
        for a in range(k_count)
        for b in range(a + 1, k_count)
        if co[a, b] / min(safe[a], safe[b]) >= edge_threshold
    }


@dataclass
class _ScorePlan:
    """Small broadcastable state for passes 2/3: the fitted transforms, no data."""

    mix_model: object
    keep: list
    field_specs: list
    whiteners: list  # [k][field] -> sqrt of inv_cov
    chart: str
    pre: list | None = None  # per-component (mu, directions) for the wide quadratic lift
    want_feats: bool = True
    # radius-pass extras (set for pass 3 only)
    fiber_means: list | None = None
    loadings: list | None = None
    fiber_scale: float | None = None


def _plan_from_stats(mix_model, stats: FiberStats, field_weights, chart: str) -> tuple[_ScorePlan, list]:
    if field_weights is None:
        field_weights = [1.0] * len(stats.fields)
    elif len(field_weights) != len(stats.fields):
        raise ValueError(
            "field_weights has %d entries but the model flattens to %d leaf fields."
            % (len(field_weights), len(stats.fields))
        )
    keep, field_specs, whiteners_by_field, labels = [], [], [], []
    for f_pos, (m, w_f) in enumerate(zip(stats.fields, field_weights)):
        if w_f < 0:
            raise ValueError("field_weights must be non-negative.")
        if m is None:
            continue
        dof = 1.0 if m.native else float(m.dim)
        keep.append(f_pos)
        field_specs.append({"dim": m.dim, "weight_sqrt": float(np.sqrt(float(w_f) / dof))})
        inv_covs = _inv_covs_from_moments(m)
        whiteners_by_field.append([_sqrt_psd(inv_covs[k]) for k in range(inv_covs.shape[0])])
        kind = "native" if m.native else "typicality"
        labels.extend([f"field{f_pos}[{kind}]:{c}" for c in range(m.dim)])
    if not keep:
        raise ValueError("no coordinate-bearing fields: nothing to lay out.")
    k_count = stats.mass.shape[0]
    whiteners = [[whiteners_by_field[f][k] for f in range(len(keep))] for k in range(k_count)]
    plan = _ScorePlan(mix_model=mix_model, keep=keep, field_specs=field_specs, whiteners=whiteners, chart=chart)
    return plan, labels


# ---------------------------------------------------------------------------------------------
# pass 2: whitened fiber moments (and chart-lifted moments once the lift is known)
# ---------------------------------------------------------------------------------------------


@dataclass
class ScoreStats:
    """Pass-2 additive statistics: per-component moments of the whitened fiber coordinates ``u``
    and (when computable in the same pass) of the chart features, plus the clamped-weight moments
    behind the spread normalization."""

    sw: np.ndarray  # (K,)
    su: list  # [k] -> (Du,)
    suu: list  # [k] -> (Du, Du)
    sf: list | None  # feats moments, None when the lift needs the pre-PCA first
    sff: list | None
    swc: np.ndarray | None  # clamped weights: sum max(z_k, 1e-12)
    sfc: list | None
    sffc: list | None

    def __add__(self, other: ScoreStats) -> ScoreStats:
        both = self.sf is not None and other.sf is not None
        return ScoreStats(
            sw=self.sw + other.sw,
            su=[a + b for a, b in zip(self.su, other.su)],
            suu=[a + b for a, b in zip(self.suu, other.suu)],
            sf=[a + b for a, b in zip(self.sf, other.sf)] if both else None,
            sff=[a + b for a, b in zip(self.sff, other.sff)] if both else None,
            swc=self.swc + other.swc if both else None,
            sfc=[a + b for a, b in zip(self.sfc, other.sfc)] if both else None,
            sffc=[a + b for a, b in zip(self.sffc, other.sffc)] if both else None,
        )


def _chunk_u(plan: _ScorePlan, chunk: list) -> tuple[np.ndarray, list]:
    """Recompute (z, [u_k]) for a shard from the broadcast plan -- the map side of passes 2-3."""
    z, _ = _posteriors_and_loglikes(plan.mix_model, data=chunk)
    terms = list(_field_log_density_features(list(plan.mix_model.components), chunk))
    xs = [np.asarray(terms[f_pos][1], dtype=np.float64) for f_pos in plan.keep]
    us = [
        np.column_stack([(x @ w) * spec["weight_sqrt"] for x, w, spec in zip(xs, plan.whiteners[k], plan.field_specs)])
        for k in range(z.shape[1])
    ]
    return z, us


def _plan_feats(plan: _ScorePlan, u: np.ndarray, k: int) -> np.ndarray | None:
    """Chart features for component k, or None when the wide-lift pre-PCA is not known yet."""
    if plan.chart == "linear":
        return u
    if u.shape[1] > _MAX_QUAD_BASE:
        if plan.pre is None:
            return None
        mu, p = plan.pre[k]
        return _lift_quadratic((u - mu) @ p)
    return _lift_quadratic(u)


def score_stats(plan: _ScorePlan, chunk) -> ScoreStats:
    """Pass 2 over one shard (pure; combine with ``+``)."""
    chunk = list(chunk)
    z, us = _chunk_u(plan, chunk)
    k_count = z.shape[1]
    sw = np.array([float(z[:, k].sum()) for k in range(k_count)])
    su, suu = [], []
    feats_known = True
    sf, sff, sfc, sffc = [], [], [], []
    swc = np.zeros(k_count)
    for k in range(k_count):
        wk = z[:, k]
        u = us[k]
        su.append(wk @ u)
        suu.append((u * wk[:, None]).T @ u)
        feats = _plan_feats(plan, u, k) if plan.want_feats else None
        if feats is None:
            feats_known = False
            continue
        wc = np.maximum(wk, 1.0e-12)
        swc[k] = float(wc.sum())
        sf.append(wk @ feats)
        sff.append((feats * wk[:, None]).T @ feats)
        sfc.append(wc @ feats)
        sffc.append((feats * wc[:, None]).T @ feats)
    if not (plan.want_feats and feats_known):
        sf = sff = sfc = sffc = None
        swc = None
    return ScoreStats(sw=sw, su=su, suu=suu, sf=sf, sff=sff, swc=swc, sfc=sfc, sffc=sffc)


# ---------------------------------------------------------------------------------------------
# pass 3: occlusion radii (dominated-point score norms; the one gathered order statistic)
# ---------------------------------------------------------------------------------------------


def radius_stats(plan: _ScorePlan, chunk) -> list:
    """Pass 3 over one shard: per component, the scaled score norms of the points it dominates."""
    chunk = list(chunk)
    z, us = _chunk_u(plan, chunk)
    dominant = z.argmax(axis=1)
    out = []
    for k in range(z.shape[1]):
        rows = dominant == k
        if not rows.any():
            out.append(np.zeros(0))
            continue
        feats = _plan_feats(plan, us[k][rows], k)
        scores = (feats - plan.fiber_means[k]) @ plan.loadings[k] * plan.fiber_scale
        out.append(np.linalg.norm(scores, axis=1))
    return out


# ---------------------------------------------------------------------------------------------
# the orchestrated front door
# ---------------------------------------------------------------------------------------------


def _compose_chunk(fitted: ModelMap, chunk) -> tuple[np.ndarray, np.ndarray]:
    chunk = list(chunk)
    z, _ = _posteriors_and_loglikes(fitted._model, data=chunk)  # noqa: SLF001 - stage fn of this module
    return fitted.place(chunk), z


def distributed_model_map(
    chunks,
    mix_model,
    emb_dim: int = 2,
    *,
    spread: float = 0.35,
    chart: str = "linear",
    occlusion: bool = True,
    occlusion_margin: float = 1.05,
    edge_threshold: float = 0.02,
    field_weights=None,
    mapper=None,
    with_points: bool = True,
) -> ModelMap:
    """:func:`~mixle.utils.hvis.direct.model_map` over sharded data (see the module docstring).

    ``chunks`` is a materialized sequence of data shards (each itself a sequence); every pass maps
    over it once. ``mapper`` is any ``map``-shaped callable -- builtin ``map`` (default),
    ``multiprocessing.Pool.map``, an MPI executor's map -- the stage functions and their broadcast
    plans are picklable. ``mix_model`` is required: fitting is the estimator machinery's distributed
    problem (``seq_estimate``), not this one. ``with_points=False`` skips the coordinate pass and
    returns geometry only (empty ``coords``/``responsibilities``) -- broadcast the result and call
    ``.place`` per shard where the data lives. ``refine`` and goals are deliberately absent: global
    sequential optimization does not shard (subsample-and-place instead).
    """
    if chart not in ("linear", "quadratic"):
        raise ValueError("chart must be 'linear' or 'quadratic'.")
    if mix_model is None:
        raise ValueError(
            "distributed_model_map requires a fitted mix_model; fit one first (seq_estimate distributes that)."
        )
    chunks = [c for c in (list(c) for c in chunks) if c]
    if not chunks:
        raise ValueError("no non-empty shards.")
    run = map if mapper is None else mapper

    stats = reduce(lambda a, b: a + b, run(partial(fiber_stats, mix_model), chunks))
    n, k_count = stats.n, stats.mass.shape[0]
    plan, labels = _plan_from_stats(mix_model, stats, field_weights, chart)
    base_labels = list(labels)

    if k_count == 1:
        vertices = np.zeros((1, emb_dim))
    else:
        strong = _strong_edges(stats.co, stats.mass, edge_threshold)
        vertices = _component_map_core(stats.sqrt_overlap, stats.mass, strong, emb_dim, "nerve")

    score = reduce(lambda a, b: a + b, run(partial(score_stats, plan), chunks))
    if score.sf is None:  # wide quadratic lift: fit the per-component pre-PCA, then one more pass
        pre = []
        for k in range(k_count):
            sw = max(float(score.sw[k]), 1.0e-12)
            mu_u = score.su[k] / sw
            cov_u = score.suu[k] / sw - np.outer(mu_u, mu_u)
            vals, vecs = np.linalg.eigh(cov_u)
            order = np.argsort(vals)[::-1][:_MAX_QUAD_BASE]
            pre.append((mu_u, _canonical_signs(vecs[:, order])))
        plan.pre = pre
        score = reduce(lambda a, b: a + b, run(partial(score_stats, plan), chunks))
    pre_list = plan.pre if plan.pre is not None else [None] * k_count

    fiber_means, loadings, residuals, stds = [], [], [], []
    for k in range(k_count):
        sw = max(float(score.sw[k]), 1.0e-12)
        mu_u = score.su[k] / sw
        cov_u = score.suu[k] / sw - np.outer(mu_u, mu_u)
        vals_u = np.sort(np.linalg.eigvalsh(cov_u))[::-1]
        total = float(vals_u.sum())
        residuals.append(0.0 if total <= 0 else float(max(0.0, 1.0 - vals_u[:emb_dim].sum() / total)))

        mu = score.sf[k] / sw
        cov = score.sff[k] / sw - np.outer(mu, mu)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:emb_dim]
        p = _canonical_signs(vecs[:, order])
        if p.shape[1] < emb_dim:
            p = np.column_stack([p, np.zeros((p.shape[0], emb_dim - p.shape[1]))])
        fiber_means.append(mu)
        loadings.append(p)

        swc = max(float(score.swc[k]), 1.0e-12)
        mc = score.sfc[k] / swc
        cov_c = score.sffc[k] / swc - np.outer(mu, mc) - np.outer(mc, mu) + np.outer(mu, mu)
        stds.append(float(np.sqrt(max(float(np.trace(p.T @ cov_c @ p)), 0.0))))

    if chart == "quadratic":
        labels = (
            _quad_labels(base_labels)
            if pre_list[0] is None
            else _quad_labels([f"pre_pc{i}" for i in range(_MAX_QUAD_BASE)])
        )

    gaps = [float(np.linalg.norm(vertices[a] - vertices[b])) for a in range(k_count) for b in range(a + 1, k_count)]
    nonzero_gaps = [g for g in gaps if g > 0]
    ref = min(nonzero_gaps) if nonzero_gaps else 1.0
    med_std = float(np.median([s for s in stds if s > 0]) or 1.0)
    scale = (spread * ref / med_std) if med_std > 0 else 1.0

    if occlusion and k_count > 1:
        plan.fiber_means, plan.loadings, plan.fiber_scale = fiber_means, loadings, scale
        per_chunk = list(run(partial(radius_stats, plan), chunks))
        radii = np.zeros(k_count)
        for k in range(k_count):
            mine = np.concatenate([arrs[k] for arrs in per_chunk])
            radii[k] = float(np.percentile(mine, 90)) if len(mine) else 0.0
        allowed = _strong_edges(stats.co, stats.mass, edge_threshold)
        vertices = _resolve_overlap(vertices, radii, allowed, margin=occlusion_margin)

    frames = []
    for k in range(k_count):
        if emb_dim != 2 or k_count < 2:
            frames.append(np.eye(emb_dim))
            continue
        others = [j for j in range(k_count) if j != k]
        nearest = min(others, key=lambda j: float(np.linalg.norm(vertices[j] - vertices[k])))
        radial = vertices[nearest] - vertices[k]
        norm = float(np.linalg.norm(radial))
        if norm <= 1.0e-12:
            frames.append(np.eye(2))
            continue
        radial = radial / norm
        tangent = np.array([-radial[1], radial[0]])
        frames.append(np.stack([tangent, radial]))

    result = ModelMap(
        coords=np.zeros((0, emb_dim)),
        vertices=vertices,
        responsibilities=np.zeros((0, k_count)),
        loadings=loadings,
        coord_labels=labels,
        frames=frames,
        chart=chart,
        chart_residuals=np.asarray(residuals),
        _model=mix_model,
        _transforms={"keep": plan.keep, "field_specs": plan.field_specs, "whiteners": plan.whiteners},
        _pre=pre_list,
        _fiber_means=fiber_means,
        _fiber_scale=scale,
        _emb_dim=emb_dim,
    )
    if with_points:
        parts = list(run(partial(_compose_chunk, result), chunks))
        result.coords = np.vstack([c for c, _z in parts]) if parts else np.zeros((n, emb_dim))
        result.responsibilities = np.vstack([zc for _c, zc in parts])
    return result
