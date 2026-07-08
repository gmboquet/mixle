"""The direct compositional layout: read the map off the model instead of re-inferring it.

t-SNE/UMAP exist to DISCOVER global and local structure from raw pairwise distances when there is
no model. HViS has a model, and the structure the neighbor optimizers spend a thousand stochastic
iterations inferring is already known in closed form: WHICH regimes exist and how they relate (the
posterior simplex and component overlap geometry), and WHERE each observation sits within its
regime (the whitened local/typicality coordinates). This module composes those two levels directly
-- the (posterior, remainder) decomposition as a layout:

    y_i = sum_k z_ik * (vertex_k + frame_k(fiber_scores_ik))

* vertices: :func:`~mixle.utils.hvis.affinity.component_map` -- geodesic layout of the cover's
  nerve (confusable regimes adjacent; rings render as rings). Deterministic.
* fibers: per component, a responsibility-weighted PCA of the per-field WHITENED local coordinates
  (native value coordinates or universal typicality coordinates -- the same geometry the 'local'
  affinity scores). Deterministic up to sign, and signs are canonicalized. The loadings are
  returned, so a fiber axis is NAMEABLE: "within regime k, axis 1 is field 2's coordinate 0".
  ``chart='quadratic'`` lifts the fiber features with degree-2 terms (explicit polynomial kernel,
  still closed-form and placeable) for regimes whose within-structure is curved; the per-component
  ``chart_residuals`` report says how much variance the linear chart leaves behind either way.
* frames: each chart gets its own on-screen frame, major axis oriented tangentially (orthogonal to
  the nearest other vertex) so neighboring charts' fringes cannot collide head-on.
* occlusion: components with NO measured overlap in the model must not overlap on screen -- a
  deterministic push-apart pass enforces it, while genuinely-overlapping components are allowed to
  overlap visually (screen overlap then MEANS model overlap).
* composition: barycentric in the posterior, so sharp points sit in their regime's local chart and
  mixed-membership points interpolate between charts.

No perplexity, no seed, no optimizer failure modes; out-of-sample placement (:meth:`ModelMap.place`)
is the same closed form, so streaming costs one matrix product. The precedent is Bishop & Tipping's
hierarchical mixture visualization / GTM: per-component local projections composed by
responsibility -- rebuilt here on HViS's field decomposition so it covers heterogeneous,
variable-length, and sequence data.

When to still reach for the neighbor optimizers: no trustworthy model, strongly nonlinear
within-regime manifolds a chart flattens, or pure exploration. ``refine=True`` runs t-SNE FROM this
layout (informative init, exaggeration off) so the optimizer only polishes local neighborhoods it
is actually good at -- the composition stays in charge of the global picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mixle.utils.hvis.affinity import (
    _field_log_density_features,
    _is_local_factor,
    _posteriors_and_loglikes,
    component_map,
    local_factors,
)

_MAX_QUAD_BASE = 8  # quadratic lift caps its base at this many pre-PCA directions


def _sqrt_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(mat, dtype=np.float64))
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T


def _canonical_signs(components: np.ndarray) -> np.ndarray:
    """Fix each principal direction's sign so its largest-magnitude loading is positive --
    determinism must not depend on LAPACK's arbitrary eigenvector signs."""
    flips = np.sign(components[np.abs(components).argmax(axis=0), np.arange(components.shape[1])])
    flips[flips == 0] = 1.0
    return components * flips


def _lift_quadratic(v: np.ndarray) -> np.ndarray:
    """Explicit degree-2 polynomial features: ``[v, v_i v_j for i <= j]`` -- a closed-form,
    deterministic, placement-compatible nonlinear chart."""
    n, m = v.shape
    prods = [v[:, i] * v[:, j] for i in range(m) for j in range(i, m)]
    return np.column_stack([v] + prods)


def _quad_labels(base: list[str]) -> list[str]:
    m = len(base)
    return list(base) + [f"{base[i]}*{base[j]}" for i in range(m) for j in range(i, m)]


def component_fiber_coords(mix_model, data, field_weights=None) -> tuple[np.ndarray, list[np.ndarray], list[str], dict]:
    """The shared fiber machinery: posteriors, per-component whitened field coordinates, labels.

    Returns ``(z, [u_k for each component], coord_labels, transforms)`` where ``u_k`` is the
    (n, D) concatenation of every coordinate-bearing field's values, whitened by component ``k``'s
    local inverse covariance and weighted per field. ``transforms`` carries everything needed to
    reproduce the coordinates for NEW data (used by :meth:`ModelMap.place` and by
    :func:`mixle.utils.hvis.topology.model_fit_health`).
    """
    data = list(data)
    z, _ = _posteriors_and_loglikes(mix_model, data=data)
    k_count = z.shape[1]
    factors = local_factors(mix_model, data, field_weights=field_weights)
    terms = list(_field_log_density_features(list(mix_model.components), data))

    field_specs, xs, labels, keep = [], [], [], []
    for f_pos, (factor, (_l, x_f, native_f)) in enumerate(zip(factors, terms)):
        if x_f is None:
            continue
        x_f = np.asarray(x_f, dtype=np.float64)
        if not _is_local_factor(factor):  # coordinate-bearing fields are always local factors
            raise ValueError("internal: coordinate-bearing fields must be local factors.")
        weight = float(factor["weight"])
        dof = float(factor.get("delta_scale", 1.0))
        field_specs.append({"dim": x_f.shape[1], "weight_sqrt": float(np.sqrt(weight / dof))})
        xs.append(x_f)
        keep.append(f_pos)
        kind = "native" if native_f else "typicality"
        labels.extend([f"field{f_pos}[{kind}]:{c}" for c in range(x_f.shape[1])])

    inv_covs = [np.asarray(factors[i]["inv_cov"], dtype=np.float64) for i in keep]
    whiteners = [[_sqrt_psd(inv_covs[f_pos][k]) for f_pos in range(len(keep))] for k in range(k_count)]
    us = [
        np.column_stack([(x @ w) * spec["weight_sqrt"] for x, w, spec in zip(xs, whiteners[k], field_specs)])
        for k in range(k_count)
    ]
    transforms = {"keep": keep, "field_specs": field_specs, "whiteners": whiteners}
    return z, us, labels, transforms


def _resolve_overlap(
    vertices: np.ndarray,
    radii: np.ndarray,
    allowed: set[tuple[int, int]],
    *,
    margin: float = 1.05,
    max_sweeps: int = 200,
) -> np.ndarray:
    """Deterministic push-apart: components with NO measured overlap in the model must not overlap
    on screen. Pairs in ``allowed`` (model overlap present) may overlap visually -- screen overlap
    then MEANS model overlap. Pushes only ever grow separations, so adjacency orderings among the
    allowed pairs are preserved."""
    v = vertices.copy()
    k = v.shape[0]
    for _ in range(int(max_sweeps)):
        moved = False
        for a in range(k):
            for b in range(a + 1, k):
                if (a, b) in allowed:
                    continue
                need = margin * (radii[a] + radii[b])
                diff = v[b] - v[a]
                dist = float(np.linalg.norm(diff))
                if dist >= need:
                    continue
                if dist <= 1.0e-12:  # coincident: split along a deterministic, pair-specific angle
                    angle = 2.0 * np.pi * (a * k + b) / float(k * k)
                    direction = np.zeros(v.shape[1])
                    direction[0], direction[min(1, v.shape[1] - 1)] = np.cos(angle), np.sin(angle)
                else:
                    direction = diff / dist
                push = (need - dist) / 2.0
                v[a] -= push * direction
                v[b] += push * direction
                moved = True
        if not moved:
            break
    return v


@dataclass
class ModelMap:
    """A fitted direct layout: coordinates plus everything needed to read and extend the map.

    ``vertices`` are the component anchors (post occlusion resolution); ``loadings[k]`` names what
    regime ``k``'s chart axes measure (rows = chart features, see ``coord_labels``); ``frames[k]``
    is the chart's on-screen frame (row 0 = the major axis's direction); ``chart_residuals[k]`` is
    the fraction of fiber variance the LINEAR chart leaves beyond ``emb_dim`` (high = this regime's
    within-structure is not 2-D-linear -- consider ``chart='quadratic'`` or ``refine=True``);
    ``place(data)`` maps NEW observations with the fit-time transforms -- closed form, so streaming
    is one call.
    """

    coords: np.ndarray
    vertices: np.ndarray
    responsibilities: np.ndarray
    loadings: list[np.ndarray]
    coord_labels: list[str]
    frames: list[np.ndarray] = field(default_factory=list)
    chart: str = "linear"
    chart_residuals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _model: object = field(repr=False, default=None)
    _transforms: dict = field(repr=False, default_factory=dict)
    _pre: list = field(repr=False, default_factory=list)  # per-k (pre_mu, pre_dirs) or None
    _fiber_means: list[np.ndarray] = field(repr=False, default_factory=list)
    _fiber_scale: float = field(repr=False, default=1.0)
    _emb_dim: int = field(repr=False, default=2)

    def _chart_features(self, u: np.ndarray, k: int) -> np.ndarray:
        if self.chart == "linear":
            return u
        pre = self._pre[k]
        v = u if pre is None else (u - pre[0]) @ pre[1]
        return _lift_quadratic(v)

    def place(self, data) -> np.ndarray:
        """Closed-form out-of-sample placement with the FIT-TIME transforms (means, whiteners,
        chart directions, frames, scale). Placing the training data reproduces ``coords`` exactly."""
        data = list(data)
        z, _ = _posteriors_and_loglikes(self._model, data=data)
        terms = list(_field_log_density_features(list(self._model.components), data))
        keep = self._transforms["keep"]
        if max(keep, default=-1) >= len(terms):
            raise ValueError("data decomposes into a different field set than the fitted map.")
        xs = []
        for f_pos, spec in zip(keep, self._transforms["field_specs"]):
            x_f = terms[f_pos][1]
            if x_f is None or np.asarray(x_f).shape[1] != spec["dim"]:
                raise ValueError("data decomposes into a different field set than the fitted map.")
            xs.append(np.asarray(x_f, dtype=np.float64))
        k_count = self.vertices.shape[0]
        y = np.zeros((len(data), self._emb_dim))
        for k in range(k_count):
            u = np.column_stack(
                [
                    (x @ w) * spec["weight_sqrt"]
                    for x, w, spec in zip(xs, self._transforms["whiteners"][k], self._transforms["field_specs"])
                ]
            )
            feats = self._chart_features(u, k)
            scores = (feats - self._fiber_means[k]) @ self.loadings[k] * self._fiber_scale
            y += z[:, k : k + 1] * (self.vertices[k][None, :] + scores @ self.frames[k])
        return y


def model_map(
    data,
    mix_model=None,
    emb_dim: int = 2,
    *,
    spread: float = 0.35,
    chart: str = "linear",
    occlusion: bool = True,
    occlusion_margin: float = 1.05,
    edge_threshold: float = 0.02,
    field_weights=None,
    max_components: int = 50,
    dpm_max_its: int = 200,
    seed: int | None = None,
    refine: bool = False,
    refine_kwargs: dict | None = None,
) -> ModelMap:
    """The deterministic model-native layout (see module docstring). Returns a :class:`ModelMap`.

    ``spread`` sets how large regime fibers render relative to the smallest inter-vertex gap --
    a LEGIBILITY choice made explicit, unlike t-SNE where cluster sizes are a meaningless artifact.
    ``chart`` is ``'linear'`` (default) or ``'quadratic'`` (explicit degree-2 features -- a curved
    within-regime chart that stays closed-form and placeable); either way ``chart_residuals``
    reports the linear chart's leftover variance per regime. ``occlusion=True`` enforces that
    components with no measured overlap never overlap on screen. ``seed``/``max_components``/
    ``dpm_max_its`` only matter when ``mix_model`` is None and a DPM must be fit first; the layout
    itself uses no randomness. ``refine=True`` polishes local neighborhoods with t-SNE initialized
    FROM this layout (exaggeration off), leaving the global arrangement model-decided.
    """
    if chart not in ("linear", "quadratic"):
        raise ValueError("chart must be 'linear' or 'quadratic'.")
    data = list(data)
    if mix_model is None:
        from mixle.utils.automatic import get_dpm_mixture

        mix_model = get_dpm_mixture(
            data, rng=np.random.RandomState(seed), max_components=max_components, max_its=dpm_max_its, out=None
        )

    z, us, base_labels, transforms = component_fiber_coords(mix_model, data, field_weights=field_weights)
    n, k_count = z.shape
    vertices = component_map(z, emb_dim=emb_dim, edge_threshold=edge_threshold)

    fiber_means, loadings, raw_scores, pre_list, residuals = [], [], [], [], []
    labels = list(base_labels)
    for k in range(k_count):
        u = us[k]
        wk = z[:, k]
        sw = max(float(wk.sum()), 1.0e-12)

        # linear-chart residual report: how much fiber variance emb_dim linear axes leave behind
        mu_u = (wk @ u) / sw
        cov_u = ((u - mu_u) * wk[:, None]).T @ (u - mu_u) / sw
        vals_u = np.sort(np.linalg.eigvalsh(cov_u))[::-1]
        total = float(vals_u.sum())
        residuals.append(0.0 if total <= 0 else float(max(0.0, 1.0 - vals_u[:emb_dim].sum() / total)))

        if chart == "quadratic":
            if u.shape[1] > _MAX_QUAD_BASE:
                vals, vecs = np.linalg.eigh(cov_u)
                order = np.argsort(vals)[::-1][:_MAX_QUAD_BASE]
                pre = (mu_u, _canonical_signs(vecs[:, order]))
                feats = _lift_quadratic((u - pre[0]) @ pre[1])
            else:
                pre = None
                feats = _lift_quadratic(u)
        else:
            pre = None
            feats = u
        pre_list.append(pre)

        mu = (wk @ feats) / sw
        centered = feats - mu
        cov = (centered * wk[:, None]).T @ centered / sw
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:emb_dim]
        p = _canonical_signs(vecs[:, order])
        if p.shape[1] < emb_dim:
            p = np.column_stack([p, np.zeros((p.shape[0], emb_dim - p.shape[1]))])
        fiber_means.append(mu)
        loadings.append(p)
        raw_scores.append(centered @ p)

    if chart == "quadratic":
        if pre_list[0] is None:
            labels = _quad_labels(base_labels)
        else:
            labels = _quad_labels([f"pre_pc{i}" for i in range(_MAX_QUAD_BASE)])

    stds = [
        float(np.sqrt(np.average(np.sum(s**2, axis=1), weights=np.maximum(z[:, k], 1e-12))))
        for k, s in enumerate(raw_scores)
    ]
    gaps = [float(np.linalg.norm(vertices[a] - vertices[b])) for a in range(k_count) for b in range(a + 1, k_count)]
    nonzero_gaps = [g for g in gaps if g > 0]
    ref = min(nonzero_gaps) if nonzero_gaps else 1.0
    med_std = float(np.median([s for s in stds if s > 0]) or 1.0)
    scale = (spread * ref / med_std) if med_std > 0 else 1.0

    if occlusion and k_count > 1:
        dominant = z.argmax(axis=1)
        radii = np.zeros(k_count)
        for k in range(k_count):
            mine = raw_scores[k][dominant == k] * scale
            radii[k] = float(np.percentile(np.linalg.norm(mine, axis=1), 90)) if len(mine) else 0.0
        masses = np.maximum(z.sum(axis=0), 1.0e-12)
        co = z.T @ z
        allowed = {
            (a, b)
            for a in range(k_count)
            for b in range(a + 1, k_count)
            if co[a, b] / min(masses[a], masses[b]) >= edge_threshold
        }
        vertices = _resolve_overlap(vertices, radii, allowed, margin=occlusion_margin)

    # per-chart frames: orient each chart's MAJOR axis tangentially (orthogonal to the nearest
    # other vertex) so neighboring charts' fringes cannot collide head-on. Computed from the FINAL
    # vertices, after occlusion.
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
        frames.append(np.stack([tangent, radial]))  # rows: major axis -> tangent, minor -> radial

    coords = np.zeros((n, emb_dim))
    for k in range(k_count):
        coords += z[:, k : k + 1] * (vertices[k][None, :] + (raw_scores[k] * scale) @ frames[k])

    result = ModelMap(
        coords=coords,
        vertices=vertices,
        responsibilities=z,
        loadings=loadings,
        coord_labels=labels,
        frames=frames,
        chart=chart,
        chart_residuals=np.asarray(residuals),
        _model=mix_model,
        _transforms=transforms,
        _pre=pre_list,
        _fiber_means=fiber_means,
        _fiber_scale=scale,
        _emb_dim=emb_dim,
    )

    if refine:
        import io

        from mixle.utils.hvis.embed import htsne

        kwargs = dict(refine_kwargs or {})
        kwargs.setdefault("max_its", 400)
        kwargs.setdefault("out", io.StringIO())
        y0 = coords.copy()
        y_std = float(y0.std())
        if y_std > 0:  # optimizer-conventional init scale; the arrangement, not the size, is the information
            y0 = y0 * (1.0e-4 / y_std)
        result.coords = np.asarray(
            htsne(data, mix_model=mix_model, emb_dim=emb_dim, Y=y0, seed=seed, **kwargs), dtype=np.float64
        )

    return result
