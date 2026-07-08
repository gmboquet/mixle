"""The direct compositional layout: read the map off the model instead of re-inferring it.

t-SNE/UMAP exist to DISCOVER global and local structure from raw pairwise distances when there is
no model. HViS has a model, and the structure the neighbor optimizers spend a thousand stochastic
iterations inferring is already known in closed form: WHICH regimes exist and how they relate (the
posterior simplex and component overlap geometry), and WHERE each observation sits within its
regime (the whitened local/typicality coordinates). This module composes those two levels directly
-- the (posterior, remainder) decomposition as a layout:

    y_i = sum_k z_ik * (vertex_k + fiber_scores_ik)

* vertices: :func:`~mixle.utils.hvis.affinity.component_map` -- classical MDS on the components'
  overlap geometry (confusable regimes adjacent). Deterministic.
* fibers: per component, a responsibility-weighted PCA of the per-field WHITENED local coordinates
  (native value coordinates or universal typicality coordinates -- the same geometry the 'local'
  affinity scores). Deterministic up to sign, and signs are canonicalized. The loadings are
  returned, so a fiber axis is NAMEABLE: "within regime k, axis 1 is field 2's coordinate 0".
* composition: barycentric in the posterior, so sharp points sit in their regime's local chart and
  mixed-membership points interpolate between charts.

No perplexity, no seed, no optimizer failure modes; out-of-sample placement (:meth:`ModelMap.place`)
is the same closed form, so streaming costs one matrix product. The precedent is Bishop & Tipping's
hierarchical mixture visualization / GTM: per-component local projections composed by
responsibility -- rebuilt here on HViS's field decomposition so it covers heterogeneous,
variable-length, and sequence data.

When to still reach for the neighbor optimizers: no trustworthy model, strongly nonlinear
within-regime manifolds a linear fiber chart flattens, or pure exploration. ``refine=True`` runs
t-SNE FROM this layout (informative init, exaggeration off) so the optimizer only polishes local
neighborhoods it is actually good at -- the composition stays in charge of the global picture.
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


def _sqrt_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(mat, dtype=np.float64))
    return vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0))) @ vecs.T


def _canonical_signs(components: np.ndarray) -> np.ndarray:
    """Fix each principal direction's sign so its largest-magnitude loading is positive --
    determinism must not depend on LAPACK's arbitrary eigenvector signs."""
    flips = np.sign(components[np.abs(components).argmax(axis=0), np.arange(components.shape[1])])
    flips[flips == 0] = 1.0
    return components * flips


@dataclass
class ModelMap:
    """A fitted direct layout: coordinates plus everything needed to read and extend the map.

    ``vertices`` are the component anchors; ``loadings[k]`` names what regime ``k``'s fiber axes
    measure (rows = concatenated whitened field coordinates, see ``coord_labels``); ``place(data)``
    maps NEW observations with the fit-time transforms -- closed form, so streaming is one call.
    """

    coords: np.ndarray
    vertices: np.ndarray
    responsibilities: np.ndarray
    loadings: list[np.ndarray]
    coord_labels: list[str]
    _model: object = field(repr=False, default=None)
    _keep_fields: list[int] = field(repr=False, default_factory=list)
    _field_specs: list[dict] = field(repr=False, default_factory=list)
    _fiber_means: list[np.ndarray] = field(repr=False, default_factory=list)
    _fiber_whiteners: list[list[np.ndarray]] = field(repr=False, default_factory=list)
    _fiber_scale: float = field(repr=False, default=1.0)
    _emb_dim: int = field(repr=False, default=2)

    def place(self, data) -> np.ndarray:
        """Closed-form out-of-sample placement with the FIT-TIME transforms (means, whiteners,
        fiber directions, scale). Placing the training data reproduces ``coords`` exactly."""
        data = list(data)
        z, _ = _posteriors_and_loglikes(self._model, data=data)
        terms = list(_field_log_density_features(list(self._model.components), data))
        if max(self._keep_fields, default=-1) >= len(terms):
            raise ValueError("data decomposes into a different field set than the fitted map.")
        xs = []
        for f_pos in self._keep_fields:
            x_f = terms[f_pos][1]
            if x_f is None:
                raise ValueError("data decomposes into a different field set than the fitted map.")
            xs.append(np.asarray(x_f, dtype=np.float64))
        k_count = self.vertices.shape[0]
        y = np.zeros((len(data), self._emb_dim))
        for k in range(k_count):
            u = self._whitened_block(xs, k)
            scores = (u - self._fiber_means[k]) @ self.loadings[k] * self._fiber_scale
            y += z[:, k : k + 1] * (self.vertices[k][None, :] + scores)
        return y

    def _whitened_block(self, xs: list[np.ndarray], k: int) -> np.ndarray:
        parts = []
        for x_f, spec, whitener in zip(xs, self._field_specs, self._fiber_whiteners[k]):
            if x_f is None or x_f.shape[1] != spec["dim"]:
                raise ValueError("field coordinates do not match the fitted map's field shapes.")
            parts.append((x_f @ whitener) * spec["weight_sqrt"])
        return np.column_stack(parts)


def model_map(
    data,
    mix_model=None,
    emb_dim: int = 2,
    *,
    spread: float = 0.35,
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
    ``seed``/``max_components``/``dpm_max_its`` only matter when ``mix_model`` is None and a DPM
    must be fit first; the layout itself uses no randomness. ``refine=True`` polishes local
    neighborhoods with t-SNE initialized FROM this layout (exaggeration off), leaving the global
    arrangement model-decided; the refined coordinates replace ``coords`` (the skeleton stays
    available as ``vertices``).
    """
    data = list(data)
    if mix_model is None:
        from mixle.utils.automatic import get_dpm_mixture

        mix_model = get_dpm_mixture(
            data, rng=np.random.RandomState(seed), max_components=max_components, max_its=dpm_max_its, out=None
        )

    z, _ = _posteriors_and_loglikes(mix_model, data=data)
    n, k_count = z.shape
    vertices = component_map(z, emb_dim=emb_dim)

    factors = local_factors(mix_model, data, field_weights=field_weights)
    terms = list(_field_log_density_features(list(mix_model.components), data))
    field_specs, xs, labels = [], [], []
    for f_pos, (factor, (_l, x_f, native_f)) in enumerate(zip(factors, terms)):
        if x_f is None:
            field_specs.append(None)
            xs.append(None)
            continue
        x_f = np.asarray(x_f, dtype=np.float64)
        weight = float(factor["weight"]) if _is_local_factor(factor) else 1.0
        dof = float(factor.get("delta_scale", 1.0)) if _is_local_factor(factor) else 1.0
        field_specs.append({"dim": x_f.shape[1], "weight_sqrt": np.sqrt(weight / dof)})
        xs.append(x_f)
        kind = "native" if native_f else "typicality"
        labels.extend([f"field{f_pos}[{kind}]:{c}" for c in range(x_f.shape[1])])
    keep = [i for i, spec in enumerate(field_specs) if spec is not None]
    field_specs = [field_specs[i] for i in keep]
    xs = [xs[i] for i in keep]
    inv_covs = [np.asarray(factors[i]["inv_cov"], dtype=np.float64) for i in keep if _is_local_factor(factors[i])]
    if len(inv_covs) != len(keep):  # a posterior-only tuple factor slipped through with coords: not expected
        raise ValueError("internal: coordinate-bearing fields must be local factors.")

    fiber_means, fiber_whiteners, loadings, raw_scores = [], [], [], []
    for k in range(k_count):
        whiteners = [_sqrt_psd(inv_covs[f_pos][k]) for f_pos in range(len(keep))]
        u = np.column_stack([(x @ w) * spec["weight_sqrt"] for x, w, spec in zip(xs, whiteners, field_specs)])
        wk = z[:, k]
        sw = float(wk.sum())
        mu = (wk @ u) / sw if sw > 0 else u.mean(axis=0)
        centered = u - mu
        cov = (centered * wk[:, None]).T @ centered / max(sw, 1.0e-12)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:emb_dim]
        p = _canonical_signs(vecs[:, order])
        if p.shape[1] < emb_dim:
            p = np.column_stack([p, np.zeros((p.shape[0], emb_dim - p.shape[1]))])
        fiber_means.append(mu)
        fiber_whiteners.append(whiteners)
        loadings.append(p)
        raw_scores.append(centered @ p)

    stds = [
        float(np.sqrt(np.average(np.sum(s**2, axis=1), weights=np.maximum(z[:, k], 1e-12))))
        for k, s in enumerate(raw_scores)
    ]
    gaps = [float(np.linalg.norm(vertices[a] - vertices[b])) for a in range(k_count) for b in range(a + 1, k_count)]
    nonzero_gaps = [g for g in gaps if g > 0]
    ref = min(nonzero_gaps) if nonzero_gaps else 1.0
    med_std = float(np.median([s for s in stds if s > 0]) or 1.0)
    scale = (spread * ref / med_std) if med_std > 0 else 1.0

    coords = np.zeros((n, emb_dim))
    for k in range(k_count):
        coords += z[:, k : k + 1] * (vertices[k][None, :] + raw_scores[k] * scale)

    result = ModelMap(
        coords=coords,
        vertices=vertices,
        responsibilities=z,
        loadings=loadings,
        coord_labels=labels,
        _model=mix_model,
        _keep_fields=keep,
        _field_specs=field_specs,
        _fiber_means=fiber_means,
        _fiber_whiteners=fiber_whiteners,
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
