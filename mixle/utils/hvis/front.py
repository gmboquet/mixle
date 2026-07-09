"""The one front door (design review R1): call one thing, get coordinates + receipts.

``hvis.map(data)`` runs the whole pipeline the package's pieces add up to: fit or accept a model,
compose the direct layout (:func:`~mixle.utils.hvis.direct.model_map` -- deterministic, occlusion-
resolved, optionally refined by t-SNE from the model-decided arrangement), and attach every receipt
so "make it make sense" is a property of the RETURNED OBJECT, not of the user's diligence:

* per-point uncertainty channels a plot can encode directly: ``posterior_entropy`` (mixed
  membership) and ``typicality`` (log-density percentile; low = the model finds this point odd);
* ``nerve`` + ``nerve_health`` -- the learned cover's topology (loops, disconnection);
* ``fit_health`` -- does the MODEL describe the DATA (merged/shattered regimes, fiber calibration);
* ``render_health`` -- does the MAP describe the MODEL (trustworthiness/continuity);
* ``zoom(components)`` -- the hierarchy: re-chart a regime group with its own fibers, aligned back
  to the parent (residual reported, never silently rotated);
* ``summary()`` -- every diagnosis in one plain-text block.

``htsne``/``humap`` remain the optimizer-flavored escape hatches; this is the default reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.utils.hvis.direct import ModelMap, model_map
from mixle.utils.hvis.topology import component_tree, embedding_health, fuzzy_nerve, model_fit_health, nerve_report

__all__ = ["Map", "hvis_map"]


@dataclass
class Map:
    """A finished map: coordinates, anchors, per-point uncertainty, and every receipt."""

    base: ModelMap
    posterior_entropy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    typicality: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nerve: dict = field(default_factory=dict)
    nerve_health: dict = field(default_factory=dict)
    fit_health: dict = field(default_factory=dict)
    render_health: dict = field(default_factory=dict)
    merge_tree: list = field(default_factory=list)
    zoom_alignment_rms: float | None = None
    _data: list = field(repr=False, default_factory=list)
    _params: dict = field(repr=False, default_factory=dict)

    # -- the readable surface (delegating to the underlying layout) --------------------------------

    @property
    def coords(self) -> np.ndarray:
        return self.base.coords

    @property
    def vertices(self) -> np.ndarray:
        return self.base.vertices

    @property
    def responsibilities(self) -> np.ndarray:
        return self.base.responsibilities

    @property
    def model(self) -> Any:
        return self.base._model  # noqa: SLF001 - the Map owns its layout

    def place(self, data) -> np.ndarray:
        return self.base.place(data)

    @property
    def diagnosis(self) -> list[str]:
        """Every receipt's findings, one flat list -- empty means no receipt has a complaint."""
        out: list[str] = []
        for report in (self.nerve_health, self.fit_health, self.render_health):
            out.extend(report.get("diagnosis", []))
        return out

    def summary(self) -> str:
        z = self.responsibilities
        lines = [
            f"map: {len(self._data)} observations, {z.shape[1]} components, chart={self.base.chart}",
            f"render: trustworthiness={self.render_health.get('trustworthiness', float('nan')):.2f} "
            f"continuity={self.render_health.get('continuity', float('nan')):.2f}",
            f"topology: {self.nerve_health.get('n_components', '?')} piece(s), "
            f"{len(self.nerve_health.get('holes', []))} hole(s)",
        ]
        if self.zoom_alignment_rms is not None:
            lines.append(f"zoom alignment rms vs parent: {self.zoom_alignment_rms:.3f}")
        findings = self.diagnosis
        lines.append("findings: none" if not findings else "findings:")
        lines.extend(f"  - {d}" for d in findings)
        return "\n".join(lines)

    # -- hierarchy ----------------------------------------------------------------------------------

    def zoom(self, components: list[int]) -> Map:
        """Re-chart one regime group with its own fibers: the sub-mixture over ``components`` maps
        the points they dominate, then the child layout is rigidly aligned (rotation/translation +
        uniform scale) onto those points' PARENT positions -- continuity is measured
        (``zoom_alignment_rms``), never assumed. Component indices in the child are positional
        within ``components``."""
        from mixle.stats import MixtureDistribution
        from mixle.utils.hvis.stream import _procrustes_align

        components = sorted(set(int(c) for c in components))
        if len(components) < 1:
            raise ValueError("zoom needs at least one component index.")
        z = self.responsibilities
        dominant = z.argmax(axis=1)
        member_idx = [i for i in range(len(self._data)) if int(dominant[i]) in components]
        if len(member_idx) < 5:
            raise ValueError(f"components {components} dominate only {len(member_idx)} points -- nothing to zoom into.")

        comps = list(self.model.components)
        w = np.asarray(self.model.w, dtype=np.float64)[components]
        sub_model = MixtureDistribution([comps[c] for c in components], w / w.sum())
        sub_data = [self._data[i] for i in member_idx]

        child = hvis_map(sub_data, sub_model, **self._params)
        if len(components) > 1 and len(member_idx) >= 3:
            aligned, rms, _scale = _procrustes_align(child.base.coords, self.coords[member_idx])
            child.base.coords = aligned
            child.zoom_alignment_rms = rms
        return child


def hvis_map(
    data,
    mix_model=None,
    emb_dim: int = 2,
    *,
    spread: float = 0.35,
    chart: str = "linear",
    occlusion: bool = True,
    refine: bool = False,
    goals=None,
    health: bool = True,
    holdout=None,
    field_weights=None,
    max_components: int = 50,
    dpm_max_its: int = 200,
    seed: int | None = None,
    refine_kwargs: dict | None = None,
) -> Map:
    """One call, one finished map (see module docstring). Deterministic unless a DPM must be fit.

    ``goals`` (anchoring / partial labels / axis objectives) require the optimizer pass, so passing
    them implies ``refine=True``. ``health=False`` skips the receipt computations (they are cheap
    and subsampled; skip only in tight loops).
    """
    data = list(data)
    kwargs = dict(refine_kwargs or {})
    if goals:
        refine = True
        kwargs.setdefault("goals", goals)

    base = model_map(
        data,
        mix_model=mix_model,
        emb_dim=emb_dim,
        spread=spread,
        chart=chart,
        occlusion=occlusion,
        field_weights=field_weights,
        max_components=max_components,
        dpm_max_its=dpm_max_its,
        seed=seed,
        refine=refine,
        refine_kwargs=kwargs if refine else None,
    )
    model = base._model  # noqa: SLF001 - the front door owns the layout it just built

    z = base.responsibilities
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(z > 0, z * np.log(z), 0.0), axis=1)

    if hasattr(model, "dist_to_encoder") and hasattr(model, "seq_log_density"):
        enc = model.dist_to_encoder().seq_encode(data)
        ll = np.asarray(model.seq_log_density(enc), dtype=np.float64)
    else:
        from mixle.utils.hvis.affinity import _posteriors_and_loglikes

        _, ll_mat = _posteriors_and_loglikes(model, data=data)
        log_w = np.asarray(model.log_w, dtype=np.float64).reshape(1, -1)
        joint = ll_mat + log_w
        mx = joint.max(axis=1, keepdims=True)
        ll = mx[:, 0] + np.log(np.exp(joint - mx).sum(axis=1))
    typicality = np.argsort(np.argsort(ll)).astype(np.float64) / max(len(ll) - 1, 1)  # percentile: low = odd

    nerve = fuzzy_nerve(z)
    result = Map(
        base=base,
        posterior_entropy=entropy,
        typicality=typicality,
        nerve=nerve,
        nerve_health=nerve_report(nerve),
        merge_tree=component_tree(nerve),
        _data=data,
        _params={
            "emb_dim": emb_dim,
            "spread": spread,
            "chart": chart,
            "occlusion": occlusion,
            "refine": refine,
            "health": health,
            "seed": seed,
        },
    )
    if health:
        result.fit_health = model_fit_health(model, data, holdout=holdout, field_weights=field_weights)
        result.render_health = embedding_health(base.coords, model, data, field_weights=field_weights)
    return result
