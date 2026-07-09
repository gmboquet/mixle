"""Pool-based DOE for task distillation and cross-modal training.

Distillation turns teacher calls, human labels, or paired cross-modal records into an expensive
experiment budget. This module chooses *which* candidate examples to spend that budget on. It is
model-agnostic: callers provide embeddings/features, optional uncertainty or preference scores, task
tags, modality tags, and label costs; DOE returns indices that balance informativeness, diversity,
multi-task coverage, cross-modal coverage, and cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import RandomState

from mixle.doe.designs import _as_rng


@dataclass(frozen=True)
class DistillationDesign:
    """Selected pool indices and diagnostics for a distillation design.

    ``indices`` point into the exact candidate pool supplied to the selector.
    ``scores`` are the sequential merits assigned to the chosen candidates.
    ``candidate_scores`` preserves the base uncertainty/preference score before
    diversity, coverage, and cost terms are applied. ``metadata`` records the
    target coverage and weights needed to audit or reproduce the design.
    """

    indices: np.ndarray
    scores: np.ndarray
    task_counts: dict[Any, int] = field(default_factory=dict)
    modality_counts: dict[Any, int] = field(default_factory=dict)
    candidate_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    metadata: dict[str, Any] = field(default_factory=dict)


def distillation_design(
    features: Any,
    n: int,
    *,
    task_labels: Sequence[Any] | Sequence[Sequence[Any]] | np.ndarray | None = None,
    modalities: Sequence[Any] | Sequence[Sequence[Any]] | np.ndarray | None = None,
    uncertainty: Any | None = None,
    preference: Any | None = None,
    cost: Any | None = None,
    task_weights: Mapping[Any, float] | Sequence[float] | None = None,
    modality_weights: Mapping[Any, float] | Sequence[float] | None = None,
    reference_features: Any | None = None,
    eligible: Any | None = None,
    uncertainty_weight: float = 1.0,
    diversity_weight: float = 1.0,
    task_coverage_weight: float = 1.0,
    modality_coverage_weight: float = 1.0,
    preference_weight: float = 1.0,
    cost_weight: float = 1.0,
    seed: int | RandomState | None = None,
) -> DistillationDesign:
    """Select a pool subset for teacher labeling or student distillation.

    ``features`` is an ``(N, d)`` embedding/feature matrix for candidate examples. ``task_labels`` and
    ``modalities`` may be one tag per row or a sequence of tags per row. ``uncertainty`` may be ``(N,)``
    or ``(N, T)``; the latter is averaged across the task tags present on each candidate. Higher
    uncertainty/preference and better coverage increase merit; higher ``cost`` lowers it.
    """
    if int(n) <= 0:
        raise ValueError("n must be positive.")
    x = _as_2d_features(features, "features")
    n_pool = x.shape[0]
    if n_pool == 0:
        raise ValueError("features must contain at least one candidate.")

    rng = _as_rng(seed)
    task_inc, task_names = _incidence(task_labels, n_pool, "task_labels")
    mod_inc, mod_names = _incidence(modalities, n_pool, "modalities")
    task_target = _targets(task_names, task_weights, int(n))
    mod_target = _targets(mod_names, modality_weights, int(n))
    eligible_mask = _eligible_mask(eligible, n_pool)
    if int(eligible_mask.sum()) < int(n):
        raise ValueError("not enough eligible candidates for the requested design size.")

    unc = _unit_scale(_aggregate_score(uncertainty, n_pool, task_inc))
    pref = _unit_scale(_as_score(preference, n_pool, default=0.0, name="preference"))
    cost_vec = _as_score(cost, n_pool, default=1.0, name="cost")
    if np.any(cost_vec <= 0.0):
        raise ValueError("cost entries must be positive.")
    cost_scale = cost_vec / max(float(np.mean(cost_vec[eligible_mask])), 1e-12)
    candidate_scores = uncertainty_weight * unc + preference_weight * pref

    z, z_ref = _standardized_feature_space(x, reference_features)
    chosen: list[int] = []
    chosen_scores: list[float] = []
    task_counts = np.zeros(len(task_names), dtype=np.float64)
    mod_counts = np.zeros(len(mod_names), dtype=np.float64)

    for _ in range(int(n)):
        available = eligible_mask.copy()
        if chosen:
            available[np.asarray(chosen, dtype=int)] = False
        diversity = _diversity_scores(z, chosen, z_ref)
        task_bonus = _coverage_bonus(task_inc, task_counts, task_target)
        mod_bonus = _coverage_bonus(mod_inc, mod_counts, mod_target)
        merit = (
            candidate_scores
            + diversity_weight * diversity
            + task_coverage_weight * task_bonus
            + modality_coverage_weight * mod_bonus
        )
        if cost_weight:
            merit = merit / np.power(np.maximum(cost_scale, 1e-12), float(cost_weight))
        merit = np.asarray(merit, dtype=np.float64)
        merit[~available] = -np.inf
        merit += rng.uniform(0.0, 1e-12, size=n_pool)
        pick = int(np.argmax(merit))
        if not np.isfinite(merit[pick]):
            raise ValueError("no finite-merit eligible candidates remain.")
        chosen.append(pick)
        chosen_scores.append(float(merit[pick]))
        if task_inc.shape[1]:
            task_counts += task_inc[pick]
        if mod_inc.shape[1]:
            mod_counts += mod_inc[pick]

    indices = np.asarray(chosen, dtype=np.int64)
    return DistillationDesign(
        indices=indices,
        scores=np.asarray(chosen_scores, dtype=np.float64),
        task_counts=_count_map(task_names, task_counts),
        modality_counts=_count_map(mod_names, mod_counts),
        candidate_scores=np.asarray(candidate_scores, dtype=np.float64),
        metadata={
            "task_targets": _count_map(task_names, task_target),
            "modality_targets": _count_map(mod_names, mod_target),
            "eligible": int(eligible_mask.sum()),
            "weights": {
                "uncertainty": float(uncertainty_weight),
                "diversity": float(diversity_weight),
                "task_coverage": float(task_coverage_weight),
                "modality_coverage": float(modality_coverage_weight),
                "preference": float(preference_weight),
                "cost": float(cost_weight),
            },
        },
    )


def multitask_distillation_design(*args: Any, **kwargs: Any) -> DistillationDesign:
    """Alias for :func:`distillation_design` that reads naturally at multi-task call sites."""
    return distillation_design(*args, **kwargs)


def cross_modal_distillation_design(
    modality_features: Mapping[Any, Any],
    n: int,
    *,
    task_labels: Sequence[Any] | Sequence[Sequence[Any]] | np.ndarray | None = None,
    uncertainty: Any | None = None,
    preference: Any | None = None,
    cost: Any | None = None,
    required_modalities: Sequence[Any] | None = None,
    min_modalities: int = 2,
    alignment_weight: float = 1.0,
    seed: int | RandomState | None = None,
    **kwargs: Any,
) -> DistillationDesign:
    """Select paired cross-modal records for distillation/alignment training.

    ``modality_features`` maps modality name to an ``(N, d_m)`` feature matrix. Rows with non-finite
    values are treated as missing for that modality. The selector fuses standardized modality features,
    tags each row by available modalities, and adds an alignment-disagreement preference for modality
    pairs that share the same embedding width. By default only rows with at least two modalities are
    eligible.
    """
    if not modality_features:
        raise ValueError("modality_features must contain at least one modality.")
    names = list(modality_features)
    arrays = [_as_2d_features(modality_features[name], f"modality_features[{name!r}]") for name in names]
    n_pool = arrays[0].shape[0]
    if any(arr.shape[0] != n_pool for arr in arrays):
        raise ValueError("all modality feature matrices must have the same number of rows.")
    if int(min_modalities) <= 0:
        raise ValueError("min_modalities must be positive.")

    z_by_mod: list[np.ndarray] = []
    present_cols: list[np.ndarray] = []
    fused_parts: list[np.ndarray] = []
    for arr in arrays:
        present = np.all(np.isfinite(arr), axis=1)
        z = _standardize_with_missing(arr, present)
        z_by_mod.append(z)
        present_cols.append(present)
        fused_parts.append(np.where(present[:, None], z, 0.0))
    presence = np.vstack(present_cols).T
    fused = np.hstack([*fused_parts, presence.astype(np.float64)])

    row_modalities = [tuple(name for name, ok in zip(names, presence[i]) if ok) for i in range(n_pool)]
    eligible = presence.sum(axis=1) >= int(min_modalities)
    if required_modalities is not None:
        required = set(required_modalities)
        missing = required.difference(names)
        if missing:
            raise ValueError("required modalities are not present: %s" % sorted(missing))
        eligible &= np.array([required.issubset(row) for row in row_modalities], dtype=bool)

    alignment = _alignment_disagreement(z_by_mod, presence)
    base_unc = _as_score(uncertainty, n_pool, default=0.0, name="uncertainty")
    combined_unc = _unit_scale(base_unc) + float(alignment_weight) * _unit_scale(alignment)
    combined_pref = _unit_scale(_as_score(preference, n_pool, default=0.0, name="preference"))

    return distillation_design(
        fused,
        n,
        task_labels=task_labels,
        modalities=row_modalities,
        uncertainty=combined_unc,
        preference=combined_pref,
        cost=cost,
        eligible=eligible,
        seed=seed,
        **kwargs,
    )


def _as_2d_features(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D numeric array.")
    return arr


def _eligible_mask(eligible: Any | None, n: int) -> np.ndarray:
    if eligible is None:
        return np.ones(n, dtype=bool)
    arr = np.asarray(eligible)
    if arr.dtype == bool:
        if arr.shape != (n,):
            raise ValueError("eligible mask length must match features.")
        return arr.astype(bool, copy=True)
    mask = np.zeros(n, dtype=bool)
    idx = np.asarray(arr, dtype=int).ravel()
    if np.any((idx < 0) | (idx >= n)):
        raise ValueError("eligible indices are out of range.")
    mask[idx] = True
    return mask


def _incidence(labels: Any | None, n: int, name: str) -> tuple[np.ndarray, list[Any]]:
    if labels is None:
        return np.zeros((n, 0), dtype=np.float64), []
    try:
        numeric = np.asarray(labels)
    except ValueError:
        numeric = np.asarray([], dtype=object)
    if numeric.ndim == 2 and (np.issubdtype(numeric.dtype, np.number) or np.issubdtype(numeric.dtype, np.bool_)):
        inc = np.asarray(numeric, dtype=np.float64) > 0.0
        if inc.shape[0] != n:
            raise ValueError(f"{name} row count must match features.")
        return inc.astype(np.float64), list(range(inc.shape[1]))
    arr = np.asarray(labels, dtype=object)
    if len(labels) != n:  # type: ignore[arg-type]
        raise ValueError(f"{name} length must match features.")
    rows = [_as_tag_tuple(row) for row in labels]
    names: list[Any] = []
    seen: set[Any] = set()
    for row in rows:
        for tag in row:
            if tag not in seen:
                seen.add(tag)
                names.append(tag)
    inc = np.zeros((n, len(names)), dtype=np.float64)
    pos = {tag: i for i, tag in enumerate(names)}
    for i, row in enumerate(rows):
        for tag in row:
            inc[i, pos[tag]] = 1.0
    return inc, names


def _as_tag_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        iter(value)
    except TypeError:
        return (value,)
    return tuple(value)


def _targets(names: Sequence[Any], weights: Mapping[Any, float] | Sequence[float] | None, n: int) -> np.ndarray:
    if not names:
        return np.zeros(0, dtype=np.float64)
    if weights is None:
        w = np.ones(len(names), dtype=np.float64)
    elif isinstance(weights, Mapping):
        w = np.asarray([weights.get(name, 0.0) for name in names], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(names),):
            raise ValueError("coverage weights must have one entry per discovered tag.")
    if np.any(~np.isfinite(w)) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
        raise ValueError("coverage weights must be finite, non-negative, and contain positive mass.")
    return n * w / float(w.sum())


def _as_score(value: Any | None, n: int, *, default: float, name: str) -> np.ndarray:
    if value is None:
        return np.full(n, float(default), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},).")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} entries must be finite.")
    return arr


def _aggregate_score(value: Any | None, n: int, task_inc: np.ndarray) -> np.ndarray:
    if value is None:
        return np.zeros(n, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return _as_score(arr, n, default=0.0, name="uncertainty")
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError("uncertainty must have shape (N,) or (N, T).")
    if np.any(~np.isfinite(arr)):
        raise ValueError("uncertainty entries must be finite.")
    if task_inc.shape[1] and arr.shape[1] == task_inc.shape[1]:
        denom = np.maximum(task_inc.sum(axis=1), 1.0)
        return np.sum(arr * task_inc, axis=1) / denom
    return np.mean(arr, axis=1)


def _unit_scale(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("scores must be finite.")
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def _standardized_feature_space(x: np.ndarray, reference_features: Any | None) -> tuple[np.ndarray, np.ndarray]:
    ref = np.empty((0, x.shape[1]), dtype=np.float64)
    if reference_features is not None:
        ref = _as_2d_features(reference_features, "reference_features")
        if ref.shape[1] != x.shape[1]:
            raise ValueError("reference_features must have the same number of columns as features.")
    both = np.vstack([x, ref])
    both = _fill_nonfinite_with_column_mean(both)
    mean = both.mean(axis=0, keepdims=True)
    scale = both.std(axis=0, keepdims=True)
    scale[scale <= 1e-12] = 1.0
    z = (both - mean) / scale
    return z[: x.shape[0]], z[x.shape[0] :]


def _fill_nonfinite_with_column_mean(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(arr)
    counts = finite.sum(axis=0)
    sums = np.where(finite, arr, 0.0).sum(axis=0)
    means = np.divide(sums, np.maximum(counts, 1), out=np.zeros(arr.shape[1], dtype=np.float64), where=counts > 0)
    bad = ~finite
    if np.any(bad):
        arr[bad] = means[np.nonzero(bad)[1]]
    return arr


def _standardize_with_missing(x: np.ndarray, present: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).copy()
    if np.any(present):
        ref = arr[present]
        mean = ref.mean(axis=0, keepdims=True)
        scale = ref.std(axis=0, keepdims=True)
        scale[scale <= 1e-12] = 1.0
    else:
        mean = np.zeros((1, arr.shape[1]), dtype=np.float64)
        scale = np.ones((1, arr.shape[1]), dtype=np.float64)
    arr[~np.isfinite(arr)] = 0.0
    return (arr - mean) / scale


def _diversity_scores(z: np.ndarray, chosen: Sequence[int], z_ref: np.ndarray) -> np.ndarray:
    if chosen or z_ref.shape[0]:
        refs = z_ref if not chosen else np.vstack([z_ref, z[np.asarray(chosen, dtype=int)]])
        diff = z[:, None, :] - refs[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    else:
        center = z.mean(axis=0, keepdims=True)
        dist = np.sqrt(np.sum((z - center) ** 2, axis=1))
    return _unit_scale(dist)


def _coverage_bonus(inc: np.ndarray, counts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if inc.shape[1] == 0:
        return np.zeros(inc.shape[0], dtype=np.float64)
    deficits = np.maximum(targets - counts, 0.0) / np.maximum(targets, 1e-12)
    if float(deficits.sum()) <= 0.0:
        return np.zeros(inc.shape[0], dtype=np.float64)
    return _unit_scale(inc @ deficits)


def _alignment_disagreement(z_by_mod: Sequence[np.ndarray], presence: np.ndarray) -> np.ndarray:
    n = presence.shape[0]
    gaps = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(len(z_by_mod)):
        for j in range(i + 1, len(z_by_mod)):
            if z_by_mod[i].shape[1] != z_by_mod[j].shape[1]:
                continue
            mask = presence[:, i] & presence[:, j]
            if not np.any(mask):
                continue
            gap = np.sqrt(np.sum((z_by_mod[i][mask] - z_by_mod[j][mask]) ** 2, axis=1))
            gaps[mask] += gap
            counts[mask] += 1.0
    return np.divide(gaps, np.maximum(counts, 1.0), out=np.zeros(n, dtype=np.float64), where=counts > 0)


def _count_map(names: Sequence[Any], values: np.ndarray) -> dict[Any, int | float]:
    out: dict[Any, int | float] = {}
    for name, value in zip(names, values):
        v = float(value)
        out[name] = int(round(v)) if abs(v - round(v)) < 1e-12 else v
    return out


__all__ = [
    "DistillationDesign",
    "distillation_design",
    "multitask_distillation_design",
    "cross_modal_distillation_design",
]
