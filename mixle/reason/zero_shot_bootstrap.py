"""Zero-shot modality bootstrap (workstream L3): a brand-new data type joins a cross-modal joint
with NO retraining of anything already fitted.

Three pieces, in dependency order:

1. :func:`induce_leaf_for_unseen_type` extends the automatic profiler (``mixle.utils.automatic``,
   see B-series work) with a real fallback chain for a genuinely unrecognized data type: try the
   profiler's own classical-family induction first (it already covers numeric/categorical/
   embedding/image/sequence shapes); if that abstains (``IgnoredDistribution``), fall back to
   classical (Gaussian/MVN) for simple numeric payloads, a generic GradLeaf-wrapped neural density
   for complex/high-dimensional numeric payloads, or a graph model / sequence model when the data
   carries that structure.

2. :func:`resonance_embedding` scores a brand-new-modality sample against an existing, already-
   fitted model zoo -- one generic "typicality" coordinate per zoo model, read off each model's own
   capability surface (:mod:`mixle.capability`'s ``HasCDF``/``HasMoments``): where does a generic
   scalar reduction of the new sample fall relative to THIS model's own typical range? Evaluation
   only -- no gradient steps, no fitting -- hence "zero training."

3. :func:`resonance_adequacy_gate` reuses the separation statistic behind
   :func:`mixle.utils.hvis.topology.model_fit_health`'s merged-regime detector (a deterministic
   2-means-style projected separation ratio, thresholded at the same measured finite-sample
   correction ``2.65 + 6/sqrt(n)``) to decide: is the resonance embedding's class separation good
   enough to use as a lightweight, training-free proxy representation, or should the modality
   GRADUATE to a real native leaf (piece 1)?

:func:`add_modality_to_joint` plugs a new per-regime leaf (a native induced leaf, or a lightweight
fit over resonance coordinates) into an existing :class:`~mixle.reason.cross_modal.CrossModalJoint`
by rebuilding each regime's :class:`~mixle.stats.combinator.composite.CompositeDistribution` with
the OLD per-modality distributions reused BY REFERENCE (never refit, never copied) plus the one new
field -- so every other modality's fitted parameters are bitwise identical before and after.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from mixle.capability import HasCDF, HasMoments, supports
from mixle.reason.cross_modal import CrossModalJoint
from mixle.stats.combinator.composite import CompositeDistribution
from mixle.stats.compute.pdist import SequenceEncodableProbabilityDistribution
from mixle.stats.latent.mixture import MixtureDistribution

__all__ = [
    "induce_leaf_for_unseen_type",
    "resonance_embedding",
    "resonance_adequacy_gate",
    "add_modality_to_joint",
    "fit_resonance_leaves",
]


# ---------------------------------------------------------------------------------------------
# 1. Automatic-profiler extension: classical -> neural -> sequence/graph fallback chain.
# ---------------------------------------------------------------------------------------------

_EMBEDDING_MIN_DIM = 16


def induce_leaf_for_unseen_type(
    samples: Sequence[Any],
    *,
    rng: np.random.RandomState | None = None,
    max_its: int = 30,
) -> SequenceEncodableProbabilityDistribution:
    """Induce and FIT a reasonable leaf distribution for ``samples`` of a data type the caller has
    never modeled before.

    Tries, in order: (a) the existing automatic profiler's own structural induction (numeric,
    categorical, embedding, image, sequence -- whatever it already recognizes); (b) for data the
    profiler abstains on (``IgnoredDistribution``: e.g. arbitrary Python objects with no scalar or
    container structure it parses), a classical family (Gaussian / multivariate Gaussian) when a
    numeric feature vector can be extracted and is low/moderate-dimensional; (c) a generic
    GradLeaf-wrapped neural density when the extracted numeric payload is high-dimensional; (d) a
    graph model when samples carry adjacency/graph structure; (e) a sequence model when samples are
    variable-length collections of otherwise-inducible elements.
    """
    rows = list(samples)
    if len(rows) == 0:
        raise ValueError("induce_leaf_for_unseen_type requires at least one sample")

    leaf = _try_automatic_profiler(rows, rng=rng, max_its=max_its)
    if leaf is not None:
        return leaf

    numeric = _coerce_numeric_matrix(rows)
    if numeric is not None:
        return _fit_numeric_fallback(numeric, rng=rng, max_its=max_its)

    graph_rows = _coerce_graph_rows(rows)
    if graph_rows is not None:
        return _fit_graph_fallback(graph_rows, max_its=max_its)

    sequence_elems = _coerce_sequence_elements(rows)
    if sequence_elems is not None:
        return _fit_sequence_fallback(rows, sequence_elems, rng=rng, max_its=max_its)

    raise TypeError(
        "induce_leaf_for_unseen_type: samples are neither profiler-recognized, numeric-coercible, "
        "graph-structured, nor sequence-structured -- no fallback family applies"
    )


def _try_automatic_profiler(rows, *, rng, max_its):
    from mixle.inference.estimation import optimize
    from mixle.stats.combinator.ignored import IgnoredDistribution
    from mixle.utils.automatic.profiling import get_prototype

    try:
        prototype = get_prototype(rows, seed=0)
        fitted = optimize(rows, prototype, max_its=max_its, rng=rng)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(fitted, IgnoredDistribution):
        return None
    return fitted


def _extract_numeric_vector(sample: Any) -> np.ndarray | None:
    if isinstance(sample, (bool, np.bool_)):
        return None
    if isinstance(sample, (int, float, np.integer, np.floating)):
        return np.asarray([float(sample)])
    if isinstance(sample, np.ndarray):
        try:
            return sample.astype(float).ravel()
        except (TypeError, ValueError):
            return None
    if isinstance(sample, (list, tuple)):
        try:
            return np.asarray(sample, dtype=float).ravel()
        except (TypeError, ValueError):
            return None
    # A generic object carrying its numeric payload under a common attribute name.
    for attr in ("vector", "vec", "embedding", "features", "values", "array", "x"):
        val = getattr(sample, attr, None)
        if val is None:
            continue
        try:
            return np.asarray(val, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
    return None


def _coerce_numeric_matrix(rows: Sequence[Any]) -> np.ndarray | None:
    vectors = []
    for row in rows:
        vec = _extract_numeric_vector(row)
        if vec is None or vec.size == 0 or not np.all(np.isfinite(vec)):
            return None
        vectors.append(vec)
    dims = {v.size for v in vectors}
    if len(dims) != 1:
        return None
    return np.stack(vectors, axis=0)


def _fit_numeric_fallback(arr: np.ndarray, *, rng, max_its):
    from mixle.inference.estimation import optimize
    from mixle.utils.automatic.factories import _has_torch

    n, dim = arr.shape
    if dim == 1:
        from mixle.stats.univariate.continuous.gaussian import GaussianEstimator

        data = [float(v) for v in arr[:, 0]]
        return optimize(data, GaussianEstimator(), max_its=max_its, rng=rng)

    if dim < _EMBEDDING_MIN_DIM or not _has_torch():
        from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianEstimator

        data = [tuple(float(v) for v in row) for row in arr]
        return optimize(data, MultivariateGaussianEstimator(dim=dim), max_its=max_its, rng=rng)

    # complex / high-dimensional numeric payload with no closed-form fingerprint the profiler
    # already knows: fall back to a generic GradLeaf-wrapped neural density (evaluation-only
    # scoring, fit by gradient ascent -- see mixle.models.grad_leaf).
    return _fit_generic_grad_density(arr, max_its=max_its)


class _GenericDensityModule:
    """A small learnable diagonal-Gaussian-mixture density -- the generic neural-density fallback
    for a numeric modality with no other recognized fingerprint. Built lazily so this module never
    imports torch at import time (GradLeaf/torch is an optional dependency)."""

    def __new__(cls, dim: int, n_components: int = 2):
        import torch
        from torch import nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dim = dim
                self.means = nn.Parameter(torch.randn(n_components, dim) * 0.1)
                self.log_vars = nn.Parameter(torch.zeros(n_components, dim))
                self.logits = nn.Parameter(torch.zeros(n_components))

            def log_density(self, x: Any) -> Any:
                var = torch.exp(self.log_vars)
                diff = x.unsqueeze(1) - self.means.unsqueeze(0)
                comp_ll = -0.5 * (
                    diff**2 / var.unsqueeze(0) + self.log_vars.unsqueeze(0) + math.log(2.0 * math.pi)
                ).sum(-1)
                log_w = torch.log_softmax(self.logits, dim=0)
                return torch.logsumexp(comp_ll + log_w.unsqueeze(0), dim=1)

            def sample(self, n: int) -> Any:
                w = torch.softmax(self.logits, dim=0)
                idx = torch.multinomial(w, n, replacement=True)
                std = torch.exp(0.5 * self.log_vars[idx])
                eps = torch.randn(n, self.dim)
                return self.means[idx] + eps * std

        return _Module()


def _fit_generic_grad_density(arr: np.ndarray, *, max_its: int, n_components: int = 2):
    from mixle.inference.estimation import optimize
    from mixle.models.grad_leaf import GradEstimator

    dim = arr.shape[1]
    module = _GenericDensityModule(dim, n_components=n_components)
    estimator = GradEstimator(module, m_steps=max(60, int(max_its) * 10))
    # NOT a tuple per row: GradLeafEncoder.seq_encode treats a tuple observation as one field PER
    # POSITION (the conditional log_density(x, y, ...) convention) -- a 24-dim row as a 24-tuple was
    # being split into 24 separate scalar fields instead of one vector field, so the module's
    # log_density(x) (a single positional arg) was being called with 24 unpacked scalars.
    data = [np.asarray(row, dtype=float) for row in arr]
    return optimize(data, estimator, max_its=1)


def _extract_adjacency(sample: Any) -> Any | None:
    if hasattr(sample, "nodes") and hasattr(sample, "edges") and not isinstance(sample, np.ndarray):
        return sample  # networkx-like; the graph encoder handles this shape directly.
    arr = None
    for attr in ("adjacency", "adjacency_matrix", "matrix", "graph"):
        val = getattr(sample, attr, None)
        if val is not None:
            try:
                arr = np.asarray(val, dtype=float)
            except (TypeError, ValueError):
                arr = None
            break
    if arr is None and isinstance(sample, np.ndarray):
        arr = sample
    if arr is None and isinstance(sample, (list, tuple)):
        try:
            arr = np.asarray(sample, dtype=float)
        except (TypeError, ValueError):
            arr = None
    if arr is not None and arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 2:
        return arr
    return None


def _coerce_graph_rows(rows: Sequence[Any]) -> list[Any] | None:
    out = []
    for row in rows:
        adj = _extract_adjacency(row)
        if adj is None:
            return None
        out.append(adj)
    return out


def _fit_graph_fallback(graph_rows: list[Any], *, max_its: int):
    from mixle.inference.estimation import optimize
    from mixle.stats.graphs.erdos_renyi_graph import ErdosRenyiGraphEstimator

    return optimize(graph_rows, ErdosRenyiGraphEstimator(), max_its=max_its)


def _coerce_sequence_elements(rows: Sequence[Any]) -> list[Any] | None:
    """A variable-length (or genuinely heterogeneous) collection of elements the profiler could not
    place; flatten to the pooled element stream so the caller can induce a leaf for the element type
    and wrap it as a sequence."""
    elems: list[Any] = []
    saw_iterable = False
    for row in rows:
        if isinstance(row, (str, bytes, dict, set, frozenset, np.ndarray)) or not hasattr(row, "__iter__"):
            return None
        items = list(row)
        if len(items) == 0:
            continue
        saw_iterable = True
        elems.extend(items)
    if not saw_iterable or not elems:
        return None
    return elems


def _fit_sequence_fallback(rows, elems, *, rng, max_its):
    from mixle.inference.estimation import optimize
    from mixle.utils.automatic.factories import get_sequence_estimator

    child_leaf = induce_leaf_for_unseen_type(elems, rng=rng, max_its=max_its)
    len_dict: dict[int, int] = {}
    for row in rows:
        length = len(list(row))
        len_dict[length] = len_dict.get(length, 0) + 1
    seq_est = get_sequence_estimator(child_leaf.estimator(), len_dict=len_dict)
    return optimize(list(rows), seq_est, max_its=max_its, rng=rng)


# ---------------------------------------------------------------------------------------------
# 2. RESONANCE embedding: evaluation-only typicality coordinates against an existing model zoo.
# ---------------------------------------------------------------------------------------------


def _generic_scalar_reduction(sample: Any) -> float:
    """A model-agnostic scalar summary of an arbitrary (possibly brand-new-modality) sample: the
    mean of its extracted numeric payload, or its own hash-derived pseudo-magnitude when no numeric
    payload can be extracted at all (still deterministic, still comparable across samples)."""
    vec = _extract_numeric_vector(sample)
    if vec is not None and vec.size:
        return float(np.mean(vec))
    return float(abs(hash(repr(sample))) % 1000) / 1000.0


def _model_typicality(model: Any, value: float) -> float:
    """One generic "how typical does ``value`` look under ``model``'s own typical range" feature,
    read off ``model`` purely by evaluation (its closed-form CDF/moments) -- no fitting, no
    gradients. Returns 0.0 for a value squarely at the model's center and approaches 1.0 the more
    atypical/extreme the value looks under that model's reference scale."""
    if supports(model, HasCDF):
        try:
            u = float(model.cdf(value))
            if math.isfinite(u):
                return float(2.0 * abs(min(max(u, 0.0), 1.0) - 0.5))
        except Exception:  # noqa: BLE001
            pass
    if supports(model, HasMoments):
        try:
            mean = float(model.mean())
            std = float(model.variance()) ** 0.5
            z = abs(value - mean) / max(std, 1.0e-9)
            return float(z / (1.0 + z))
        except Exception:  # noqa: BLE001
            pass
    return 0.5


def resonance_embedding(
    new_modality_samples: Sequence[Any],
    model_zoo: Sequence[SequenceEncodableProbabilityDistribution],
) -> np.ndarray:
    """K-dim (K = ``len(model_zoo)``) embedding of each brand-new-modality sample, obtained by
    EVALUATION ONLY against every existing, already-fitted zoo model (regardless of the modality it
    was originally fitted to): coordinate ``k`` is how atypical a generic scalar reduction of the
    sample looks under zoo model ``k``'s own closed-form typical range (its CDF/quantile position, or
    a moment-normalized z-score when no CDF is exposed). No gradient steps, no fitting -- "zero
    training."
    """
    if len(model_zoo) == 0:
        raise ValueError("resonance_embedding requires a non-empty model_zoo")
    rows = []
    for sample in new_modality_samples:
        value = _generic_scalar_reduction(sample)
        rows.append([_model_typicality(model, value) for model in model_zoo])
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------------------------
# 3. Fit-health-style adequacy gate: is the resonance embedding good enough, or graduate?
# ---------------------------------------------------------------------------------------------


def _deterministic_two_means_split(proj: np.ndarray) -> np.ndarray:
    """The same deterministic 2-means procedure (top-PC sign init, Lloyd iterations) used by
    :func:`mixle.utils.hvis.topology.model_fit_health`'s merged-regime detector."""
    assign = proj > float(proj.mean())
    for _ in range(15):
        if assign.all() or (~assign).all():
            break
        c1, c0 = float(proj[assign].mean()), float(proj[~assign].mean())
        new_assign = np.abs(proj - c1) < np.abs(proj - c0)
        if bool(np.all(new_assign == assign)):
            break
        assign = new_assign
    return assign


def _top_pc_projection(z: np.ndarray) -> np.ndarray:
    mu = z.mean(axis=0)
    centered = z - mu
    if z.shape[1] == 1:
        return centered[:, 0]
    cov = centered.T @ centered / max(z.shape[0] - 1, 1) + 1.0e-9 * np.eye(z.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    return centered @ axis


def _separation_ratio(proj: np.ndarray, group_a: np.ndarray, group_b: np.ndarray) -> float | None:
    pa, pb = proj[group_a], proj[group_b]
    if pa.size < 2 or pb.size < 2:
        return None
    within = np.average([pa.var(), pb.var()], weights=[pa.size, pb.size])
    return float(abs(pa.mean() - pb.mean())) / max(float(np.sqrt(within)), 1.0e-12)


def resonance_adequacy_gate(
    embedding_samples: np.ndarray,
    labels_or_structure: Sequence[Any] | None = None,
    *,
    threshold: float | None = None,
) -> bool:
    """Reuse the fit-health merged-regime separation statistic (see
    :func:`mixle.utils.hvis.topology.model_fit_health`) to decide whether the resonance embedding's
    class separation is adequate to use as a lightweight, training-free proxy representation
    indefinitely (``True``), or whether the modality should GRADUATE to a real native leaf
    (``False``).

    ``labels_or_structure``, when given, is the known class/cluster label per embedding row; the
    worst (minimum) pairwise separation across classes is compared against the SAME finite-sample
    threshold ``model_fit_health`` uses for its own merged/unmerged call:
    ``2.65 + 6/sqrt(n)`` (population value ~2.65 for a unimodal normal, inflated at small n).
    Without labels, the same deterministic 2-means split ``model_fit_health`` runs internally is used
    to discover a candidate 2-way structure and test whether IT is well separated.
    """
    z = np.atleast_2d(np.asarray(embedding_samples, dtype=float))
    n = z.shape[0]
    if n < 4:
        return False  # too little evidence to trust the proxy -- graduate.

    proj = _top_pc_projection(z)
    thr = threshold if threshold is not None else 2.65 + 6.0 / math.sqrt(float(n))

    if labels_or_structure is not None:
        labels = np.asarray(list(labels_or_structure))
        if labels.shape[0] != n:
            raise ValueError("labels_or_structure must have one entry per embedding row")
        uniq = np.unique(labels)
        if uniq.size < 2:
            return False
        seps = []
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                sep = _separation_ratio(proj, labels == a, labels == b)
                if sep is not None:
                    seps.append(sep)
        if not seps:
            return False
        return min(seps) > thr

    assign = _deterministic_two_means_split(proj)
    if assign.all() or (~assign).all():
        return False  # no structure at all was found -- nothing to be confident about.
    sep = _separation_ratio(proj, assign, ~assign)
    return sep is not None and sep > thr


# ---------------------------------------------------------------------------------------------
# 4. Integration: add the new modality to a CrossModalJoint without retraining anything else.
# ---------------------------------------------------------------------------------------------


def add_modality_to_joint(
    joint: CrossModalJoint,
    name: str,
    per_regime_leaves: Sequence[SequenceEncodableProbabilityDistribution],
) -> CrossModalJoint:
    """Return a NEW :class:`CrossModalJoint` with modality ``name`` added, one leaf per existing
    regime, WITHOUT touching any other modality's fitted parameters.

    Every existing per-regime :class:`~mixle.stats.combinator.composite.CompositeDistribution` is
    rebuilt with its OLD field distributions reused by reference (never copied, never refit) plus the
    one new field appended; the mixture weights are reused unchanged. ``per_regime_leaves`` can be
    the modality's induced native leaf (see :func:`induce_leaf_for_unseen_type`) replicated across
    regimes, or per-regime leaves fit only over resonance-embedding coordinates (see
    :func:`fit_resonance_leaves`) -- either way, this function itself performs no fitting at all.
    """
    components = joint.joint.components
    if len(per_regime_leaves) != len(components):
        raise ValueError(
            f"per_regime_leaves must supply one distribution per existing regime "
            f"({len(components)} regimes), got {len(per_regime_leaves)}"
        )
    new_components = [
        CompositeDistribution(list(component.dists) + [leaf]) for component, leaf in zip(components, per_regime_leaves)
    ]
    new_joint = MixtureDistribution(new_components, w=joint.joint.w.copy())
    return CrossModalJoint(names=joint.names + (name,), joint=new_joint)


def fit_resonance_leaves(
    resonance_embeddings: np.ndarray,
    regime_labels: Sequence[int],
    num_regimes: int,
) -> list[SequenceEncodableProbabilityDistribution]:
    """Fit one lightweight per-regime leaf directly over the K-dim resonance-embedding coordinates
    (a closed-form multivariate Gaussian, or a univariate Gaussian when K == 1), using known/assigned
    regime labels for the new-modality samples.

    This fit touches ONLY the new modality's own (yet-to-exist) leaf -- it never reads or writes any
    other modality's parameters, so combining its output with :func:`add_modality_to_joint` cannot
    retrain the rest of the joint.
    """
    from mixle.inference.estimation import optimize
    from mixle.stats.univariate.continuous.gaussian import GaussianEstimator

    z = np.atleast_2d(np.asarray(resonance_embeddings, dtype=float))
    labels = np.asarray(list(regime_labels))
    dim = z.shape[1]
    leaves: list[SequenceEncodableProbabilityDistribution] = []
    for k in range(num_regimes):
        rows = z[labels == k]
        if rows.shape[0] == 0:
            rows = z  # no assigned samples for this regime -- fall back to the pooled fit.
        if dim == 1:
            leaves.append(optimize([float(v) for v in rows[:, 0]], GaussianEstimator(), max_its=10))
        else:
            from mixle.stats.multivariate.multivariate_gaussian import MultivariateGaussianEstimator

            data = [tuple(float(v) for v in row) for row in rows]
            leaves.append(optimize(data, MultivariateGaussianEstimator(dim=dim), max_its=10))
    return leaves
