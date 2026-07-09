"""Per-NODE precision planning for a composed distribution tree.

``mixle.inference.precision_plan`` picks ONE compute precision for a whole model. This module
generalizes that decision to every NODE of a composed tree (a :class:`~mixle.stats.latent.mixture.
MixtureDistribution` of components, a :class:`~mixle.stats.combinator.composite.CompositeDistribution`
of factors, and any nesting of the two): each node gets its own safety verdict, reusing the exact
per-leaf safety check ``precision_plan`` already validates (family whitelist + variance floor), then
those leaf verdicts are aggregated UP the tree. This is the roadmap's "fits are deterministic given
seed, and sufficient statistics are ADDITIVE, so error bounds compose like stats" insight applied
literally: a non-leaf node is float32-safe iff every leaf beneath it is, and its advertised summed-LL
error bound is the SUM of its leaves' bounds (each leaf's bound independently verified, see
``precision_plan``'s module docstring).

Two things live here:

1. :func:`recommend_tree_precision` -- walks the WHOLE tree and returns a :class:`TreePrecisionPlan`:
   an inspectable, path-keyed mapping from every node (leaf and non-leaf) to its chosen precision and
   rationale. This is the "D1-reported property / D6-H3 action" surface: a caller (a future node-report
   or a block-freeze policy) can read exactly which sub-blocks are safe to run cheap, without re-deriving
   the verdict.

2. :func:`mixed_precision_fit` -- actually EXECUTES an EM fit where each top-level child of the root
   combinator (each mixture component, or each composite factor) runs its E-step scoring and
   sufficient-statistic accumulation at ITS OWN assigned precision. See the "Execution scope" note in
   that function's docstring for exactly how far genuine per-node execution reaches in the current
   architecture, and where it honestly falls back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.inference.precision_plan import _FP32_SAFE

# The verified reduced-precision band from precision_plan's module docstring: the fused float32 score
# stays within ~1e-6 RELATIVE summed-log-likelihood error of the float64 computation. Because
# sufficient statistics (and, to first order, per-row log-densities) are ADDITIVE across independent
# leaves/factors, the advertised bound for a subtree with N reduced-precision leaves is N * this
# constant -- see TreePrecisionPlan.advertised_bound.
FUSED_FP32_REL_LL_BOUND = 1e-6

Path = tuple[str, ...]


@dataclass(frozen=True)
class NodePrecision:
    """The precision verdict for ONE node of a composed tree.

    Attributes:
        path: Field-path identifying this node from the tree root, e.g. ``("components", "0",
            "dists", "1")`` for the second factor of the first mixture component. ``()`` is the root.
        node_type: The node's class name (``"MixtureDistribution"``, ``"CompositeDistribution"``, or
            the leaf's own class name).
        is_leaf: True for an actual distribution leaf (not a mixture/composite combinator).
        compute_dtype: The chosen dtype (``np.float32`` or ``np.float64``).
        rationale: Human-readable reason, reusing precision_plan's per-leaf wording where applicable.
        rel_error_bound: The advertised relative summed-log-likelihood error bound FOR THIS NODE's
            subtree (0.0 when ``compute_dtype`` is float64 -- exact).
        leaf_count: Number of leaves in this node's subtree (1 for a leaf itself).
    """

    path: Path
    node_type: str
    is_leaf: bool
    compute_dtype: Any
    rationale: str
    rel_error_bound: float
    leaf_count: int

    def reduced(self) -> bool:
        return np.dtype(self.compute_dtype) != np.float64


@dataclass
class TreePrecisionPlan:
    """The full per-node precision plan for a composed tree: path -> :class:`NodePrecision`.

    This is the inspectable/actionable artifact the roadmap calls for: iterate ``nodes`` to see every
    block's verdict, call :meth:`dtype_for` to look up one node, or :meth:`reduced_paths` /
    :meth:`frozen_candidates` to drive a future block-freeze / precision-drop policy (D6/H3). Hooking
    this into D1's ``NodeReport`` (once that lands) is a natural follow-up -- see the module docstring.
    """

    root_type: str
    nodes: dict[Path, NodePrecision] = field(default_factory=dict)

    def dtype_for(self, path: Path) -> Any:
        return self.nodes[path].compute_dtype

    def leaf_paths(self) -> list[Path]:
        return [p for p, n in self.nodes.items() if n.is_leaf]

    def reduced_paths(self) -> list[Path]:
        """Paths (any node, leaf or subtree) allocated float32."""
        return [p for p, n in self.nodes.items() if n.reduced()]

    def top_level_child_paths(self) -> list[Path]:
        """Paths one level below the root -- the granularity :func:`mixed_precision_fit` can actually
        execute at independently (see that function's docstring for why)."""
        return sorted((p for p in self.nodes if len(p) == 1), key=lambda p: int(p[-1]) if p[-1].isdigit() else p[-1])

    def advertised_bound(self, path: Path = ()) -> float:
        """The advertised relative summed-log-likelihood error bound for ``path``'s subtree (default:
        whole tree). Sums the verified per-leaf bound (``FUSED_FP32_REL_LL_BOUND``) over every REDUCED
        leaf beneath ``path`` -- the additive composition the roadmap calls for."""
        return self.nodes[path].rel_error_bound

    def summary(self) -> str:
        lines = [f"TreePrecisionPlan({self.root_type}):"]
        for path, n in sorted(self.nodes.items(), key=lambda kv: (len(kv[0]), kv[0])):
            label = "root" if not path else ".".join(path)
            lines.append(f"  {label}: {np.dtype(n.compute_dtype).name} -- {n.rationale}")
        return "\n".join(lines)


def _leaf_safety(leaf: Any, min_variance: float) -> tuple[bool, str]:
    """Reuse of precision_plan's per-leaf safety check: family whitelist + variance floor.

    Returns (is_safe, rationale). Pulled out so both the model-global allocator (precision_plan) and
    this per-node allocator apply the IDENTICAL verified criteria to a leaf -- one source of truth.
    """
    name = type(leaf).__name__
    if name not in _FP32_SAFE:
        return False, "%s is not float32-safe -> float64" % name
    s2 = getattr(leaf, "sigma2", None)
    if s2 is not None and float(s2) < min_variance:
        return False, "near-degenerate component (var %.1e) -> float64 for accuracy" % float(s2)
    return True, "%s is float32-safe" % name


def _data_magnitude_safe(data: Any, max_magnitude: float, sample_size: int) -> tuple[bool, str, float | None]:
    """Reuse of precision_plan's data-magnitude guard (stride-sampled, not a leading prefix)."""
    from mixle.engines.precision import _numeric_data_sample

    if hasattr(data, "__getitem__") and hasattr(data, "__len__"):
        n = len(data)
        if n > sample_size:
            step = n / sample_size
            sample = [data[int(i * step)] for i in range(sample_size)]
        else:
            sample = data
    else:
        sample = data
    s = _numeric_data_sample(sample, sample_size)
    if s is None or s.size == 0:
        return False, "non-numeric / empty data -> float64", None
    amax = float(np.max(np.abs(s)))
    if amax > max_magnitude:
        return False, "data magnitude %.1e too large for float32 -> float64" % amax, amax
    return True, "bounded magnitude (|x|<=%.0e)" % amax, amax


def _walk(
    model: Any,
    path: Path,
    data_safe: bool,
    data_rationale: str,
    min_variance: float,
    nodes: dict[Path, NodePrecision],
) -> NodePrecision:
    """Recursively compute the per-node verdict, aggregating bottom-up (post-order): a non-leaf node
    is float32-safe iff every child is, and its bound is the sum of its children's bounds."""
    tname = type(model).__name__
    if tname == "MixtureDistribution":
        children = [
            _walk(c, path + ("components", str(i)), data_safe, data_rationale, min_variance, nodes)
            for i, c in enumerate(model.components)
        ]
    elif tname == "CompositeDistribution":
        children = [
            _walk(d, path + ("dists", str(i)), data_safe, data_rationale, min_variance, nodes)
            for i, d in enumerate(model.dists)
        ]
    else:
        children = None

    if children is not None:
        safe = data_safe and all(c.reduced() for c in children)
        leaf_count = sum(c.leaf_count for c in children)
        if not data_safe:
            rationale = data_rationale
        elif safe:
            rationale = "all %d leaves float32-safe -> float32" % leaf_count
        else:
            unsafe = [".".join(c.path) or "<child>" for c in children if not c.reduced()]
            rationale = "unsafe leaf(ren) below %s -> float64" % (", ".join(unsafe) or "?")
        dtype = np.float32 if safe else np.float64
        bound = FUSED_FP32_REL_LL_BOUND * leaf_count if safe else 0.0
        node = NodePrecision(
            path=path,
            node_type=tname,
            is_leaf=False,
            compute_dtype=dtype,
            rationale=rationale,
            rel_error_bound=bound,
            leaf_count=leaf_count,
        )
        nodes[path] = node
        return node

    # leaf
    leaf_safe, leaf_rationale = _leaf_safety(model, min_variance)
    safe = data_safe and leaf_safe
    if not data_safe:
        rationale = data_rationale
    else:
        rationale = leaf_rationale
    dtype = np.float32 if safe else np.float64
    node = NodePrecision(
        path=path,
        node_type=tname,
        is_leaf=True,
        compute_dtype=dtype,
        rationale=rationale,
        rel_error_bound=FUSED_FP32_REL_LL_BOUND if safe else 0.0,
        leaf_count=1,
    )
    nodes[path] = node
    return node


def recommend_tree_precision(
    model: Any,
    data: Any,
    min_variance: float = 1e-6,
    max_magnitude: float = 1e6,
    sample_size: int = 4096,
) -> TreePrecisionPlan:
    """Return the per-NODE precision plan for a composed tree.

    Walks ``model`` (a Mixture / Composite / leaf, and any nesting thereof) and computes a safety
    verdict at EVERY node: leaves get the identical per-leaf check ``precision_plan`` uses (family
    whitelist + variance floor); non-leaf nodes aggregate their children (safe iff ALL children are
    safe) and their advertised error bound is the additive sum of their leaves' bounds. The (global)
    data-magnitude check is evaluated once against ``data`` and gates every node uniformly, matching
    ``precision_plan.recommend_compute_precision`` -- only the leaf/family conditioning genuinely
    varies node-to-node in this codebase (see this module's and ``mixed_precision_fit``'s docstrings
    for why data conditioning isn't yet split per-node).

    Args:
        model: The composed distribution tree (root).
        data: Representative data used to check the magnitude guard (see precision_plan).
        min_variance: Leaves with ``sigma2`` below this are treated as near-degenerate -> float64.
        max_magnitude: Data magnitude guard, identical semantics to precision_plan.
        sample_size: Stride-sample size used for the magnitude guard.

    Returns:
        A :class:`TreePrecisionPlan` with one :class:`NodePrecision` per node (root included, at
        path ``()``).
    """
    nodes: dict[Path, NodePrecision] = {}
    if model is None:
        nodes[()] = NodePrecision((), "NoneType", True, np.float64, "no model to inspect -> float64", 0.0, 0)
        return TreePrecisionPlan(root_type="NoneType", nodes=nodes)

    data_safe, data_rationale, _ = _data_magnitude_safe(data, max_magnitude, sample_size)
    _walk(model, (), data_safe, data_rationale, min_variance, nodes)
    return TreePrecisionPlan(root_type=type(model).__name__, nodes=nodes)


# --------------------------------------------------------------------------------------------------
# Execution: run a mixed-precision EM fit using the per-node plan.
# --------------------------------------------------------------------------------------------------


def mixed_precision_fit(
    model: Any,
    data: Any,
    plan: TreePrecisionPlan | None = None,
    max_its: int = 10,
    delta: float | None = 1.0e-9,
    weights: np.ndarray | None = None,
) -> Any:
    """Fit ``model`` with each TOP-LEVEL CHILD of the root combinator executing its E-step (scoring +
    sufficient-statistic accumulation) at its OWN precision, per ``plan``.

    Execution scope (read this before trusting "mixed precision" claims elsewhere): the numba fused
    kernel (``mixle.stats.compute.fused_codegen``) compiles ONE kernel per fusible subtree and runs it
    at ONE dtype end to end -- there is no way to hand it two different literal dtypes inside a single
    call. So the finest granularity at which this codebase can genuinely execute *different* literal
    precisions *within one fit* is the boundary between independently-callable fused subtrees, which
    is exactly the immediate children of the root combinator: each mixture COMPONENT, or each
    composite FACTOR. Nesting deeper than that (e.g. two factors of a Composite that is itself one
    mixture component) shares one dtype -- the whole subtree is one fused-kernel call, so it gets the
    AND-aggregated verdict :func:`recommend_tree_precision` already computes for it.

    This is genuinely DIFFERENT from ``mixle.inference.optimize(precision=...)``: that entry point
    threads exactly one ``engine`` (one dtype) through the WHOLE fit via a single ``NumpyEngine`` /
    ``FusedKernel`` -- there is currently no per-node engine plumbed through ``optimize``'s EM loop.
    This function does NOT go through ``optimize``; it is a standalone driver, scoped to a root
    ``MixtureDistribution`` or ``CompositeDistribution`` (any nesting below each top-level child is
    fine -- it just shares that child's one dtype, as described above). Anything else (a bare leaf, or
    a combinator this driver doesn't recognize) is fit at plain float64 with a warning-free no-op
    fallback (there is nothing to split).

    Each child accumulates sufficient statistics in its OWN accumulator, at its OWN dtype for the row
    arithmetic; every accumulator's OUTPUT (and the softmax/logsumexp responsibility normalization for
    a mixture) is float64, matching the "accumulation is ALWAYS float64" invariant precision_plan
    documents -- so, like the model-global allocator, results never drift regardless of which nodes
    ran reduced; only the per-row SCORE of the reduced nodes is computed cheaper.

    Args:
        model: MixtureDistribution or CompositeDistribution to fit (used as both the shape AND the
            starting parameter estimate -- pass an initialized model, e.g. from ``estimator().estimate``
            or a previous ``optimize`` call).
        data: Training data.
        plan: A :class:`TreePrecisionPlan` (e.g. from :func:`recommend_tree_precision`). ``None``
            computes one internally against ``data``.
        max_its: Maximum EM iterations.
        delta: Convergence threshold on the per-iteration total log-likelihood change. ``None`` runs
            exactly ``max_its`` iterations.
        weights: Optional per-observation weights (default: uniform 1.0).

    Returns:
        The fitted model (same top-level type as ``model``).
    """
    tname = type(model).__name__
    if plan is None:
        plan = recommend_tree_precision(model, data)

    if tname == "MixtureDistribution":
        return _mixed_precision_fit_mixture(model, data, plan, max_its, delta, weights)
    if tname == "CompositeDistribution":
        return _mixed_precision_fit_composite(model, data, plan, max_its, delta, weights)
    # Nothing to split at the top level -- fall back to the ordinary (float64) fit for correctness.
    from mixle.inference.estimation import optimize

    return optimize(data, model.estimator(), prev_estimate=model, max_its=max_its, delta=delta, out=None)


def _child_score(child: Any, enc: Any, compute_dtype: Any) -> np.ndarray:
    from mixle.stats.compute.fused_codegen import fused_seq_log_density, fusible

    dtype = compute_dtype if compute_dtype is not None and np.dtype(compute_dtype) != np.float64 else None
    if fusible(child):
        return fused_seq_log_density(child, enc, compute_dtype=dtype)
    return np.asarray(child.seq_log_density(enc), dtype=np.float64)


def _child_accumulate(child: Any, enc: Any, w: np.ndarray, compute_dtype: Any) -> Any:
    from mixle.stats.compute.fused_codegen import fused_accumulate, fusible

    dtype = compute_dtype if compute_dtype is not None and np.dtype(compute_dtype) != np.float64 else None
    if fusible(child):
        return fused_accumulate(child, enc, w, compute_dtype=dtype)
    acc = child.accumulator_factory().make()
    acc.seq_update(enc, w, child)
    return acc.value()


def _mixed_precision_fit_mixture(
    model: Any,
    data: Any,
    plan: TreePrecisionPlan,
    max_its: int,
    delta: float | None,
    weights: np.ndarray | None,
) -> Any:
    n = len(data)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    K = model.num_components
    encs = [model.components[i].dist_to_encoder().seq_encode(data) for i in range(K)]
    dtypes = [plan.dtype_for(("components", str(i))) for i in range(K)]

    components = list(model.components)
    log_w = np.log(np.asarray(model.w, dtype=np.float64) + 1e-300)
    prev_total_ll: float | None = None

    for _ in range(max_its):
        ll_mat = np.full((n, K), -np.inf, dtype=np.float64)
        for i in range(K):
            ll_mat[:, i] = _child_score(components[i], encs[i], dtypes[i]) + log_w[i]

        ll_max = ll_mat.max(axis=1, keepdims=True)
        bad = np.isinf(ll_max.flatten())
        if np.any(bad):
            ll_mat[bad, :] = log_w.copy()
            ll_max[bad] = np.max(log_w)
        shifted = ll_mat - ll_max
        np.exp(shifted, out=shifted)
        row_sum = shifted.sum(axis=1, keepdims=True)
        total_ll = float(np.dot(w, (ll_max[:, 0] + np.log(row_sum[:, 0]))))
        resp = shifted * (w[:, None] / row_sum)

        comp_counts = resp.sum(axis=0)
        suff_stats = [_child_accumulate(components[i], encs[i], resp[:, i], dtypes[i]) for i in range(K)]

        estimators = model.estimator().estimators
        components = [estimators[i].estimate(comp_counts[i], suff_stats[i]) for i in range(K)]
        total = comp_counts.sum()
        new_w = comp_counts / total if total > 0 else np.asarray(model.w, dtype=np.float64)
        log_w = np.log(new_w + 1e-300)

        if delta is not None and prev_total_ll is not None and abs(total_ll - prev_total_ll) < delta:
            prev_total_ll = total_ll
            break
        prev_total_ll = total_ll

    from mixle.stats.latent.mixture import MixtureDistribution

    return MixtureDistribution(components, list(np.exp(log_w)))


def _mixed_precision_fit_composite(
    model: Any,
    data: Any,
    plan: TreePrecisionPlan,
    max_its: int,
    delta: float | None,
    weights: np.ndarray | None,
) -> Any:
    n = len(data)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    m = len(model.dists)
    # Composite factors observe x[i] of each tuple observation.
    factor_data = [[x[i] for x in data] for i in range(m)]
    encs = [model.dists[i].dist_to_encoder().seq_encode(factor_data[i]) for i in range(m)]
    dtypes = [plan.dtype_for(("dists", str(i))) for i in range(m)]

    dists = list(model.dists)
    prev_total_ll: float | None = None

    for _ in range(max_its):
        total_ll = 0.0
        suff_stats = []
        for i in range(m):
            total_ll += float(np.dot(w, _child_score(dists[i], encs[i], dtypes[i])))
            suff_stats.append(_child_accumulate(dists[i], encs[i], w, dtypes[i]))

        estimator = model.estimator()
        child_estimators = getattr(estimator, "estimators", None)
        if child_estimators is None:
            # single-key estimators (e.g. tied factors) aren't supported by this standalone driver.
            child_estimators = [d.estimator() for d in dists]
        dists = [child_estimators[i].estimate(w.sum(), suff_stats[i]) for i in range(m)]

        if delta is not None and prev_total_ll is not None and abs(total_ll - prev_total_ll) < delta:
            prev_total_ll = total_ll
            break
        prev_total_ll = total_ll

    from mixle.stats.combinator.composite import CompositeDistribution

    return CompositeDistribution(tuple(dists))
