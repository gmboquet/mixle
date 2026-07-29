"""Executed-subtree precision planning for a composed distribution tree.

``mixle.inference.precision_plan`` picks ONE compute precision for a whole model. This module
generalizes that decision to every NODE of a composed tree (a :class:`~mixle.stats.latent.mixture.
MixtureDistribution` of components, a :class:`~mixle.stats.combinator.composite.CompositeDistribution`
of factors, and any nesting of the two): each node gets its own safety verdict, reusing the exact
per-leaf safety check ``precision_plan`` already validates (family whitelist + variance floor).
The proposed verdict is then reconciled with the subtrees the mixed-precision driver can actually
execute. Reduced subtrees are compared against float64 on the supplied data and carry an absolute
log-score enclosure. Relative likelihood errors are never added: cancellation can make that
composition unbounded.

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

import hashlib
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from mixle.inference.precision_plan import _FP32_SAFE, _has_non_finite

# An executable subtree remains reduced only when its observed absolute
# per-row log-score error is within this validation tolerance.
FUSED_FP32_ABS_LOG_TOLERANCE = 1e-4

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
        abs_log_error_bound: Triangle-inequality enclosure on the absolute error of the unweighted
            summed log score over the validation rows. It is not a relative or out-of-sample bound.
        leaf_count: Number of leaves in this node's subtree (1 for a leaf itself).
        validation_rows: Number of rows used to establish ``abs_log_error_bound``.
        execution_scope: Whether this node executes independently, inherits a parent's dtype, or is
            combined in float64.
    """

    path: Path
    node_type: str
    is_leaf: bool
    compute_dtype: Any
    rationale: str
    abs_log_error_bound: float | None
    leaf_count: int
    validation_rows: int = 0
    execution_scope: str = "proposed"

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
    validation_fingerprint: str | None = None
    validation_rows: int = 0

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
        paths = [
            path for path in self.nodes if len(path) == 2 and path[0] in {"components", "dists"} and path[1].isdigit()
        ]
        return sorted(paths, key=lambda path: (path[0], int(path[1])))

    def advertised_bound(self, path: Path = ()) -> float:
        """Return the validation-data absolute summed-log-score error enclosure for ``path``."""
        bound = self.nodes[path].abs_log_error_bound
        if bound is None:
            raise ValueError(f"node {path!r} does not execute independently and has no separate error enclosure")
        return bound

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
    # Check the RECORDS for non-finiteness before reducing them to a numeric sample. Composite and
    # mixture data is record-shaped -- tuples, dicts -- and _numeric_data_sample flattens to whatever
    # numeric leaves it can find, so a NaN sitting in one field of one record could be dropped on the
    # way in. The gate is documented as applying uniformly to every node, and it cannot do that on a
    # sample the NaN never reached: the fallback then happened only where a leaf was independently
    # unsafe, leaving otherwise-safe leaves reduced on data containing a NaN.
    if any(_has_non_finite(record) for record in sample):
        return False, "non-finite data (NaN/Inf) -> float64", None
    s = _numeric_data_sample(sample, sample_size)
    if s is None or s.size == 0:
        return False, "non-numeric / empty data -> float64", None
    if not np.all(np.isfinite(s)):
        # NaN/Inf anywhere in the sample must route to the safe fallback EXPLICITLY (MXR-080-0145,
        # mirroring precision_plan.recommend_compute_precision's identical guard): IEEE-754 defines
        # every comparison against NaN as False, so a NaN `amax` would make `amax > max_magnitude`
        # below silently evaluate to False and fall through to the SAFE-looking `return True` instead
        # of the correct float64 verdict -- the opposite of what "risk could not be computed" should
        # mean.
        return False, "non-finite data (NaN/Inf) -> float64", None
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
    """Recursively compute a proposed verdict before execution-boundary reconciliation."""
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
        node = NodePrecision(
            path=path,
            node_type=tname,
            is_leaf=False,
            compute_dtype=dtype,
            rationale=rationale,
            abs_log_error_bound=0.0,
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
        abs_log_error_bound=0.0,
        leaf_count=1,
    )
    nodes[path] = node
    return node


def _data_fingerprint(data: list[Any]) -> str:
    """Bind a plan to the exact validation rows used for its error enclosure."""
    digest = hashlib.sha256()
    for row in data:
        if isinstance(row, np.ndarray):
            arr = np.ascontiguousarray(row)
            digest.update(str(arr.dtype).encode())
            digest.update(repr(arr.shape).encode())
            digest.update(arr.tobytes())
        else:
            digest.update(repr(row).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _score_error_enclosure(child: Any, data: list[Any]) -> tuple[np.ndarray | None, str]:
    """Compare the exact executable fused subtree in float32 and float64 on ``data``."""
    from mixle.stats.compute.fused_codegen import fused_seq_log_density, fusible

    if not fusible(child):
        return None, "subtree is not fusible, so reduced precision cannot execute"
    try:
        encoded = child.dist_to_encoder().seq_encode(data)
        score64 = np.asarray(fused_seq_log_density(child, encoded, compute_dtype=None), dtype=np.float64)
        score32 = np.asarray(
            fused_seq_log_density(child, encoded, compute_dtype=np.float32),
            dtype=np.float64,
        )
    except Exception as exc:  # noqa: BLE001 - failed validation means safe float64 fallback
        return None, f"reduced-score validation failed ({type(exc).__name__})"
    if score32.shape != score64.shape or score64.shape != (len(data),):
        return None, "reduced and float64 scorers did not return one aligned score per row"
    if (
        np.any(np.isnan(score32))
        or np.any(np.isnan(score64))
        or np.any(np.isposinf(score32))
        or np.any(np.isposinf(score64))
    ):
        return None, "non-finite score encountered during precision validation"
    impossible32 = np.isneginf(score32)
    impossible64 = np.isneginf(score64)
    if not np.array_equal(impossible32, impossible64):
        return None, "reduced precision changed which observations are impossible"
    error = np.zeros(len(data), dtype=np.float64)
    finite = ~impossible64
    error[finite] = np.abs(score32[finite] - score64[finite])
    if not np.all(np.isfinite(error)):
        return None, "score-error enclosure was non-finite"
    maximum = float(np.max(error)) if error.size else 0.0
    if maximum > FUSED_FP32_ABS_LOG_TOLERANCE:
        return None, (f"observed absolute row-score error {maximum:.3g} exceeds {FUSED_FP32_ABS_LOG_TOLERANCE:.3g}")
    return error, f"validated max absolute row-score error {maximum:.3g}"


def _execution_children(model: Any, data: list[Any]) -> list[tuple[Path, Any, list[Any]]]:
    """Return the independent subtree calls made by :func:`mixed_precision_fit`."""
    tname = type(model).__name__
    if tname == "MixtureDistribution":
        return [(("components", str(i)), child, data) for i, child in enumerate(model.components)]
    if tname == "CompositeDistribution":
        children: list[tuple[Path, Any, list[Any]]] = []
        for i, child in enumerate(model.dists):
            try:
                child_data = [row[i] for row in data]
            except (IndexError, TypeError) as exc:
                raise ValueError(f"composite training row does not contain field {i}") from exc
            children.append((("dists", str(i)), child, child_data))
        return children
    return []


def _assign_executed_subtree(
    nodes: dict[Path, NodePrecision],
    path: Path,
    dtype: Any,
    rationale: str,
    bound: float,
    validation_rows: int,
) -> None:
    """Make every descendant reflect the one dtype actually used by its fused parent call."""
    for node_path, node in list(nodes.items()):
        if node_path[: len(path)] != path:
            continue
        if node_path == path:
            nodes[node_path] = replace(
                node,
                compute_dtype=dtype,
                rationale=rationale,
                abs_log_error_bound=bound,
                validation_rows=validation_rows,
                execution_scope="independent_subtree",
            )
        else:
            nodes[node_path] = replace(
                node,
                compute_dtype=dtype,
                rationale=f"{node.rationale}; inherits executed dtype from {'.'.join(path)}",
                abs_log_error_bound=0.0 if np.dtype(dtype) == np.float64 else None,
                validation_rows=validation_rows,
                execution_scope=f"inherited_from:{'.'.join(path)}",
            )


def _reconcile_execution(model: Any, data: list[Any], nodes: dict[Path, NodePrecision]) -> None:
    """Replace proposed leaf verdicts with the dtypes and bounds that actually execute."""
    children = _execution_children(model, data)
    if not children:
        root = nodes[()]
        nodes[()] = replace(
            root,
            compute_dtype=np.float64,
            rationale="no independently executable child subtree -> float64 fallback",
            abs_log_error_bound=0.0,
            validation_rows=len(data),
            execution_scope="float64_fallback",
        )
        return

    child_errors: list[np.ndarray] = []
    for path, child, child_data in children:
        proposed = nodes[path]
        error: np.ndarray | None = None
        detail = proposed.rationale
        if proposed.reduced():
            error, detail = _score_error_enclosure(child, child_data)
        if error is None:
            dtype = np.float64
            bound = 0.0
            if proposed.reduced():
                detail = f"{detail} -> float64"
            child_errors.append(np.zeros(len(data), dtype=np.float64))
        else:
            dtype = np.float32
            bound = float(np.sum(error))
            child_errors.append(error)
        _assign_executed_subtree(nodes, path, dtype, detail, bound, len(data))

    stacked = np.stack(child_errors, axis=1) if child_errors else np.zeros((len(data), 0))
    if type(model).__name__ == "MixtureDistribution":
        # logsumexp is 1-Lipschitz in the infinity norm.
        root_error = np.max(stacked, axis=1) if stacked.shape[1] else np.zeros(len(data))
    else:
        # Composite log scores add; triangle inequality gives a valid enclosure.
        root_error = np.sum(stacked, axis=1)
    root = nodes[()]
    root_rationale = (
        "child subtree scores combined in float64"
        if any(nodes[path].reduced() for path, _child, _data in children)
        else f"{root.rationale}; child subtree scores combined in float64"
    )
    nodes[()] = replace(
        root,
        compute_dtype=np.float64,
        rationale=root_rationale,
        abs_log_error_bound=float(np.sum(root_error)),
        validation_rows=len(data),
        execution_scope="float64_combine",
    )


def recommend_tree_precision(
    model: Any,
    data: Any,
    min_variance: float = 1e-6,
    max_magnitude: float = 1e6,
    sample_size: int = 4096,
) -> TreePrecisionPlan:
    """Return the per-NODE precision plan for a composed tree.

    Walks ``model`` (a Mixture / Composite / leaf, and any nesting thereof) and computes a safety
    proposed verdict at every node, then reconciles it with the independently callable subtrees the
    executor actually uses. Each reduced executable subtree is scored in float32 and float64 on
    ``data``; the plan records a triangle-inequality absolute error enclosure. Descendants inherit
    their executable parent's dtype and do not advertise independent bounds. The root mixture or
    composite combines child scores in float64.

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
        nodes[()] = NodePrecision(
            (),
            "NoneType",
            True,
            np.float64,
            "no model to inspect -> float64",
            0.0,
            0,
            execution_scope="float64_fallback",
        )
        return TreePrecisionPlan(root_type="NoneType", nodes=nodes)

    rows = list(data)
    data_safe, data_rationale, _ = _data_magnitude_safe(rows, max_magnitude, sample_size)
    _walk(model, (), data_safe, data_rationale, min_variance, nodes)
    _reconcile_execution(model, rows, nodes)
    return TreePrecisionPlan(
        root_type=type(model).__name__,
        nodes=nodes,
        validation_fingerprint=_data_fingerprint(rows),
        validation_rows=len(rows),
    )


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

    Reduced precision applies only to scoring. Sufficient-statistic accumulation and mixture
    normalization remain float64. Every reduced score call is compared with float64 at runtime; a
    subtree that stops meeting the absolute error tolerance is permanently downgraded to float64 in
    the supplied plan.

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
    rows = list(data)
    if not rows:
        raise ValueError("mixed_precision_fit requires at least one observation")
    if isinstance(max_its, bool) or not isinstance(max_its, (int, np.integer)) or int(max_its) < 1:
        raise ValueError("max_its must be a positive integer")
    if delta is not None and (not np.isfinite(delta) or float(delta) < 0.0):
        raise ValueError("delta must be None or a finite non-negative number")

    tname = type(model).__name__
    if plan is None:
        plan = recommend_tree_precision(model, rows)
    elif plan.root_type != tname:
        raise ValueError(f"precision plan is for {plan.root_type}, not {tname}")
    elif plan.validation_fingerprint != _data_fingerprint(rows):
        raise ValueError("precision plan was validated on different data")

    if tname == "MixtureDistribution":
        return _mixed_precision_fit_mixture(model, rows, plan, int(max_its), delta, weights)
    if tname == "CompositeDistribution":
        return _mixed_precision_fit_composite(model, rows, plan, int(max_its), delta, weights)
    # Nothing to split at the top level -- fall back to the ordinary (float64) fit for correctness.
    from mixle.inference.estimation import optimize

    return optimize(rows, model.estimator(), prev_estimate=model, max_its=int(max_its), delta=delta, out=None)


def _child_score(child: Any, enc: Any, compute_dtype: Any) -> tuple[np.ndarray, Any, np.ndarray]:
    """Score one executable subtree, dynamically checking any reduced execution."""
    from mixle.stats.compute.fused_codegen import fused_seq_log_density, fusible

    reduced = compute_dtype is not None and np.dtype(compute_dtype) != np.float64
    if not fusible(child):
        score = np.asarray(child.seq_log_density(enc), dtype=np.float64)
        return score, np.float64, np.zeros(score.shape, dtype=np.float64)

    score64 = np.asarray(fused_seq_log_density(child, enc, compute_dtype=None), dtype=np.float64)
    if not reduced:
        return score64, np.float64, np.zeros(score64.shape, dtype=np.float64)
    score32 = np.asarray(fused_seq_log_density(child, enc, compute_dtype=np.float32), dtype=np.float64)
    if score32.shape != score64.shape:
        return score64, np.float64, np.zeros(score64.shape, dtype=np.float64)
    invalid = np.isnan(score32) | np.isnan(score64) | np.isposinf(score32) | np.isposinf(score64)
    if np.any(invalid) or not np.array_equal(np.isneginf(score32), np.isneginf(score64)):
        return score64, np.float64, np.zeros(score64.shape, dtype=np.float64)
    error = np.zeros(score64.shape, dtype=np.float64)
    finite = np.isfinite(score64)
    error[finite] = np.abs(score32[finite] - score64[finite])
    if np.any(error > FUSED_FP32_ABS_LOG_TOLERANCE):
        return score64, np.float64, np.zeros(score64.shape, dtype=np.float64)
    return score32, np.float32, error


def _child_accumulate(child: Any, enc: Any, w: np.ndarray) -> Any:
    """Accumulate sufficient statistics in float64 regardless of scoring precision."""
    from mixle.stats.compute.fused_codegen import fused_accumulate, fusible

    if fusible(child):
        return fused_accumulate(child, enc, w, compute_dtype=None)
    acc = child.accumulator_factory().make()
    acc.seq_update(enc, w, child)
    return acc.value()


def _validated_weights(n: int, weights: np.ndarray | None) -> np.ndarray:
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0.0) or not float(np.sum(w)) > 0.0:
        raise ValueError("weights must be a finite non-negative vector aligned with data and have positive mass")
    return w


def _update_runtime_plan(
    plan: TreePrecisionPlan,
    paths: list[Path],
    dtypes: list[Any],
    errors: list[np.ndarray],
    root_kind: str,
) -> None:
    for path, dtype, error in zip(paths, dtypes, errors):
        bound = float(np.sum(error))
        rationale = (
            f"runtime-validated max absolute row-score error {float(np.max(error)) if error.size else 0.0:.3g}"
            if np.dtype(dtype) == np.float32
            else "runtime validation selected float64"
        )
        _assign_executed_subtree(plan.nodes, path, dtype, rationale, bound, len(error))
    stacked = np.stack(errors, axis=1)
    root_error = np.max(stacked, axis=1) if root_kind == "MixtureDistribution" else np.sum(stacked, axis=1)
    root = plan.nodes[()]
    plan.nodes[()] = replace(
        root,
        compute_dtype=np.float64,
        rationale="runtime child scores combined in float64",
        abs_log_error_bound=float(np.sum(root_error)),
        validation_rows=len(root_error),
        execution_scope="float64_combine",
    )


def _mixed_precision_fit_mixture(
    model: Any,
    data: Any,
    plan: TreePrecisionPlan,
    max_its: int,
    delta: float | None,
    weights: np.ndarray | None,
) -> Any:
    n = len(data)
    w = _validated_weights(n, weights)
    K = model.num_components
    encs = [model.components[i].dist_to_encoder().seq_encode(data) for i in range(K)]
    paths = [("components", str(i)) for i in range(K)]
    dtypes = [plan.dtype_for(path) for path in paths]

    components = list(model.components)
    mixture_weights = np.asarray(model.w, dtype=np.float64)
    if (
        mixture_weights.shape != (K,)
        or not np.all(np.isfinite(mixture_weights))
        or np.any(mixture_weights < 0.0)
        or not float(np.sum(mixture_weights)) > 0.0
    ):
        raise ValueError("mixture weights must be a finite non-negative vector with positive mass")
    mixture_weights = mixture_weights / np.sum(mixture_weights)
    with np.errstate(divide="ignore"):
        log_w = np.log(mixture_weights)
    prev_total_ll: float | None = None

    for _ in range(max_its):
        ll_mat = np.full((n, K), -np.inf, dtype=np.float64)
        score_errors: list[np.ndarray] = []
        for i in range(K):
            score, actual_dtype, error = _child_score(components[i], encs[i], dtypes[i])
            if score.shape != (n,) or error.shape != (n,):
                raise ValueError(f"mixture component {i} scorer did not return one aligned score per observation")
            dtypes[i] = actual_dtype
            score_errors.append(error)
            ll_mat[:, i] = score + log_w[i]

        if np.any(np.isnan(ll_mat)) or np.any(np.isposinf(ll_mat)):
            raise ValueError("mixture component scoring produced NaN or +inf")
        ll_max = ll_mat.max(axis=1, keepdims=True)
        impossible = np.isneginf(ll_max[:, 0])
        rejected = np.flatnonzero(impossible & (w > 0.0))
        if rejected.size:
            preview = rejected[:8].tolist()
            raise ValueError(f"mixture assigns zero probability to positive-weight observations at rows {preview}")
        valid = ~impossible
        resp = np.zeros((n, K), dtype=np.float64)
        log_norm = np.zeros(n, dtype=np.float64)
        if np.any(valid):
            shifted = ll_mat[valid] - ll_max[valid]
            shifted = np.exp(shifted)
            row_sum = shifted.sum(axis=1, keepdims=True)
            if np.any(~np.isfinite(row_sum)) or np.any(row_sum <= 0.0):
                raise ValueError("mixture responsibility normalization failed")
            resp[valid] = shifted * (w[valid, None] / row_sum)
            log_norm[valid] = ll_max[valid, 0] + np.log(row_sum[:, 0])
        total_ll = float(np.dot(w[valid], log_norm[valid]))
        _update_runtime_plan(plan, paths, dtypes, score_errors, "MixtureDistribution")

        comp_counts = resp.sum(axis=0)
        suff_stats = [_child_accumulate(components[i], encs[i], resp[:, i]) for i in range(K)]

        estimators = model.estimator().estimators
        components = [estimators[i].estimate(comp_counts[i], suff_stats[i]) for i in range(K)]
        total = comp_counts.sum()
        new_w = comp_counts / total if total > 0 else np.asarray(model.w, dtype=np.float64)
        with np.errstate(divide="ignore"):
            log_w = np.log(new_w)

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
    w = _validated_weights(n, weights)
    m = len(model.dists)
    # Composite factors observe x[i] of each tuple observation.
    try:
        factor_data = [[x[i] for x in data] for i in range(m)]
    except (IndexError, TypeError) as exc:
        raise ValueError("composite training rows do not match the model's factors") from exc
    encs = [model.dists[i].dist_to_encoder().seq_encode(factor_data[i]) for i in range(m)]
    paths = [("dists", str(i)) for i in range(m)]
    dtypes = [plan.dtype_for(path) for path in paths]

    dists = list(model.dists)
    prev_total_ll: float | None = None

    for _ in range(max_its):
        total_ll = 0.0
        suff_stats = []
        score_errors: list[np.ndarray] = []
        for i in range(m):
            score, actual_dtype, error = _child_score(dists[i], encs[i], dtypes[i])
            if score.shape != (n,) or error.shape != (n,):
                raise ValueError(f"composite factor {i} scorer did not return one aligned score per observation")
            dtypes[i] = actual_dtype
            score_errors.append(error)
            if np.any(np.isnan(score)) or np.any(np.isposinf(score)):
                raise ValueError(f"composite factor {i} scoring produced NaN or +inf")
            impossible = np.isneginf(score)
            rejected = np.flatnonzero(impossible & (w > 0.0))
            if rejected.size:
                raise ValueError(
                    f"composite factor {i} assigns zero probability to positive-weight observations "
                    f"at rows {rejected[:8].tolist()}"
                )
            valid = ~impossible
            total_ll += float(np.dot(w[valid], score[valid]))
            suff_stats.append(_child_accumulate(dists[i], encs[i], w))
        _update_runtime_plan(plan, paths, dtypes, score_errors, "CompositeDistribution")

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
