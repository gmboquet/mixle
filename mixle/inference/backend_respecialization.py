"""Backend re-specialization mid-fit -- learned-scheduler-ready decision logic + execution
machinery for swapping a NODE's execution backend in response to structural changes observed
during a fit (workstream D6).

Frame (see the ConditionalJIT track, D1-D6): **the estimator tree is an IR**. D1
(:mod:`mixle.inference.node_report`) instruments every node with an ``update_kind``
classification and an E/M cost proxy. D2 (:mod:`mixle.inference.freeze_rollup`) freezes subtrees
that stop moving. D3 (:mod:`mixle.inference.block_em`) schedules WHICH blocks get updated each
round, so a node's "hot" (updated every round) vs. "cold" (rarely scheduled) status is a live,
observable signal DURING a fit, not a static property. K1's per-node precision plan
(:mod:`mixle.inference.precision_plan`, and the wider per-node walk on the ``per-node-precision``
branch) reports when a node's safe compute precision drops. Every one of these is a moment where
the OPTIMAL execution BACKEND for a node -- eager vs. ``torch.compile``-d, full vs. reduced
precision, computed-on-the-fly vs. table-cached -- might also change.

Correctness backbone (unchanged from the rest of the D-track): re-specialization is a SCHEDULING/
EXECUTION optimization only. It never changes what a node computes -- :func:`NodeBackend.__call__`
must return the tolerance-equal value regardless of which backend is currently active (see
``mixle.tests.backend_respecialization_test``'s tolerance-equal test) -- only how fast it computes
it. Any interleaving of these backend swaps with the block-EM schedule (D3) or freeze/roll-up
caching (D2) is still coordinate ascent on the SAME Neal-Hinton free energy F the rest of the
track climbs; F itself never notices which backend answered a query.

Two things live here:

1. **Compile economics** (:func:`estimate_compile_cost` / :func:`estimate_compile_benefit` /
   :func:`estimate_table_cost` / :func:`estimate_table_benefit`, and the trigger functions built
   on top of them) -- a real cost/benefit tradeoff, not a fixed rule: an upfront re-specialization
   cost is only worth paying when it is amortized over enough EXPECTED remaining executions of the
   node at its current hot/frozen/precision status. :class:`RespecializationDecision` is the
   inspectable receipt of that tradeoff -- the "compile economics exposed to D5" interface the
   roadmap calls for: a later learned controller (D5, not built here) could plausibly consume its
   ``estimated_cost`` / ``estimated_benefit`` / ``net_benefit`` fields as training features and
   ``action`` as a label, without needing to re-derive the economics itself.

2. **Execution** (:class:`NodeBackend`, :class:`DensityTable`, :func:`compile_forward`) -- actually
   APPLIES a chosen re-specialization: wraps a node's forward call in ``torch.compile`` (reusing
   :meth:`mixle.engines.torch_engine.TorchEngine.compile`'s exact convention -- the SAME
   ``compile_enabled and hasattr(torch, "compile")`` gate, not a parallel one), or swaps in a
   precomputed density-table lookup for repeated identical/near-identical queries. This is a real
   mechanism with a measurable effect, not just a recommendation (see the acceptance tests).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from mixle.inference.node_report import NodeReport

# ---------------------------------------------------------------------------------------------
# Compile-economics constants. Proxy units, not wall-clock -- same convention D1's node_report
# module documents for its own E/M cost proxy (see that module's docstring): callers who have a
# real measured per-call cost should pass it explicitly rather than rely on these defaults.
# ---------------------------------------------------------------------------------------------

_DEFAULT_COMPILE_FIXED_OVERHEAD = 200.0
_DEFAULT_COMPILE_PER_PARAM_OVERHEAD = 50.0
_DEFAULT_COMPILE_SPEEDUP_FACTOR = 4.0

_DEFAULT_TABLE_FIXED_OVERHEAD = 20.0
_DEFAULT_TABLE_PER_POINT_OVERHEAD = 1.0
_DEFAULT_TABLE_SPEEDUP_FACTOR = 8.0

_DEFAULT_HOT_ACTIVATION_RATIO = 0.7  # a node scheduled active >=70% of recent rounds counts "hot"
_DEFAULT_STABLE_Q_GAIN_TOL = 1.0e-6  # reuses D2/D3's own convergence-tolerance convention


class RespecializationAction(StrEnum):
    """The backend choice a :class:`RespecializationDecision` recommends (and :class:`NodeBackend`
    can actually apply)."""

    NONE = "none"
    COMPILE = "compile"
    FUSE = "fuse"
    REDUCE_PRECISION = "reduce_precision"
    DENSITY_TABLE = "density_table"


@dataclass(frozen=True)
class RespecializationDecision:
    """The cost/benefit tradeoff and chosen action for ONE node's backend -- the D5-facing
    receipt.

    This is deliberately a flat, inspectable dataclass (not just a boolean): ``estimated_cost``,
    ``estimated_benefit``, and ``expected_remaining_calls`` are exactly the features a later
    learned controller (D5, out of scope here) would want as training signal, and ``action`` /
    ``triggered_by`` double as the label and the (attributable) reason. Nothing in this dataclass
    is invented for D5's benefit alone -- every field is also what THIS module's own decision
    functions compute and act on today.
    """

    field_path: str
    node_type: str
    action: RespecializationAction
    triggered_by: str
    estimated_cost: float
    estimated_benefit: float
    expected_remaining_calls: float
    rationale: str

    @property
    def net_benefit(self) -> float:
        """``estimated_benefit - estimated_cost`` -- positive iff re-specializing is worth it."""
        return self.estimated_benefit - self.estimated_cost

    @property
    def worth_it(self) -> bool:
        return self.action != RespecializationAction.NONE and self.net_benefit > 0.0


def _per_call_cost(report: NodeReport, per_call_cost: float | None) -> float:
    if per_call_cost is not None:
        return max(float(per_call_cost), 0.0)
    return max(report.e_step_cost + report.m_step_cost, 0.0)


def estimate_compile_cost(
    report: NodeReport,
    *,
    fixed_overhead: float = _DEFAULT_COMPILE_FIXED_OVERHEAD,
    per_param_overhead: float = _DEFAULT_COMPILE_PER_PARAM_OVERHEAD,
) -> float:
    """Estimate the upfront cost of ``torch.compile``-ing ``report``'s node: a fixed
    graph-capture/tracing overhead plus a per-parameter term (more parameters -> a bigger graph to
    trace and specialize), mirroring D1's own ``param_count``-proxy convention rather than
    inventing a new cost unit.
    """
    return float(fixed_overhead) + float(per_param_overhead) * float(report.param_count)


def estimate_compile_benefit(
    report: NodeReport,
    expected_remaining_calls: float,
    *,
    per_call_cost: float | None = None,
    speedup_factor: float = _DEFAULT_COMPILE_SPEEDUP_FACTOR,
) -> float:
    """Estimate the total cost SAVED by compiling ``report``'s node, amortized over
    ``expected_remaining_calls`` more executions at its current per-call cost (``per_call_cost``,
    or D1's own E/M cost proxy when not supplied). A real cost-benefit tradeoff, not a fixed rule:
    a node executed many more times pays back a fixed compile overhead; a node executed only a
    handful more times does not (see the decision-boundary test in
    ``mixle.tests.backend_respecialization_test``).
    """
    base = _per_call_cost(report, per_call_cost)
    speedup_factor = max(float(speedup_factor), 1.0)
    saved_per_call = base * (1.0 - 1.0 / speedup_factor)
    return max(saved_per_call, 0.0) * max(float(expected_remaining_calls), 0.0)


def estimate_table_cost(
    n_query_points: int,
    *,
    fixed_overhead: float = _DEFAULT_TABLE_FIXED_OVERHEAD,
    per_point_overhead: float = _DEFAULT_TABLE_PER_POINT_OVERHEAD,
) -> float:
    """Estimate the upfront cost of building a density table over ``n_query_points`` seed
    points: a fixed bookkeeping overhead plus one evaluation per seed point."""
    return float(fixed_overhead) + float(per_point_overhead) * max(int(n_query_points), 0)


def estimate_table_benefit(
    report: NodeReport,
    expected_remaining_calls: float,
    *,
    per_call_cost: float | None = None,
    speedup_factor: float = _DEFAULT_TABLE_SPEEDUP_FACTOR,
) -> float:
    """Estimate the total cost saved by serving ``report``'s node from a precomputed density
    table instead of computing on the fly -- same amortization shape as
    :func:`estimate_compile_benefit`, with a table's own (typically larger, since a dict lookup is
    cheaper than a re-traced graph call) default speedup factor."""
    base = _per_call_cost(report, per_call_cost)
    speedup_factor = max(float(speedup_factor), 1.0)
    saved_per_call = base * (1.0 - 1.0 / speedup_factor)
    return max(saved_per_call, 0.0) * max(float(expected_remaining_calls), 0.0)


# ---------------------------------------------------------------------------------------------
# Triggers -- hook into D2/D3/K1's structural-change signals and decide, via the economics above,
# whether re-specializing is worth it.
# ---------------------------------------------------------------------------------------------


def decide_hot_compile(
    report: NodeReport,
    *,
    activation_ratio: float,
    expected_remaining_calls: float,
    already_compiled: bool = False,
    hot_threshold: float = _DEFAULT_HOT_ACTIVATION_RATIO,
    per_call_cost: float | None = None,
    speedup_factor: float = _DEFAULT_COMPILE_SPEEDUP_FACTOR,
) -> RespecializationDecision:
    """Decide whether ``report``'s node should be compiled because D3's scheduler has been
    running it "hot" (active in ``activation_ratio`` fraction of recent rounds, >= ``hot_threshold``).

    A frozen node (D2/D1's own ``update_kind``) is never a compile candidate regardless of a
    stale activation ratio -- it is not going to run again. An already-compiled node is left
    alone (no double-compile).
    """
    cost = estimate_compile_cost(report)
    benefit = estimate_compile_benefit(
        report, expected_remaining_calls, per_call_cost=per_call_cost, speedup_factor=speedup_factor
    )
    if report.update_kind == "frozen":
        action, rationale = RespecializationAction.NONE, "node is frozen -- will not execute again, never compile"
    elif already_compiled:
        action, rationale = RespecializationAction.NONE, "already compiled -- nothing to do"
    elif activation_ratio < hot_threshold:
        action, rationale = (
            RespecializationAction.NONE,
            "activation_ratio %.2f < hot_threshold %.2f -- not hot enough to bother"
            % (activation_ratio, hot_threshold),
        )
    elif benefit > cost:
        action, rationale = (
            RespecializationAction.COMPILE,
            "hot node (activation_ratio %.2f): benefit %.1f > compile cost %.1f over %.1f expected calls"
            % (activation_ratio, benefit, cost, expected_remaining_calls),
        )
    else:
        action, rationale = (
            RespecializationAction.NONE,
            "hot but too few expected_remaining_calls (%.1f): compile cost %.1f >= benefit %.1f"
            % (expected_remaining_calls, cost, benefit),
        )
    return RespecializationDecision(
        field_path=report.field_path,
        node_type=report.node_type,
        action=action,
        triggered_by="hot",
        estimated_cost=cost,
        estimated_benefit=benefit,
        expected_remaining_calls=float(expected_remaining_calls),
        rationale=rationale,
    )


def decide_frozen_precision_drop(
    report: NodeReport,
    *,
    q_gain_tol: float = _DEFAULT_STABLE_Q_GAIN_TOL,
    already_reduced: bool = False,
) -> RespecializationDecision:
    """Decide whether a newly-frozen/near-converged node (D2's freeze signal, or D1's own
    near-zero Q-gain) should drop to reduced precision -- the K1 "precision drops" trigger. Unlike
    compile/table decisions this one has no amortization term: a frozen node's remaining work is
    (by definition) near zero, so the ONLY real cost is the negligible act of re-tagging its
    compute dtype, and the benefit is every future (typically read-only/health-check) touch of the
    node running cheaper -- so this is a near-free action whenever the trigger fires at all.
    """
    is_frozen = report.update_kind == "frozen"
    is_converged = report.q_gain is not None and abs(report.q_gain) < q_gain_tol
    cost = 1.0  # re-tagging dtype is O(1) -- no graph to retrace, no table to build
    if already_reduced:
        action, rationale = RespecializationAction.NONE, "already at reduced precision"
        benefit = 0.0
    elif is_frozen:
        action, rationale = RespecializationAction.REDUCE_PRECISION, "node is frozen -- safe to drop precision"
        benefit = 10.0
    elif is_converged:
        action, rationale = (
            RespecializationAction.REDUCE_PRECISION,
            "q_gain %.3e < tol %.3e -- converged, safe to drop precision" % (report.q_gain, q_gain_tol),
        )
        benefit = 10.0
    else:
        action, rationale = RespecializationAction.NONE, "node is still moving -- keep full precision"
        benefit = 0.0
    return RespecializationDecision(
        field_path=report.field_path,
        node_type=report.node_type,
        action=action,
        triggered_by="frozen_or_converged",
        estimated_cost=cost,
        estimated_benefit=benefit,
        expected_remaining_calls=0.0,
        rationale=rationale,
    )


def decide_density_table(
    report: NodeReport,
    *,
    expected_remaining_calls: float,
    n_query_points: int,
    structure_stable: bool,
    per_call_cost: float | None = None,
    speedup_factor: float = _DEFAULT_TABLE_SPEEDUP_FACTOR,
) -> RespecializationDecision:
    """Decide whether a fully closed-form node (``update_kind`` in ``{"closed_form",
    "conjugate_closed_form"}`` -- no gradient loop, nothing to compile) whose surrounding tree
    structure has stabilized (``structure_stable``, e.g. D3's scheduler has stopped changing which
    blocks are active) is worth precomputing a density table for.
    """
    cost = estimate_table_cost(n_query_points)
    benefit = estimate_table_benefit(
        report, expected_remaining_calls, per_call_cost=per_call_cost, speedup_factor=speedup_factor
    )
    closed_form = report.update_kind in ("closed_form", "conjugate_closed_form")
    if not closed_form:
        action, rationale = (
            RespecializationAction.NONE,
            "%s update kind has no closed-form density to table-cache" % report.update_kind,
        )
    elif not structure_stable:
        action, rationale = RespecializationAction.NONE, "tree structure not yet stable -- table would thrash"
    elif benefit > cost:
        action, rationale = (
            RespecializationAction.DENSITY_TABLE,
            "closed-form + stable structure: benefit %.1f > table-build cost %.1f" % (benefit, cost),
        )
    else:
        action, rationale = (
            RespecializationAction.NONE,
            "too few expected_remaining_calls (%.1f): table cost %.1f >= benefit %.1f"
            % (expected_remaining_calls, cost, benefit),
        )
    return RespecializationDecision(
        field_path=report.field_path,
        node_type=report.node_type,
        action=action,
        triggered_by="stable_closed_form",
        estimated_cost=cost,
        estimated_benefit=benefit,
        expected_remaining_calls=float(expected_remaining_calls),
        rationale=rationale,
    )


# ---------------------------------------------------------------------------------------------
# Execution -- actually apply the chosen re-specialization.
# ---------------------------------------------------------------------------------------------


def _to_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion of a query point (numpy array, python scalar, or torch tensor) to a
    numpy array, for use as a density-table cache key."""
    to_numpy = getattr(x, "detach", None)
    if callable(to_numpy):
        x = x.detach()
        cpu = getattr(x, "cpu", None)
        x = cpu() if callable(cpu) else x
        numpy_fn = getattr(x, "numpy", None)
        if callable(numpy_fn):
            return np.asarray(numpy_fn())
    return np.asarray(x)


def _default_torch_engine() -> Any:
    """Build the SAME kind of compile-enabled engine :mod:`mixle.engines.torch_engine` already
    exposes, rather than inventing a parallel compile convention."""
    from mixle.engines.torch_engine import TorchEngine

    return TorchEngine(compile=True)


def compile_forward(fn: Callable[..., Any], engine: Any | None = None) -> Callable[..., Any]:
    """Wrap ``fn`` with ``torch.compile`` via a compile-enabled engine's own ``.compile`` method
    (:meth:`mixle.engines.torch_engine.TorchEngine.compile`) -- reuses that method's exact
    ``compile_enabled and hasattr(torch, "compile")`` gate so this module never has a second,
    possibly-divergent opinion about when compilation is available."""
    engine = engine if engine is not None else _default_torch_engine()
    compile_method = getattr(engine, "compile", None)
    if callable(compile_method):
        return compile_method(fn)
    return fn


class DensityTable:
    """A precomputed cache of a closed-form node's density/forward function, keyed by a quantized
    input -- a real memoization mechanism for repeated identical/near-identical query patterns
    (see the D6 module docstring). ``quantum`` controls how near "near-identical" is: inputs whose
    quantized representation matches an existing key are served from the table; anything else
    falls back to (and populates the table with) the underlying function.
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        seed_points: list[Any] | None = None,
        *,
        quantum: float = 1.0e-9,
    ) -> None:
        self._fn = fn
        self._quantum = float(quantum)
        self._cache: dict[tuple[int, ...], Any] = {}
        for point in seed_points or []:
            self._cache[self._key(point)] = fn(point)

    def _key(self, x: Any) -> tuple[int, ...]:
        arr = _to_numpy(x).astype(np.float64, copy=False)
        return tuple(np.round(arr / self._quantum).astype(np.int64).ravel().tolist())

    def __len__(self) -> int:
        return len(self._cache)

    def lookup(self, x: Any) -> Any:
        """Return the cached value for ``x`` if a (quantized) match exists, else compute it via
        the underlying function and cache it for next time."""
        key = self._key(x)
        if key in self._cache:
            return self._cache[key]
        value = self._fn(x)
        self._cache[key] = value
        return value


class NodeBackend:
    """Holds and executes the CURRENTLY CHOSEN backend for one node's forward call.

    Wraps an eager ``forward`` callable (defaulting to ``dist.kernel(engine=...).score``, the same
    engine-aware evaluation kernel the rest of the codebase already uses -- see
    :mod:`mixle.stats.compute.kernel`) and lets :meth:`apply` swap in a compiled or table-cached
    variant per a :class:`RespecializationDecision`, while ``__call__`` always dispatches to
    whichever backend is currently active. Every backend must return the tolerance-equal value for
    the same input -- re-specializing changes only HOW the value is computed (see the module
    docstring's correctness backbone).
    """

    def __init__(
        self,
        dist: Any,
        forward: Callable[[Any], Any] | None = None,
        *,
        engine: Any | None = None,
    ) -> None:
        self.dist = dist
        self.engine = engine
        self._eager_forward = forward if forward is not None else self._default_forward
        self._compiled_forward: Callable[[Any], Any] | None = None
        self._table: DensityTable | None = None
        self.action: RespecializationAction = RespecializationAction.NONE
        self.decision: RespecializationDecision | None = None

    def _default_forward(self, enc: Any) -> Any:
        return self.dist.kernel(engine=self.engine).score(enc)

    def apply(self, decision: RespecializationDecision, *, table_seed_points: list[Any] | None = None) -> None:
        """Apply ``decision``'s chosen action, actually building whatever machinery it implies."""
        self.decision = decision
        if decision.action == RespecializationAction.COMPILE:
            self._compiled_forward = compile_forward(self._eager_forward, engine=self.engine)
            self.action = RespecializationAction.COMPILE
        elif decision.action == RespecializationAction.DENSITY_TABLE:
            self._table = DensityTable(self._eager_forward, table_seed_points)
            self.action = RespecializationAction.DENSITY_TABLE
        elif decision.action in (RespecializationAction.FUSE, RespecializationAction.REDUCE_PRECISION):
            # Recorded as the chosen action for D5's benefit, but re-fusing multiple adjacent
            # nodes into one compiled unit, and swapping a node's own compute dtype, are both
            # decisions that must be executed by the CALLER (which owns the tree/estimator and can
            # rebuild the relevant subtree or hand it a new engine) -- see the module docstring's
            # "compile economics exposed to D5" note; this class documents rather than hides that.
            self.action = decision.action
        else:
            self.action = RespecializationAction.NONE

    def __call__(self, enc: Any) -> Any:
        if self.action == RespecializationAction.COMPILE and self._compiled_forward is not None:
            return self._compiled_forward(enc)
        if self.action == RespecializationAction.DENSITY_TABLE and self._table is not None:
            return self._table.lookup(enc)
        return self._eager_forward(enc)
