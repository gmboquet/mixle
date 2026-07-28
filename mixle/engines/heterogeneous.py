"""Precision-aware planning for distributed EM across heterogeneous compute.

Large worker pools are rarely uniform: some workers may have GPU tensor cores,
while others are CPU-only or accuracy-oriented. This module chooses, per
worker, how many E-step rows to assign and which precision band to run. The
selected precision is the fastest supported band that still satisfies the
requested error budget. When NO available precision can satisfy that budget
for some worker, planning fails closed by default -- it raises
:class:`InfeasiblePrecisionError` quoting the quantified gap -- rather than
silently returning a plan that violates the accuracy it claims to hold; pass
``allow_infeasible=True`` for an explicit best-effort plan instead (see
:meth:`HeterogeneousPlan.is_feasible`).

Meeting a budget and *certifying* it are different claims, and this module
makes only the first: ``meets_target`` says the chosen format's gamma-n
roundoff score clears the target at the plan's ``op_count``, whereas
:meth:`HeterogeneousPlan.is_certified` is ``False`` for every constrained plan
because the planner has no operation graph, magnitude/conditioning evidence,
reduction order, or communication-topology certificate with which to prove a
scientific target. A caller wanting numbers ranked need not opt out of
anything; a caller wanting a proof will not find one here.

The plan also sizes the 4-way (radix-4) reduction depth so fixed-size
sufficient-statistic payloads fold in ``O(log W)`` instead of a single-root
fan-in.

This module is the pure-Python planning layer. Spark, MPI, or other distributed
dispatchers consume the returned plan from the inference layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

from mixle.engines.affine import UNIT_ROUNDOFF

# Relative throughput multipliers per ``(device, precision)`` for planning.
# Lower precision is faster on GPUs; sub-float32 arithmetic is slower on CPUs
# without native support, and double-double arithmetic is much more expensive.
# This is also the closed set of (device, precision) pairs the planner knows how
# to model at all -- a Worker declaring a pair outside this table is rejected
# at construction rather than silently defaulting to a made-up throughput.
_THROUGHPUT = {
    ("gpu", "fp8"): 4.0,
    ("gpu", "bfloat16"): 2.5,
    ("gpu", "float16"): 2.5,
    ("gpu", "float32"): 1.5,
    ("gpu", "float64"): 1.0,
    ("cpu", "float32"): 1.4,
    ("cpu", "float64"): 1.0,
    ("cpu", "dd"): 1.0 / 15.0,
}

_KNOWN_DEVICES = frozenset(device for device, _ in _THROUGHPUT)


class InfeasiblePrecisionError(ValueError):
    """Raised by :func:`plan_heterogeneous` when some worker cannot satisfy ``target_rel_error`` at any
    allowed precision and the caller did not opt into a best-effort plan via ``allow_infeasible=True``."""


def _require_nonneg_int(value: int, name: str) -> None:
    """Raise unless ``value`` is a plain, nonnegative Python ``int`` (booleans excluded)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a nonnegative int, got %r" % (name, value))


@dataclass(frozen=True)
class Worker:
    """A compute worker: its device, the precisions it can run (any order), and a base throughput.

    Validated at construction so a malformed worker can never silently reach planning: ``device`` must be
    a known device, ``precisions`` must be nonempty, every declared precision must be a supported
    ``(device, precision)`` pair (present in :data:`_THROUGHPUT` -- planning has no throughput model for
    an unlisted pair and used to silently default to an arbitrary 0.5x instead of rejecting it), and
    ``base_throughput`` must be finite and strictly positive (a zero or negative throughput could
    previously divide-by-zero or invert the load balancer's row split; see MXR-080-0134).
    """

    name: str
    device: str  # "cpu" or "gpu"
    precisions: tuple[str, ...]
    base_throughput: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("worker name must be a nonempty string")
        if isinstance(self.precisions, (str, bytes)):
            raise ValueError("worker precisions must be a nonempty sequence of precision names")
        canonical_precisions = tuple(self.precisions)
        object.__setattr__(self, "precisions", canonical_precisions)
        if self.device not in _KNOWN_DEVICES:
            raise ValueError(
                "worker %r has unknown device %r; expected one of %s" % (self.name, self.device, sorted(_KNOWN_DEVICES))
            )
        if not self.precisions:
            raise ValueError("worker %r declares an empty precision set" % (self.name,))
        unsupported = [p for p in self.precisions if (self.device, p) not in _THROUGHPUT]
        if unsupported:
            raise ValueError(
                "worker %r declares unsupported (device, precision) pairs: %s (known precisions for %r: %s)"
                % (
                    self.name,
                    [(self.device, p) for p in unsupported],
                    self.device,
                    sorted(p for d, p in _THROUGHPUT if d == self.device),
                )
            )
        if (
            isinstance(self.base_throughput, bool)
            or not isinstance(self.base_throughput, (int, float))
            or not math.isfinite(self.base_throughput)
            or self.base_throughput <= 0
        ):
            raise ValueError(
                "worker %r base_throughput must be finite and positive, got %r" % (self.name, self.base_throughput)
            )
        object.__setattr__(self, "base_throughput", float(self.base_throughput))


@dataclass(frozen=True)
class WorkerAssignment:
    """One worker's row allocation, precision, and effective throughput.

    ``meets_target`` is a statement about numbers: whether ``precision``'s modeled roundoff at the plan's
    ``op_count`` actually lands at or below the plan's ``target_rel_error`` (see
    :func:`plan_heterogeneous`), quantified by ``achieved_rel_error`` (``None`` when the format's roundoff
    isn't modeled, e.g. ``fp8`` -- which is exactly why such a format can never *meet* a target). Both
    default to the trivially-satisfied case so callers that never pass a ``target_rel_error`` see
    unconstrained, always-``True`` assignments.

    Whether the planner can *certify* that number is a separate question, answered by ``evidence_kind``
    and :attr:`is_certified`: a ``"heuristic"`` assignment may legitimately report ``meets_target=True``
    while remaining uncertified.
    """

    name: str
    rows: int
    precision: str
    effective_throughput: float
    meets_target: bool = True
    achieved_rel_error: float | None = None
    evidence_kind: str = "unconstrained"

    def __post_init__(self) -> None:
        # Output-side sanity check, independent of plan_heterogeneous's own input validation: a
        # WorkerAssignment can never represent a negative/fractional row count or a non-positive
        # effective throughput, regardless of how it was constructed (MXR-080-0134).
        _require_nonneg_int(self.rows, "rows")
        if self.precision not in {precision for _, precision in _THROUGHPUT}:
            raise ValueError("precision %r is not a modeled heterogeneous precision" % self.precision)
        if not isinstance(self.meets_target, bool):
            raise ValueError("meets_target must be an actual Boolean")
        if (
            isinstance(self.effective_throughput, bool)
            or not isinstance(self.effective_throughput, (int, float))
            or not math.isfinite(self.effective_throughput)
            or self.effective_throughput <= 0
        ):
            raise ValueError("effective_throughput must be finite and positive, got %r" % (self.effective_throughput,))
        if self.achieved_rel_error is not None and (
            isinstance(self.achieved_rel_error, bool)
            or not math.isfinite(self.achieved_rel_error)
            or self.achieved_rel_error < 0
        ):
            raise ValueError("achieved_rel_error must be None or a finite nonnegative heuristic estimate")
        if self.evidence_kind not in {"unconstrained", "heuristic"}:
            raise ValueError("evidence_kind must be 'unconstrained' or 'heuristic'")
        if self.evidence_kind == "unconstrained":
            if not self.meets_target or self.achieved_rel_error is not None:
                raise ValueError("unconstrained assignments must meet_target=True and carry no error estimate")
        elif self.meets_target and self.achieved_rel_error is None:
            # A heuristic assignment MAY report meets_target=True -- that is a claim about the format's
            # score, not a certificate (see is_certified) -- but never without the number behind it.
            raise ValueError("a heuristic assignment cannot claim meets_target=True with no achieved_rel_error")

    @property
    def is_certified(self) -> bool:
        """Whether ``meets_target`` rests on an accuracy certificate rather than a ranking heuristic.

        ``True`` only for ``evidence_kind='unconstrained'``, where there is no target to certify. For a
        ``"heuristic"`` assignment this is always ``False``, including when ``meets_target`` is ``True``:
        the gamma-n score ranks formats, but the planner holds no operation graph, magnitude/conditioning
        evidence, reduction order, or communication-topology certificate with which to prove a scientific
        target.
        """
        return self.evidence_kind == "unconstrained"


def _reduce_depth(n_workers: int) -> int:
    """Depth of the planner's 4-way (radix-4) worker-reduction tree: ``ceil(log4(W))``, 0 for one worker.

    Deliberately radix-4, not binary: the payloads folded here are fixed-size sufficient statistics, so a
    wider node keeps the tree shallow (16 workers fold in 2 rounds, not 4) while still avoiding the
    single-root fan-in. Callers that actually execute the fold pick their own arity -- see
    ``mixle.inference.heterogeneous_executor.tree_reduce_values(..., branch=...)``; this number is the
    plan's advertised radix-4 depth, so read it as ``ceil(log4(W))`` and not as a binary tree height.
    """
    return 0 if n_workers <= 1 else math.ceil(math.log2(n_workers) / 2)


@dataclass(frozen=True)
class HeterogeneousPlan:
    """Assignments and reduction depth for heterogeneous execution."""

    assignments: tuple[WorkerAssignment, ...]
    reduce_depth: int

    def __post_init__(self) -> None:
        canonical = tuple(self.assignments)
        if not canonical:
            raise ValueError("HeterogeneousPlan requires at least one assignment")
        if not all(isinstance(assignment, WorkerAssignment) for assignment in canonical):
            raise ValueError("assignments must contain only WorkerAssignment records")
        object.__setattr__(self, "assignments", canonical)
        _require_nonneg_int(self.reduce_depth, "reduce_depth")
        expected_depth = _reduce_depth(len(canonical))
        if self.reduce_depth != expected_depth:
            raise ValueError(
                "reduce_depth must match the planner's 4-way worker-reduction tree, ceil(log4(W)) "
                "(%d for %d workers)" % (expected_depth, len(canonical))
            )

    def total_rows(self) -> int:
        """Return total rows assigned across workers."""
        return sum(a.rows for a in self.assignments)

    def is_feasible(self) -> bool:
        """Whether every assignment's precision actually meets the plan's requested ``target_rel_error``.

        Always ``True`` for an unconstrained plan (``target_rel_error=None``). Can only be ``False`` when
        ``plan_heterogeneous`` was called with ``allow_infeasible=True`` and at least one worker's best
        available precision still could not satisfy the budget -- inspect :meth:`infeasible_assignments`
        for which workers, and each one's quantified ``achieved_rel_error``. Without that opt-in the
        planner raises rather than returning an infeasible plan, so this is ``True`` for every plan a
        default call returns.

        Feasible is not certified: see :meth:`is_certified`.
        """
        return all(a.meets_target for a in self.assignments)

    def is_certified(self) -> bool:
        """Whether the plan's accuracy claims are certificates rather than format-ranking scores.

        ``True`` only for an unconstrained plan (``target_rel_error=None``), which claims nothing about
        accuracy in the first place. For any constrained plan this is ``False`` even when
        :meth:`is_feasible` is ``True``: the gamma-n score behind ``meets_target`` can rank formats and
        rule options out, but the planner has no operation graph, magnitude/conditioning evidence,
        reduction order, or communication-topology certificate with which to prove a scientific target.
        Supply a workload error certificate elsewhere if the target must be guaranteed.
        """
        return all(a.is_certified for a in self.assignments)

    def infeasible_assignments(self) -> tuple[WorkerAssignment, ...]:
        """The subset of assignments whose precision does not meet the plan's ``target_rel_error``."""
        return tuple(a for a in self.assignments if not a.meets_target)


def _estimated_rel_error(precision: str, op_count: int) -> float | None:
    """A gamma-n roundoff heuristic used only to rank formats; never an accuracy certificate."""
    u = UNIT_ROUNDOFF.get(precision)
    if u is None:
        return None
    product = op_count * u
    return product / (1.0 - product) if product < 1.0 else math.inf


def _meets_budget(precision: str, op_count: int, target_rel_error: float | None) -> bool:
    if target_rel_error is None:
        return True
    achieved = _estimated_rel_error(precision, op_count)
    return achieved is not None and achieved <= target_rel_error  # roundoff accumulates ~op_count * u (relative)


class _PrecisionChoice(NamedTuple):
    """One worker's chosen precision plus whether it actually satisfies the accuracy budget."""

    precision: str
    meets_target: bool
    achieved_rel_error: float | None
    reason: str | None = None  # set when meets_target is False; human-readable, used in the raised error


def _best_precision(
    worker: Worker, allowed: tuple[str, ...], op_count: int, target_rel_error: float | None
) -> _PrecisionChoice:
    """The highest-throughput precision the worker supports that is allowed and meets the accuracy budget.

    When no supported+allowed precision meets ``target_rel_error`` -- including when the worker supports
    none of ``allowed`` at all -- returns the worker's most ACCURATE available precision instead (never a
    merely-fast one: the previous fallback here picked the highest-*throughput* candidate, contradicting
    its own "most accurate" doc comment and silently under-delivering accuracy on top of not reporting the
    shortfall at all; see MXR-080-0133) with ``meets_target=False``, a quantified ``achieved_rel_error``,
    and a human-readable ``reason``. The caller (:func:`plan_heterogeneous`) decides whether that is an
    error or an explicit best-effort assignment.
    """
    supported = [p for p in worker.precisions if p in allowed]
    if not supported:
        fallback = min(worker.precisions, key=lambda p: UNIT_ROUNDOFF.get(p, math.inf))
        return _PrecisionChoice(
            precision=fallback,
            meets_target=False,
            achieved_rel_error=_estimated_rel_error(fallback, op_count),
            reason="worker %r supports none of the allowed precisions %s (worker supports %s)"
            % (worker.name, allowed, worker.precisions),
        )
    candidates = [p for p in supported if _meets_budget(p, op_count, target_rel_error)]
    if candidates:
        best = max(candidates, key=lambda p: _THROUGHPUT[(worker.device, p)])
        return _PrecisionChoice(best, True, _estimated_rel_error(best, op_count))
    # Nothing supported+allowed meets the budget: report the MOST ACCURATE option, i.e. the tightest bound
    # actually achievable, so an infeasible-plan caller sees "how close can you get" rather than an
    # arbitrary (possibly much worse) substitute.
    best = min(supported, key=lambda p: UNIT_ROUNDOFF.get(p, math.inf))
    achieved = _estimated_rel_error(best, op_count)
    return _PrecisionChoice(
        precision=best,
        meets_target=False,
        achieved_rel_error=achieved,
        reason="worker %r cannot meet target_rel_error=%r at op_count=%d: its most accurate allowed "
        "precision (%r) only achieves ~%s relative error" % (worker.name, target_rel_error, op_count, best, achieved),
    )


def plan_heterogeneous(
    workers: list[Worker],
    n_rows: int,
    allowed_precisions: tuple[str, ...] = ("fp8", "bfloat16", "float16", "float32", "float64", "dd"),
    target_rel_error: float | None = None,
    op_count: int = 1000,
    allow_infeasible: bool = False,
) -> HeterogeneousPlan:
    """Assign rows + a precision band to each worker, balanced by precision-adjusted throughput.

    Each worker runs the fastest precision its hardware supports that stays within ``target_rel_error``
    (``None`` = no accuracy constraint); rows are split proportionally to the resulting throughput so all
    workers finish together. ``reduce_depth`` is the 4-way (radix-4) tree depth for folding the
    sufficient-statistic payloads (``ceil(log4(W)) == ceil(log2(W)/2)``), avoiding the single-root fan-in.

    Fails closed by default: if ``target_rel_error`` is not ``None`` and some worker cannot satisfy it at
    any allowed precision, raises :class:`InfeasiblePrecisionError` identifying the worker(s) and the
    quantified gap, instead of silently returning a plan that violates the requested accuracy
    (MXR-080-0133). Pass ``allow_infeasible=True`` to instead receive a best-effort ``HeterogeneousPlan``
    whose ``is_feasible()`` is ``False`` and whose offending assignments carry ``meets_target=False`` plus
    a quantified ``achieved_rel_error``.

    A target the modeled roundoff DOES clear needs no opt-in: the plan is returned normally, with
    ``meets_target=True`` and ``is_feasible()`` -- factual statements about the format's gamma-n score at
    ``op_count``. What such a plan is not is *certified*: ``is_certified()`` is ``False`` for every
    constrained plan, and ``evidence_kind`` stays ``"heuristic"``, because ranking formats is not proving
    a workload's error. Requiring ``allow_infeasible=True`` for these callers was the old behavior; it
    claimed a target was "not achievable" while quoting an achieved error orders of magnitude below it.

    Raises ``ValueError`` for malformed input: no workers, a non-int or negative ``n_rows``/``op_count``,
    an empty ``allowed_precisions``, or a ``target_rel_error`` that isn't ``None`` or a finite positive
    number (MXR-080-0134). Each ``Worker`` validates its own device/precisions/throughput at construction,
    which is what keeps the row-split arithmetic below free of division-by-zero and negative-row bugs.
    """
    if not workers:
        raise ValueError("need at least one worker")
    if not all(isinstance(worker, Worker) for worker in workers):
        raise ValueError("workers must contain only Worker records")
    _require_nonneg_int(n_rows, "n_rows")
    _require_nonneg_int(op_count, "op_count")
    if not isinstance(allow_infeasible, bool):
        raise ValueError("allow_infeasible must be an actual Boolean")
    if isinstance(allowed_precisions, (str, bytes)):
        raise ValueError("allowed_precisions must be a sequence, not a string")
    allowed_precisions = tuple(allowed_precisions)
    if not allowed_precisions:
        raise ValueError("allowed_precisions must be nonempty")
    if target_rel_error is not None and (
        isinstance(target_rel_error, bool)
        or not isinstance(target_rel_error, (int, float))
        or not math.isfinite(target_rel_error)
        or target_rel_error <= 0
    ):
        raise ValueError("target_rel_error must be None or a finite positive number, got %r" % (target_rel_error,))

    chosen = []
    for w in workers:
        choice = _best_precision(w, allowed_precisions, op_count, target_rel_error)
        eff = w.base_throughput * _THROUGHPUT[(w.device, choice.precision)]
        chosen.append((w, choice, eff))

    if not allow_infeasible:
        # Fail closed on the workers whose NUMBERS miss: no supported+allowed precision scores at or below
        # the target (or the worker cannot run any allowed precision at all). A worker whose score does
        # clear the target is planned normally -- the plan simply is not certified (plan.is_certified()),
        # because this planner has no operation graph, magnitude/conditioning evidence, reduction order,
        # or communication topology certificate. Ranking is not proving; but refusing to rank, and calling
        # an achieved ~1e-13 "not achievable" against a 1e-3 ask, was a false statement about numbers.
        infeasible = [choice for _, choice, _ in chosen if not choice.meets_target]
        if infeasible:
            raise InfeasiblePrecisionError(
                "plan_heterogeneous: cannot certify target_rel_error=%r for %d of %d worker(s): %s. "
                "Pass allow_infeasible=True to receive a best-effort HeterogeneousPlan instead (then "
                "check plan.is_feasible() and plan.infeasible_assignments())."
                % (
                    target_rel_error,
                    len(infeasible),
                    len(chosen),
                    "; ".join(
                        c.reason or "no allowed precision reaches the target and no reason was recorded"
                        for c in infeasible
                    ),
                )
            )

    # Every eff is finite and strictly positive here: Worker.__post_init__ enforces base_throughput > 0
    # and only ever lets choice.precision be one of worker.precisions, each of which is guaranteed (also
    # by Worker.__post_init__) to have a positive _THROUGHPUT entry for worker.device. So total_eff > 0
    # given workers is nonempty, and the zero/negative-throughput division-by-zero and negative-row-
    # assignment failure modes from MXR-080-0134 are unreachable once every input has been validated.
    total_eff = sum(eff for _, _, eff in chosen)
    assignments: list[WorkerAssignment] = []
    assigned = 0
    for i, (w, choice, eff) in enumerate(chosen):
        if i == len(chosen) - 1:
            rows = n_rows - assigned  # last worker takes the remainder (exact total)
        else:
            rows = int(round(n_rows * eff / total_eff))
            rows = min(rows, n_rows - assigned)
        assigned += rows
        # Unconstrained means BOTH that no target was asked for and that the worker could actually run an
        # allowed precision: a capability miss (supports none of allowed_precisions) is a real miss and is
        # reported as one even with target_rel_error=None, never relabelled as trivially satisfied.
        unconstrained = target_rel_error is None and choice.meets_target
        assignments.append(
            WorkerAssignment(
                name=w.name,
                rows=rows,
                precision=choice.precision,
                effective_throughput=eff,
                meets_target=choice.meets_target,
                achieved_rel_error=None if unconstrained else choice.achieved_rel_error,
                evidence_kind="unconstrained" if unconstrained else "heuristic",
            )
        )

    return HeterogeneousPlan(assignments=tuple(assignments), reduce_depth=_reduce_depth(len(workers)))
