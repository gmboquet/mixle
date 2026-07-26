"""Verifiable oracle boundary for de novo optimization.

Given a design goal for which there is no data yet but there IS a way to check a candidate (a
simulator, an executable test, held-out truth, an assay), :func:`optimize_under_oracle` proposes
candidates, verifies them against the oracle, and keeps a full receipted history of what was tried and
why -- the design-test-learn loop, made accountable, rather than "synthesize data and train" with no
account of what verified it.

The one hard precondition, checked before anything else: there must be a verifiable oracle.
:class:`VerifiableOracle` rejects a "self-graded by a model" tier at CONSTRUCTION -- that is the banned
reward this boundary exists to forbid -- and :func:`optimize_under_oracle` refuses to run at all
without one (``oracle=None`` -> the explicit "no verifiable objective; cannot optimize" refusal, never a
fabricated candidate).

This is a first, deliberately narrow slice: continuous/low-dimensional candidate spaces only, using the
GP Bayesian-optimization loop already in :mod:`mixle.doe` (:class:`~mixle.doe.optimizer.BayesianOptimizer`)
as the proposal model, validated here against a low-cost closed-form oracle before any domain oracle exists.
Not in this slice: structured/discrete candidate spaces (a protein
sequence, a program), amortizing the oracle into a calibrated surrogate, the shared expected-information-
gain acquisition, and full receipt objects -- each is a separate surface and is
left explicit rather than half-built here.

A timeout is an abstention, not a data point: it means the oracle did not answer, not that the true
score at that candidate is catastrophic. :class:`OracleResult` carries an explicit ``abstained`` flag
for exactly this, and :func:`optimize_under_oracle` never feeds an abstained result to the proposal
model's fit as if it were ground truth, even though it is still kept in ``DesignRun.history`` for the
receipted record. ``VerifiableOracle`` also validates every result crossing its boundary (return type,
finite score unless abstained, finite nonnegative cost, receipt shape) rather than letting a malformed
``score_fn`` return silently corrupt the run, and offers a best-effort cooperative cancellation signal to
a timed-out call's abandoned worker (Python cannot forcibly kill a thread; see
``VerifiableOracle._call_with_timeout``) plus after-the-fact accounting for whatever that abandoned work
eventually returns, so a side-effecting oracle's cost is never simply lost.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.doe.designs import Bounds
from mixle.system.fault import abstain_on_timeout

# The declared verifiability tiers, weakest to strongest. "self_graded" is deliberately excluded: a
# model grading its own candidates is the banned reward, rejected at VerifiableOracle construction.
VERIFIABILITY_TIERS = frozenset({"executable", "simulation", "held_out_truth", "real_measurement"})


def _validated_real(value: Any, label: str) -> float:
    """Return ``value`` as a ``float`` if it is a genuine real number; raise ``TypeError`` otherwise.

    Rejects ``bool`` (a ``bool`` is an ``int`` subclass in Python but is never a meaningful score, cost,
    or timeout) and anything that is not an ``int``/``float``/numpy real scalar.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}: {value!r}.")
    return float(value)


def _accepts_keyword(fn: Callable[..., Any], keyword: str) -> bool:
    """Return whether ``fn`` can be called with ``keyword=...`` (a named parameter, or ``**kwargs``).

    Used once, at :class:`VerifiableOracle` construction, to detect whether a ``score_fn`` opted into
    the cooperative ``cancel_event`` cancellation boundary (see ``_call_with_timeout``) -- a plain,
    single-argument ``score_fn`` (every existing caller as of this writing) is left completely
    unaffected either way.
    """
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    for p in params:
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == keyword and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            return True
    return False


@dataclass
class OracleResult:
    """One candidate's verification outcome: its score, a receipt of how it was scored, and its cost.

    Construction validates its own shape (MXR-080-0189): a real, finite ``score`` unless
    ``abstained=True``, a finite nonnegative ``cost``, and a ``dict`` ``receipt``. This is the boundary
    a ``score_fn``'s raw, untrusted return value crosses (see ``VerifiableOracle._validate_returned``),
    so a wrong type, a NaN/inf that leaked in from some computation, or a negative "cost" is caught
    immediately here rather than silently reaching a GP fit or a cost report.

    ``abstained`` marks a result that is NOT a genuine observation of the oracle's true objective at the
    scored candidate -- for example, :class:`VerifiableOracle`'s timeout abstention. An abstained
    result's ``score`` is a placeholder (``-inf`` by convention, never a claim about the true objective)
    kept only so the receipted history stays complete; consumers such as :func:`optimize_under_oracle`
    gate on ``abstained`` and must never feed one to a fit as if it were ground truth.
    """

    score: float
    receipt: dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    abstained: bool = False
    passed: bool = True
    valid: bool = True

    def __post_init__(self) -> None:
        self.score = _validated_real(self.score, "OracleResult.score")
        if not self.abstained and not math.isfinite(self.score):
            raise ValueError(
                f"OracleResult.score={self.score} is not finite. A genuine (non-abstained) oracle "
                "result must report a real, finite score -- an oracle that cannot score a candidate "
                "should return abstained=True instead of encoding 'unknown' as an infinite or NaN "
                "score, which would be indistinguishable from a genuinely catastrophic observation."
            )
        self.cost = _validated_real(self.cost, "OracleResult.cost")
        if not math.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError(f"OracleResult.cost must be finite and nonnegative, got {self.cost}.")
        if not isinstance(self.receipt, dict):
            raise TypeError(
                f"OracleResult.receipt must be a dict, got {type(self.receipt).__name__}: {self.receipt!r}."
            )
        if not isinstance(self.passed, bool) or not isinstance(self.valid, bool):
            raise TypeError("OracleResult passed and valid verdicts must be boolean")
        if self.abstained:
            self.passed = False
            self.valid = False


@dataclass
class LateOracleResult:
    """A timed-out oracle call's outcome, captured after the fact.

    ``VerifiableOracle`` cannot forcibly kill the worker thread a timeout abandons -- Python has no API
    to do so (see ``_call_with_timeout``). If a ``score_fn`` that ignores (or does not support) the
    cooperative ``cancel_event`` eventually finishes anyway, its outcome is appended to
    ``VerifiableOracle.late_results`` instead of being silently discarded, so a side-effecting oracle's
    eventual cost/outcome is still visible somewhere, even though the caller had already moved on and
    received an abstention in its place.
    """

    candidate: Any
    ok: bool
    result: OracleResult | None
    error: BaseException | None


@dataclass
class VerifiableOracle:
    """A callable ``candidate -> OracleResult`` that declares its verifiability tier and fidelity.

    ``score_fn`` does the actual verification (wrap a simulator, an executable check, a held-out
    ground-truth lookup, or a real measurement pipeline; :mod:`mixle.task.toolcall`'s ``ToolCaller``
    is the same "external check as a callable" shape for tool calls). Construction raises if ``tier``
    is not one of :data:`VERIFIABILITY_TIERS` -- "self-graded by a model" is not a valid tier and is
    rejected here, not silently accepted and discovered later.

    ``score_fn`` may optionally accept a ``cancel_event: threading.Event`` keyword argument (detected
    once, at construction, via introspection -- see ``_accepts_keyword``). A side-effecting ``score_fn``
    that wraps something cancellable (a subprocess, an HTTP request) should periodically check
    ``cancel_event.is_set()`` and tear its underlying call down promptly once set. This is a best-effort,
    cooperative cancellation boundary, not a forced kill -- Python cannot forcibly terminate a thread, so
    a ``score_fn`` that does not accept ``cancel_event`` (every existing caller as of this writing)
    simply keeps running to completion in the background after a timeout abstains, exactly as before,
    except its eventual outcome is now captured in ``late_results`` rather than lost. See
    ``_call_with_timeout``.
    """

    name: str
    tier: str
    score_fn: Callable[[Any], OracleResult]
    fidelity: str | None = None
    timeout: float | None = None  # seconds; FAULT-a oracle_timeout: abstain rather than block or guess
    late_results: list[LateOracleResult] = field(default_factory=list, init=False, repr=False, compare=False)
    _accepts_cancel_event: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tier not in VERIFIABILITY_TIERS:
            raise ValueError(
                f"VerifiableOracle tier {self.tier!r} is not a recognized verifiability tier "
                f"{sorted(VERIFIABILITY_TIERS)}; in particular, an oracle 'self-graded by a model' is "
                "the banned reward this boundary forbids, and is rejected at construction."
            )
        if self.timeout is not None:
            timeout = _validated_real(self.timeout, "VerifiableOracle.timeout")
            if not math.isfinite(timeout) or timeout <= 0.0:
                raise ValueError(
                    "VerifiableOracle.timeout must be a finite, positive number of seconds, or None for "
                    f"no timeout; got {self.timeout}."
                )
            self.timeout = timeout
        self._accepts_cancel_event = _accepts_keyword(self.score_fn, "cancel_event")

    def __call__(self, candidate: Any) -> OracleResult:
        if candidate is None:
            raise ValueError(
                f"oracle {self.name!r} was called with candidate=None; a null candidate cannot be "
                "scored, and this is rejected as a caller bug rather than silently forwarded to "
                "score_fn."
            )
        if self.timeout is None:
            return self._validate_returned(self.score_fn(candidate))
        return self._call_with_timeout(candidate)

    def _validate_returned(self, returned: Any) -> OracleResult:
        """Reject anything ``score_fn`` returns that is not an :class:`OracleResult`.

        A raw float, a dict, ``None``, or any other shape is a caller bug, not a score of zero or a
        silent no-op -- every verification must be receipted, so an unreceipted return is a hard error,
        never fabricated into an ``OracleResult`` on the caller's behalf.
        """
        if not isinstance(returned, OracleResult):
            raise TypeError(
                f"oracle {self.name!r}'s score_fn must return an OracleResult, got "
                f"{type(returned).__name__}: {returned!r}. Construct an OracleResult(score=..., "
                "receipt=..., cost=...) so every verification is receipted, rather than returning a "
                "raw score, dict, or None."
            )
        return returned

    def _record_late_result(self, candidate: Any, ok: bool, payload: Any) -> None:
        """Record a timed-out call's outcome once it eventually arrives, instead of losing it."""
        if not ok:
            self.late_results.append(LateOracleResult(candidate=candidate, ok=False, result=None, error=payload))
            return
        try:
            validated = self._validate_returned(payload)
        except TypeError as exc:
            self.late_results.append(LateOracleResult(candidate=candidate, ok=False, result=None, error=exc))
            return
        self.late_results.append(LateOracleResult(candidate=candidate, ok=True, result=validated, error=None))

    def _call_with_timeout(self, candidate: Any) -> OracleResult:
        """FAULT-a ``oracle_timeout``: abstain (a maximally-uninformative, zero-cost, explicitly flagged
        result) rather than block the caller or guess a score, if a single scoring call runs over budget.

        Runs ``score_fn`` on a worker thread so a hang there cannot hang the caller either --
        ``Thread.join(timeout=...)`` returns on schedule whether or not the thread actually finished, so
        a genuinely hung ``score_fn`` leaves its thread running in the background indefinitely (Python has
        no API to forcibly kill a thread). The worker is created with ``daemon=True`` so that leak cannot
        also keep the whole *process* alive once everything else is done.

        Deliberately a raw ``threading.Thread``, not a ``concurrent.futures.ThreadPoolExecutor``: routing
        this through an executor looks like the safer, more "proper" choice, but it is not --
        ``concurrent.futures.thread`` registers a process-exit hook (``_python_exit``, wired in via
        ``threading._register_atexit``) that unconditionally runs an *untimed* ``Thread.join()`` on every
        worker thread any executor ever created, daemon or not, before the interpreter is allowed to
        exit. A single truly-hung oracle call would therefore still hang the whole process at exit even
        behind ``pool.shutdown(wait=False)`` -- confirmed empirically: an executor-based version of this
        method returns its timeout promptly, but the *process* never exits when ``score_fn`` never
        returns. A raw, non-registered daemon thread has no such hook waiting on it.

        Abandoning the worker does not mean abandoning accountability for it (MXR-080-0189). Two things
        happen at the moment of abandonment, both best-effort since Python cannot forcibly stop the
        thread: (1) ``cancel_event`` is set -- a cooperative cancellation signal a ``score_fn`` written
        to check it (see the class docstring) can use to tear down its own underlying side effect
        promptly; (2) a ``threading.Lock``-guarded "first past the post" race between this call's timeout
        and the worker's eventual completion decides, exactly once and without ambiguity, whether the
        call counts as on-time (delivered through ``result`` below) or late (recorded to
        ``self.late_results`` by whichever side loses the race) -- so a ``score_fn`` that keeps running
        (or the cost it keeps accruing) after a timeout is reported is captured somewhere, never silently
        dropped on the floor.
        """
        import queue
        import threading

        result: queue.Queue = queue.Queue(maxsize=1)
        cancel_event = threading.Event()
        decision_lock = threading.Lock()
        decided: list[str | None] = [None]  # single-slot cell: None -> "on_time" | "timed_out", set once

        def _target() -> None:
            try:
                if self._accepts_cancel_event:
                    payload = self.score_fn(candidate, cancel_event=cancel_event)
                else:
                    payload = self.score_fn(candidate)
                payload = self._validate_returned(payload)
                ok = True
            except BaseException as exc:  # noqa: BLE001 -- ferried to whichever side wins the race below
                payload, ok = exc, False
            with decision_lock:
                if decided[0] is None:
                    decided[0] = "on_time"
                    result.put((ok, payload))
                    return
            # Lost the race: the caller already timed out and moved on without us. Account for this
            # instead of letting it vanish -- see the method docstring.
            self._record_late_result(candidate, ok, payload)

        def _run() -> OracleResult:
            worker = threading.Thread(target=_target, daemon=True)
            worker.start()
            worker.join(timeout=self.timeout)
            with decision_lock:
                if decided[0] is None:
                    decided[0] = "timed_out"
                    cancel_event.set()
            if decided[0] == "timed_out":
                # score_fn blew its budget (or is truly hung). Abandon it -- Python cannot forcibly kill
                # a thread -- and let the caller move on now instead of waiting on it any further.
                raise TimeoutError(f"oracle {self.name!r} exceeded its {self.timeout}s timeout")
            ok, payload = result.get()
            if not ok:
                raise payload
            return payload

        outcome = abstain_on_timeout(_run)
        if outcome.degraded:
            return OracleResult(
                score=float("-inf"),
                receipt={
                    "oracle_id": self.name,
                    "tier": self.tier,
                    "cancel_requested": True,
                    "cooperative_cancel_supported": self._accepts_cancel_event,
                    **outcome.to_receipt_fields(),
                },
                cost=0.0,
                abstained=True,
            )
        return outcome.value


@dataclass
class DesignCandidate:
    """One proposed-and-verified candidate: the point tried and what the oracle said about it."""

    x: np.ndarray
    result: OracleResult


@dataclass
class DesignRun:
    """The full receipted history of a design loop: every candidate tried, and the oracle's identity."""

    oracle_name: str
    oracle_tier: str
    oracle_fidelity: str | None
    history: list[DesignCandidate] = field(default_factory=list)

    @property
    def oracle_calls(self) -> int:
        """Return the number of candidates attempted against the oracle (abstentions included)."""
        return len(self.history)

    @property
    def genuine_history(self) -> list[DesignCandidate]:
        """Return ``history`` filtered to genuine (non-abstained) observations.

        An abstention (see ``OracleResult.abstained`` -- e.g. a timeout) is not a real observation of
        the oracle's objective, so it is excluded here; use ``history`` directly for the full,
        unfiltered receipted record, abstentions included.
        """
        return [c for c in self.history if not c.result.abstained]

    @property
    def best(self) -> DesignCandidate:
        """Return the highest-scoring GENUINE candidate in the run history.

        Abstained candidates (see ``OracleResult.abstained``) never win: an abstention is not a real
        observation of the oracle's objective, and surfacing one as "the best candidate found" would
        report a fabricated result as if it were verified. Raises if every candidate abstained (no
        genuine observation exists to report), distinct from the empty-history case.
        """
        if not self.history:
            raise ValueError("no candidates were proposed; the run history is empty.")
        genuine = self.genuine_history
        if not genuine:
            raise ValueError(
                f"all {len(self.history)} candidate(s) in the run history abstained (e.g. every oracle "
                "call timed out); there is no genuine, verified observation to report as the best."
            )
        return max(genuine, key=lambda c: c.result.score)

    def scores(self) -> np.ndarray:
        """Return the run's oracle scores in chronological order, abstentions included (not hidden)."""
        return np.asarray([c.result.score for c in self.history], dtype=float)

    def report(self) -> dict[str, Any]:
        """Named receipt of the run: which oracle, at what tier/fidelity, the best candidate found.

        ``abstained_calls`` is the count of attempted-but-abstained calls (e.g. timeouts) folded into
        ``oracle_calls`` but excluded from ``best_*``; ``total_cost`` sums ``OracleResult.cost`` exactly
        as receipted at the time of the call and does not retroactively include a late-arriving cost
        from an abandoned oracle call that completes after the fact (see
        ``VerifiableOracle.late_results``).
        """
        b = self.best
        return {
            "oracle": self.oracle_name,
            "tier": self.oracle_tier,
            "fidelity": self.oracle_fidelity,
            "oracle_calls": self.oracle_calls,
            "abstained_calls": sum(1 for c in self.history if c.result.abstained),
            "best_score": b.result.score,
            "best_x": b.x.tolist(),
            "best_cost": b.result.cost,
            "best_receipt": dict(b.result.receipt),
            "total_cost": float(sum(c.result.cost for c in self.history)),
        }


def optimize_under_oracle(
    oracle: VerifiableOracle | None,
    bounds: Bounds,
    *,
    n_init: int = 5,
    n_iter: int = 15,
    seed: Any = None,
    **bo_kwargs: Any,
) -> DesignRun:
    """Run a propose-verify-refit design loop under a fixed oracle budget.

    The loop proposes candidates, verifies each one with ``oracle``, keeps the receipted history,
    refits the proposal model on every GENUINE observation, and repeats under an ``n_init + n_iter``
    budget of oracle calls.

    ``oracle=None`` raises immediately with the explicit refusal ("no verifiable objective; cannot
    optimize") -- the hard precondition checked before any candidate is proposed.
    Continuous/low-dimensional ``bounds`` only (see module docstring); the proposal model is
    :class:`~mixle.doe.optimizer.BayesianOptimizer`, maximizing the oracle's score.

    An abstained result (``OracleResult.abstained`` -- e.g. a timeout; see ``VerifiableOracle``) is
    never told to the proposal model as if it were a real observation (MXR-080-0189): a timeout means
    the oracle did not answer, not that the true score at that candidate is ``-inf``, and training a fit
    on a fabricated ``-inf`` "observation" can corrupt it. The abstained candidate is still appended to
    ``run.history`` -- the receipted record keeps every attempt, genuine or not -- it is simply excluded
    from what the optimizer learns from. ``BayesianOptimizer.ask`` is explicitly designed to tolerate
    this (its initial-design dispensing is gated on points asked, not points told), so skipping ``tell``
    for an abstention does not desynchronize or corrupt the proposal model's state.
    """
    if oracle is None:
        raise ValueError(
            "no verifiable objective; cannot optimize. optimize_under_oracle requires a "
            "VerifiableOracle -- this is a hard precondition, not a missing default."
        )
    from mixle.doe.optimizer import BayesianOptimizer

    opt = BayesianOptimizer(bounds, maximize=True, n_init=n_init, seed=seed, **bo_kwargs)
    run = DesignRun(oracle_name=oracle.name, oracle_tier=oracle.tier, oracle_fidelity=oracle.fidelity)
    for _ in range(int(n_init) + int(n_iter)):
        x = np.asarray(opt.ask(), dtype=np.float64)
        result = oracle(x)
        run.history.append(DesignCandidate(x=x, result=result))
        if not result.abstained:
            opt.tell(x, result.score)
    return run
