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
``score_fn`` return silently corrupt the run. Timeout workers are bounded by a circuit breaker,
cooperative cancellation is requested, and eventual late outcomes are linked back to their run by
immutable call IDs so side-effecting cost is never silently lost.
"""

from __future__ import annotations

import inspect
import itertools
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from mixle.doe.designs import Bounds, _require_exact_positive_int
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


def _freeze_receipt_value(value: Any, path: str) -> Any:
    """Copy and recursively freeze durable receipt data."""
    if value is None or type(value) in (bool, int, float, str):
        if type(value) is float and not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{path} contains a non-finite number.")
        return converted
    if type(value) is dict:
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} contains non-string key {key!r}.")
            frozen[key] = _freeze_receipt_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if type(value) in (list, tuple):
        return tuple(_freeze_receipt_value(item, f"{path}[{index}]") for index, item in enumerate(value))
    if type(value) in (set, frozenset):
        return frozenset(_freeze_receipt_value(item, path) for item in value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
            raise ValueError(f"{path} contains a non-finite array.")
        array.setflags(write=False)
        return array
    raise TypeError(f"{path} contains unsupported mutable value {type(value).__name__}.")


def _receipt_to_plain(value: Any) -> Any:
    """Return a detached report-friendly view of frozen receipt data."""
    if isinstance(value, Mapping):
        return {key: _receipt_to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_receipt_to_plain(item) for item in value]
    if isinstance(value, frozenset):
        return [_receipt_to_plain(item) for item in sorted(value, key=repr)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@dataclass(frozen=True)
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
    receipt: Mapping[str, Any] = field(default_factory=dict)
    cost: float = 1.0
    abstained: bool = False
    passed: bool = True
    valid: bool = True
    call_id: str | None = None
    call_started: bool = True

    def __post_init__(self) -> None:
        score = _validated_real(self.score, "OracleResult.score")
        if type(self.abstained) is not bool:
            raise TypeError("OracleResult.abstained must be an actual bool.")
        if type(self.passed) is not bool or type(self.valid) is not bool or type(self.call_started) is not bool:
            raise TypeError("OracleResult passed, valid, and call_started verdicts must be actual booleans.")
        if self.call_id is not None and (type(self.call_id) is not str or not self.call_id):
            raise ValueError("OracleResult.call_id must be None or a nonempty string.")
        if type(self.receipt) is not dict and not isinstance(self.receipt, MappingProxyType):
            raise TypeError(
                f"OracleResult.receipt must be a dict, got {type(self.receipt).__name__}: {self.receipt!r}."
            )
        frozen_receipt = _freeze_receipt_value(dict(self.receipt), "OracleResult.receipt")
        if not self.abstained and not math.isfinite(score):
            raise ValueError(
                f"OracleResult.score={self.score} is not finite. A genuine (non-abstained) oracle "
                "result must report a real, finite score -- an oracle that cannot score a candidate "
                "should return abstained=True instead of encoding 'unknown' as an infinite or NaN "
                "score, which would be indistinguishable from a genuinely catastrophic observation."
            )
        if self.abstained:
            if score != float("-inf"):
                raise ValueError(
                    "An abstained OracleResult must use score=-inf as its explicit non-observation marker."
                )
            reason = (
                frozen_receipt.get("reason") or frozen_receipt.get("degraded_reason") or frozen_receipt.get("status")
            )
            if type(reason) is not str or not reason.strip():
                raise ValueError("An abstained OracleResult receipt requires a nonempty reason or status.")
        cost = _validated_real(self.cost, "OracleResult.cost")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"OracleResult.cost must be finite and nonnegative, got {cost}.")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "receipt", frozen_receipt)
        if self.abstained:
            object.__setattr__(self, "passed", False)
            object.__setattr__(self, "valid", False)


def _deeply_frozen(value: Any) -> Any:
    """Return an immutable stand-in for ``value``, recursively.

    ``ndarray.setflags(write=False)`` freezes the *buffer*, not what an object-dtype element points at,
    so a list stored in one stayed mutable and a "frozen" receipt could still be edited through its
    own elements (MXR-080-1851). Containers are converted rather than merely copied, because a copy of
    a list is still a list.
    """
    if isinstance(value, np.ndarray):
        frozen = value.copy()
        frozen.setflags(write=False)
        return frozen
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deeply_frozen(item) for key, item in value.items()})
    if isinstance(value, (str, bytes, bytearray)):
        return bytes(value) if isinstance(value, bytearray) else value
    if isinstance(value, (set, frozenset)):
        return frozenset(_deeply_frozen(item) for item in value)
    if isinstance(value, Sequence):
        return tuple(_deeply_frozen(item) for item in value)
    return value


@dataclass(frozen=True)
class LateOracleResult:
    """A timed-out oracle call's outcome, captured after the fact.

    ``VerifiableOracle`` cannot forcibly kill the worker thread a timeout abandons -- Python has no API
    to do so (see ``_call_with_timeout``). If a ``score_fn`` that ignores (or does not support) the
    cooperative ``cancel_event`` eventually finishes anyway, its outcome is appended to
    ``VerifiableOracle.late_results`` instead of being silently discarded, so a side-effecting oracle's
    eventual cost/outcome is still visible somewhere, even though the caller had already moved on and
    received an abstention in its place.
    """

    call_id: str
    candidate: Any
    ok: bool
    result: OracleResult | None
    error: str | None

    def __post_init__(self) -> None:
        try:
            candidate = np.asarray(self.candidate, dtype=np.float64).copy()
        except (TypeError, ValueError):
            # This module scopes itself to numeric candidates (see the module docstring), but this
            # object is built on the abandoned worker thread of a timed-out call, where a raise is
            # nobody's to catch -- it escaped as PytestUnhandledThreadExceptionWarning and lost the
            # late result entirely. Keep the candidate as an immutable object array instead: recording
            # a side-effecting oracle's eventual outcome is the whole point of this class, and the
            # out-of-contract candidate type is not a reason to drop it.
            candidate = np.asarray(self.candidate, dtype=object).copy()
        candidate.setflags(write=False)
        if candidate.dtype == object:
            # Only the object path can hold mutable elements; the float64 path is frozen by setflags
            # alone. Store the deeply frozen form so the receipt cannot be edited through its elements.
            candidate = _deeply_frozen(candidate.tolist())
        object.__setattr__(self, "candidate", candidate)


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
    cooperative cancellation boundary, not a forced kill -- Python cannot forcibly terminate a thread.
    Calls that ignore cancellation remain quarantined, count against ``max_outstanding_timeouts``, and
    eventually open a circuit that prevents additional worker creation. Their outcomes are captured in
    ``late_results`` when they finish. See ``_call_with_timeout``.
    """

    name: str
    tier: str
    score_fn: Callable[[Any], OracleResult]
    fidelity: str | None = None
    timeout: float | None = None  # seconds; FAULT-a oracle_timeout: abstain rather than block or guess
    max_outstanding_timeouts: int = 4
    _late_results: list[LateOracleResult] = field(default_factory=list, init=False, repr=False, compare=False)
    _accepts_cancel_event: bool = field(default=False, init=False, repr=False, compare=False)
    _call_counter: Any = field(default_factory=lambda: itertools.count(1), init=False, repr=False, compare=False)
    _state_lock: Any = field(default_factory=threading.Lock, init=False, repr=False, compare=False)
    _outstanding_call_ids: set[str] = field(default_factory=set, init=False, repr=False, compare=False)

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
        self.max_outstanding_timeouts = _require_exact_positive_int(
            self.max_outstanding_timeouts, "max_outstanding_timeouts"
        )
        self._accepts_cancel_event = _accepts_keyword(self.score_fn, "cancel_event")

    @property
    def late_results(self) -> tuple[LateOracleResult, ...]:
        """Immutable snapshot of late outcomes accumulated so far."""
        with self._state_lock:
            return tuple(self._late_results)

    @property
    def outstanding_timeouts(self) -> int:
        """Number of timed-out workers that have not cooperatively stopped or completed."""
        with self._state_lock:
            return len(self._outstanding_call_ids)

    def _next_call_id(self) -> str:
        with self._state_lock:
            return f"{self.name}:{next(self._call_counter)}"

    def __call__(self, candidate: Any) -> OracleResult:
        if candidate is None:
            raise ValueError(
                f"oracle {self.name!r} was called with candidate=None; a null candidate cannot be "
                "scored, and this is rejected as a caller bug rather than silently forwarded to "
                "score_fn."
            )
        call_id = self._next_call_id()
        if self.timeout is None:
            return replace(self._validate_returned(self.score_fn(candidate)), call_id=call_id, call_started=True)
        return self._call_with_timeout(candidate, call_id)

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

    def _record_late_result(self, call_id: str, candidate: Any, ok: bool, payload: Any) -> None:
        """Record a timed-out call's outcome once it eventually arrives, instead of losing it."""
        if not ok:
            late = LateOracleResult(
                call_id=call_id,
                candidate=candidate,
                ok=False,
                result=None,
                error=f"{type(payload).__name__}: {payload}",
            )
            with self._state_lock:
                self._late_results.append(late)
            return
        try:
            validated = replace(self._validate_returned(payload), call_id=call_id, call_started=True)
        except (TypeError, ValueError) as exc:
            late = LateOracleResult(
                call_id=call_id,
                candidate=candidate,
                ok=False,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            with self._state_lock:
                self._late_results.append(late)
            return
        late = LateOracleResult(call_id=call_id, candidate=candidate, ok=True, result=validated, error=None)
        with self._state_lock:
            self._late_results.append(late)

    def _call_with_timeout(self, candidate: Any, call_id: str) -> OracleResult:
        """FAULT-a ``oracle_timeout``: abstain (a maximally-uninformative, zero-cost, explicitly flagged
        result) rather than block the caller or guess a score, if a single scoring call runs over budget.

        Runs ``score_fn`` on a worker thread so a hang there cannot hang the caller either --
        ``Thread.join(timeout=...)`` returns on schedule whether or not the thread actually finished, so
        a genuinely hung ``score_fn`` leaves its thread running in the background (Python has no API to
        forcibly kill a thread). At most ``max_outstanding_timeouts`` such workers can exist per oracle;
        further calls receive an explicit ``oracle_quarantined`` abstention and start no thread. Workers
        are daemonized so they cannot keep the process alive.

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

        result: queue.Queue = queue.Queue(maxsize=1)
        cancel_event = threading.Event()
        decision_lock = threading.Lock()
        decided: list[str | None] = [None]  # single-slot cell: None -> "on_time" | "timed_out", set once

        with self._state_lock:
            if len(self._outstanding_call_ids) >= self.max_outstanding_timeouts:
                return OracleResult(
                    score=float("-inf"),
                    receipt={
                        "status": "oracle_quarantined",
                        "reason": "outstanding timed-out worker limit reached",
                        "oracle_id": self.name,
                        "tier": self.tier,
                        "outstanding_timeouts": len(self._outstanding_call_ids),
                    },
                    cost=0.0,
                    abstained=True,
                    call_id=call_id,
                    call_started=False,
                )
            self._outstanding_call_ids.add(call_id)

        def _target() -> None:
            try:
                if self._accepts_cancel_event:
                    payload = self.score_fn(candidate, cancel_event=cancel_event)
                else:
                    payload = self.score_fn(candidate)
                payload = replace(self._validate_returned(payload), call_id=call_id, call_started=True)
                ok = True
            except BaseException as exc:  # noqa: BLE001 -- ferried to whichever side wins the race below
                payload, ok = exc, False
            with self._state_lock:
                self._outstanding_call_ids.discard(call_id)
            with decision_lock:
                if decided[0] is None:
                    decided[0] = "on_time"
                    result.put((ok, payload))
                    return
            # Lost the race: the caller already timed out and moved on without us. Account for this
            # instead of letting it vanish -- see the method docstring.
            self._record_late_result(call_id, candidate, ok, payload)

        def _run() -> OracleResult:
            worker = threading.Thread(target=_target, daemon=True)
            try:
                worker.start()
            except BaseException:
                with self._state_lock:
                    self._outstanding_call_ids.discard(call_id)
                raise
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
                    "external_call_started": True,
                    **outcome.to_receipt_fields(),
                },
                cost=0.0,
                abstained=True,
                call_id=call_id,
                call_started=True,
            )
        return outcome.value


@dataclass(frozen=True)
class DesignCandidate:
    """One proposed-and-verified candidate: the point tried and what the oracle said about it."""

    x: np.ndarray
    result: OracleResult
    phase: str | None = None

    def __post_init__(self) -> None:
        point = np.asarray(self.x, dtype=np.float64)
        if point.ndim != 1 or point.size == 0 or not np.all(np.isfinite(point)):
            raise ValueError("DesignCandidate.x must be a nonempty finite one-dimensional point.")
        if not isinstance(self.result, OracleResult):
            raise TypeError("DesignCandidate.result must be an OracleResult.")
        if self.phase not in (None, "initial", "adaptive"):
            raise ValueError("DesignCandidate.phase must be None, 'initial', or 'adaptive'.")
        point = point.copy()
        point.setflags(write=False)
        object.__setattr__(self, "x", point)


@dataclass(frozen=True)
class DesignRun:
    """The full receipted history of a design loop: every candidate tried, and the oracle's identity."""

    oracle_name: str
    oracle_tier: str
    oracle_fidelity: str | None
    _oracle: VerifiableOracle | None = field(default=None, repr=False, compare=False)
    _history: list[DesignCandidate] = field(default_factory=list, init=False, repr=False, compare=False)

    @property
    def history(self) -> tuple[DesignCandidate, ...]:
        """Immutable chronological snapshot of the append-only candidate ledger."""
        return tuple(self._history)

    def append(self, candidate: DesignCandidate) -> None:
        """Append one immutable candidate record; existing ledger entries can never be replaced."""
        if not isinstance(candidate, DesignCandidate):
            raise TypeError("DesignRun.append requires a DesignCandidate.")
        self._history.append(candidate)

    @property
    def oracle_calls(self) -> int:
        """Return the number of candidates attempted against the oracle (abstentions included)."""
        return sum(1 for candidate in self._history if candidate.result.call_started)

    @property
    def candidate_attempts(self) -> int:
        """All proposed candidates, including circuit-breaker abstentions that started no external call."""
        return len(self._history)

    @property
    def genuine_history(self) -> tuple[DesignCandidate, ...]:
        """Return ``history`` filtered to genuine (non-abstained) observations.

        An abstention (see ``OracleResult.abstained`` -- e.g. a timeout) is not a real observation of
        the oracle's objective, so it is excluded here; use ``history`` directly for the full,
        unfiltered receipted record, abstentions included.
        """
        return tuple(c for c in self._history if not c.result.abstained)

    @property
    def best(self) -> DesignCandidate:
        """Return the highest-scoring GENUINE candidate in the run history.

        Abstained candidates (see ``OracleResult.abstained``) never win: an abstention is not a real
        observation of the oracle's objective, and surfacing one as "the best candidate found" would
        report a fabricated result as if it were verified. Raises if every candidate abstained (no
        genuine observation exists to report), distinct from the empty-history case.
        """
        if not self._history:
            raise ValueError("no candidates were proposed; the run history is empty.")
        genuine = self.genuine_history
        if not genuine:
            raise ValueError(
                f"all {len(self._history)} candidate(s) in the run history abstained (e.g. every oracle "
                "call timed out); there is no genuine, verified observation to report as the best."
            )
        return max(genuine, key=lambda c: c.result.score)

    def scores(self) -> np.ndarray:
        """Return the run's oracle scores in chronological order, abstentions included (not hidden)."""
        return np.asarray([c.result.score for c in self._history], dtype=float)

    def report(self) -> dict[str, Any]:
        """Named receipt of the run: which oracle, at what tier/fidelity, the best candidate found.

        ``provisional_total_cost`` is the cost known when calls returned. Late outcomes linked by
        immutable call IDs are reconciled into ``settled_late_cost`` and ``settled_total_cost``.
        ``total_cost`` is only populated once no timed-out call remains outstanding or unresolved.
        """
        genuine = self.genuine_history
        best = max(genuine, key=lambda candidate: candidate.result.score) if genuine else None
        late_results = self._oracle.late_results if self._oracle is not None else ()
        history_call_ids = {
            candidate.result.call_id for candidate in self._history if candidate.result.call_id is not None
        }
        linked_late = tuple(late for late in late_results if late.call_id in history_call_ids)
        late_by_call = {late.call_id: late for late in linked_late}
        timed_out_call_ids = {
            candidate.result.call_id
            for candidate in self._history
            if candidate.result.call_id is not None
            and candidate.result.receipt.get("degraded_mode") == "oracle_timeout"
        }
        outstanding = timed_out_call_ids - late_by_call.keys()
        unresolved = {late.call_id for late in linked_late if not late.ok or late.result is None}
        settled_late_cost = float(sum(late.result.cost for late in linked_late if late.ok and late.result is not None))
        provisional_total = float(sum(candidate.result.cost for candidate in self._history))
        settled_total = provisional_total + settled_late_cost
        cost_status = "settled"
        if outstanding:
            cost_status = "provisional"
        elif unresolved:
            cost_status = "unresolved"
        return {
            "oracle": self.oracle_name,
            "tier": self.oracle_tier,
            "fidelity": self.oracle_fidelity,
            "oracle_calls": self.oracle_calls,
            "candidate_attempts": self.candidate_attempts,
            "initial_calls": sum(
                1 for candidate in self._history if candidate.phase == "initial" and candidate.result.call_started
            ),
            "adaptive_calls": sum(
                1 for candidate in self._history if candidate.phase == "adaptive" and candidate.result.call_started
            ),
            "abstained_calls": sum(1 for candidate in self._history if candidate.result.abstained),
            "status": "verified_result" if best is not None else "no_verified_result",
            "best_score": None if best is None else best.result.score,
            "best_x": None if best is None else best.x.tolist(),
            "best_cost": None if best is None else best.result.cost,
            "best_receipt": None if best is None else _receipt_to_plain(best.result.receipt),
            "provisional_total_cost": provisional_total,
            "settled_late_cost": settled_late_cost,
            "settled_total_cost": settled_total,
            "total_cost": settled_total if cost_status == "settled" else None,
            "cost_status": cost_status,
            "outstanding_late_calls": len(outstanding),
            "unresolved_late_calls": len(unresolved),
            "late_results": [
                {
                    "call_id": late.call_id,
                    "ok": late.ok,
                    "cost": late.result.cost if late.result is not None else None,
                    "error": late.error,
                }
                for late in linked_late
            ],
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

    If the *entire* initial phase abstains, however, the model has no observation at all to fit an
    acquisition step on, and ``ask`` raises rather than guessing. The adaptive phase is therefore not
    entered in that case: the run stops and is returned with its receipts intact, so an all-abstention
    run still yields its audit report (``status="no_verified_result"``) instead of dying inside the
    proposal model (MXR-080-1488).
    """
    if oracle is None:
        raise ValueError(
            "no verifiable objective; cannot optimize. optimize_under_oracle requires a "
            "VerifiableOracle -- this is a hard precondition, not a missing default."
        )
    n_init = _require_exact_positive_int(n_init, "n_init")
    n_iter = _require_exact_positive_int(n_iter, "n_iter", minimum=0)
    from mixle.doe.optimizer import BayesianOptimizer

    opt = BayesianOptimizer(bounds, maximize=True, n_init=n_init, seed=seed, **bo_kwargs)
    run = DesignRun(
        oracle_name=oracle.name,
        oracle_tier=oracle.tier,
        oracle_fidelity=oracle.fidelity,
        _oracle=oracle,
    )
    circuit_open = False
    for phase, budget in (("initial", n_init), ("adaptive", n_iter)):
        if phase == "adaptive" and not run.genuine_history:
            # Every initial call abstained (a hung oracle timing out on every candidate is the
            # motivating case), so the proposal model was never told anything and its acquisition step
            # has no observations to fit. `BayesianOptimizer.ask` raises in exactly that state -- by
            # design, and atomically -- once the space-filling initial design is exhausted, so entering
            # the adaptive phase here would propagate that ValueError out of `optimize_under_oracle`
            # and destroy the whole receipted run at precisely the moment it is most needed
            # operationally (MXR-080-1488). Stop and return the run instead: `report()` already
            # publishes `status="no_verified_result"` alongside the full call and cost evidence for
            # every attempt that was actually made.
            break
        for _ in range(budget):
            x = np.asarray(opt.ask(), dtype=np.float64)
            result = oracle(x)
            run.append(DesignCandidate(x=x, result=result, phase=phase))
            if result.receipt.get("status") == "oracle_quarantined":
                circuit_open = True
                break
            if not result.abstained:
                opt.tell(x, result.score)
        if circuit_open:
            break
    return run
