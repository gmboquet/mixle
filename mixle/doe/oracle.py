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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.doe.designs import Bounds
from mixle.fault import abstain_on_timeout

# The declared verifiability tiers, weakest to strongest. "self_graded" is deliberately excluded: a
# model grading its own candidates is the banned reward, rejected at VerifiableOracle construction.
VERIFIABILITY_TIERS = frozenset({"executable", "simulation", "held_out_truth", "real_measurement"})


@dataclass
class OracleResult:
    """One candidate's verification outcome: its score, a receipt of how it was scored, and its cost."""

    score: float
    receipt: dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0


@dataclass
class VerifiableOracle:
    """A callable ``candidate -> OracleResult`` that declares its verifiability tier and fidelity.

    ``score_fn`` does the actual verification (wrap a simulator, an executable check, a held-out
    ground-truth lookup, or a real measurement pipeline; :mod:`mixle.task.toolcall`'s ``ToolCaller``
    is the same "external check as a callable" shape for tool calls). Construction raises if ``tier``
    is not one of :data:`VERIFIABILITY_TIERS` -- "self-graded by a model" is not a valid tier and is
    rejected here, not silently accepted and discovered later.
    """

    name: str
    tier: str
    score_fn: Callable[[Any], OracleResult]
    fidelity: str | None = None
    timeout: float | None = None  # seconds; FAULT-a oracle_timeout: abstain rather than block or guess

    def __post_init__(self) -> None:
        if self.tier not in VERIFIABILITY_TIERS:
            raise ValueError(
                f"VerifiableOracle tier {self.tier!r} is not a recognized verifiability tier "
                f"{sorted(VERIFIABILITY_TIERS)}; in particular, an oracle 'self-graded by a model' is "
                "the banned reward this boundary forbids, and is rejected at construction."
            )

    def __call__(self, candidate: Any) -> OracleResult:
        if self.timeout is None:
            return self.score_fn(candidate)
        return self._call_with_timeout(candidate)

    def _call_with_timeout(self, candidate: Any) -> OracleResult:
        """FAULT-a ``oracle_timeout``: abstain (a maximally-uninformative, zero-cost, explicitly flagged
        result) rather than block the caller or guess a score, if a single scoring call runs over budget.
        Uses a worker thread so a ``score_fn`` that never returns cannot hang the caller either."""
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        def _run() -> OracleResult:
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(self.score_fn, candidate)
            try:
                return future.result(timeout=self.timeout)
            finally:
                # don't block __call__ waiting for a score_fn that already blew its budget -- let the
                # worker thread finish (or leak, for a truly hung score_fn) on its own time, not ours.
                pool.shutdown(wait=False)

        # future.result() raises concurrent.futures.TimeoutError, which is a DISTINCT class from the
        # builtin TimeoutError on Python 3.10 (only aliased to it in 3.11+) -- catch that exact type so
        # oracle_timeout abstention works on every supported interpreter, not just 3.11+.
        outcome = abstain_on_timeout(_run, timeout_error=FuturesTimeoutError)
        if outcome.degraded:
            return OracleResult(
                score=float("-inf"),
                receipt={"oracle_id": self.name, "tier": self.tier, **outcome.to_receipt_fields()},
                cost=0.0,
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
        """Return the number of candidates scored by the oracle."""
        return len(self.history)

    @property
    def best(self) -> DesignCandidate:
        """Return the highest-scoring candidate in the run history."""
        if not self.history:
            raise ValueError("no candidates were proposed; the run history is empty.")
        return max(self.history, key=lambda c: c.result.score)

    def scores(self) -> np.ndarray:
        """Return the run's oracle scores in chronological order."""
        return np.asarray([c.result.score for c in self.history], dtype=float)

    def report(self) -> dict[str, Any]:
        """Named receipt of the run: which oracle, at what tier/fidelity, the best candidate found."""
        b = self.best
        return {
            "oracle": self.oracle_name,
            "tier": self.oracle_tier,
            "fidelity": self.oracle_fidelity,
            "oracle_calls": self.oracle_calls,
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
    refits the proposal model on every observation, and repeats under an ``n_init + n_iter`` budget.

    ``oracle=None`` raises immediately with the explicit refusal ("no verifiable objective; cannot
    optimize") -- the hard precondition checked before any candidate is proposed.
    Continuous/low-dimensional ``bounds`` only (see module docstring); the proposal model is
    :class:`~mixle.doe.optimizer.BayesianOptimizer`, maximizing the oracle's score.
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
        opt.tell(x, result.score)
        run.history.append(DesignCandidate(x=x, result=result))
    return run
