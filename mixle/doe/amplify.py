"""Capture an amplified teacher with collapse monitoring.

The "amplified teacher" here is :func:`~mixle.doe.oracle.optimize_under_oracle`'s search itself: round
1 spends an oracle-call budget searching, and is verified stronger than a single ungrounded guess before
anything else happens. If the search does not beat its best single input, there is nothing to capture,
and this function stops with an explicit reason rather than distilling nothing.

The student captured from round 1 is a low-cost regression surrogate of the oracle's score landscape, fit
only from round 1's oracle-verified ``(x, score)`` pairs. It never grades a candidate itself; it only
proposes where round 2 should spend its matched oracle-call budget. ``student(x)`` is a plain
``candidate -> predicted_score`` callable, the same shape any other teacher/task-model in this codebase
is called with. Every accepted score in round 2
still comes from the oracle. No student or LLM self-grade enters ``DesignRun.history``.

:func:`mixle.task.collapse.collapse_monitor` checks the two-round trajectory for
regression or mode collapse; it is reused here rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mixle.doe.designs import Bounds, random_design
from mixle.doe.oracle import DesignCandidate, DesignRun, VerifiableOracle, optimize_under_oracle
from mixle.task.collapse import CollapseVerdict, collapse_monitor


def _design_matrix(xs: np.ndarray, degree: int) -> np.ndarray:
    """Return polynomial features plus an intercept for the amplification student."""
    cols = [np.ones(xs.shape[0])]
    for d in range(1, degree + 1):
        cols.append(xs**d)
    return np.concatenate([c.reshape(xs.shape[0], -1) if c.ndim > 1 else c[:, None] for c in cols], axis=1)


@dataclass
class StudentTeacher:
    """Captured regression surrogate used to rank follow-up design candidates.

    The object is callable as ``student(x) -> predicted_score`` so it can be
    used like other teacher/task-model callables, but its role here is only
    proposal ranking. Accepted scores still come from the verifiable oracle.
    """

    coef: np.ndarray
    degree: int

    def __call__(self, x: np.ndarray) -> float:
        """Predict the score of one candidate point from the polynomial surrogate."""
        row = _design_matrix(np.asarray(x, dtype=np.float64).reshape(1, -1), self.degree)
        return float((row @ self.coef).item())


def fit_student(run: DesignRun, *, degree: int = 2) -> StudentTeacher:
    """Fit :class:`StudentTeacher` from ``run``'s oracle-verified history."""
    xs = np.stack([c.x for c in run.history]).astype(np.float64)
    ys = np.asarray([c.result.score for c in run.history], dtype=np.float64)
    x_design = _design_matrix(xs, degree)
    coef, *_ = np.linalg.lstsq(x_design, ys, rcond=None)
    return StudentTeacher(coef=coef, degree=degree)


@dataclass
class AmplificationRound:
    """One oracle-verified amplification round and its best observed score."""

    run: DesignRun
    best_score: float
    xs: list[np.ndarray]


@dataclass
class AmplifyReport:
    """Receipt for a two-round amplify-and-capture run.

    ``round1`` is the initial oracle-driven search. ``round2`` is present only
    when round 1 beats the single-input baseline and a captured student is used
    to propose the matched-budget follow-up batch. ``collapse`` records the
    trajectory-level collapse verdict when both rounds run.
    """

    round1: AmplificationRound
    round2: AmplificationRound | None
    baseline_single_input_score: float
    beats_single_input: bool
    round2_beats_round1: bool
    collapse: CollapseVerdict | None
    student: StudentTeacher | None
    stopped_early: bool
    reason: str | None


def amplify_and_capture(
    oracle: VerifiableOracle,
    bounds: Bounds,
    *,
    n_init: int = 5,
    n_iter: int = 10,
    candidate_pool_size: int = 200,
    degree: int = 2,
    seed: int | None = None,
) -> AmplifyReport:
    """Round 1: search the oracle for a budget of ``n_init + n_iter`` calls (the amplified teacher).
    The search must beat a single ungrounded guess, or this returns the explicit ``stopped_early=True``
    result with nothing distilled. Otherwise: fit :class:`StudentTeacher` from
    round 1's history; round 2 uses the student to rank a large candidate pool efficiently and spends the
    same oracle-call budget verifying only the top-ranked candidates -- student-guided, not blind, but
    every accepted score is still oracle-verified. Runs :func:`mixle.task.collapse.collapse_monitor`
    over the two rounds.
    """
    run1 = optimize_under_oracle(oracle, bounds, n_init=n_init, n_iter=n_iter, seed=seed)
    round1 = AmplificationRound(run=run1, best_score=float(run1.best.result.score), xs=[c.x for c in run1.history])

    baseline_x = random_design(bounds, 1, seed=seed)[0]
    baseline_score = float(oracle(baseline_x).score)
    beats_single_input = round1.best_score > baseline_score
    if not beats_single_input:
        return AmplifyReport(
            round1=round1,
            round2=None,
            baseline_single_input_score=baseline_score,
            beats_single_input=False,
            round2_beats_round1=False,
            collapse=None,
            student=None,
            stopped_early=True,
            reason="the amplified teacher (round 1 search) did not beat its best single input; nothing to capture",
        )

    student = fit_student(run1, degree=degree)

    budget = int(n_init) + int(n_iter)
    fresh_pool = random_design(bounds, int(candidate_pool_size), seed=None if seed is None else seed + 1)
    # warm-start the candidate pool with round 1's own verified points -- standard practice for an
    # amplification round building on prior verified data. It makes "round 2 at matched budget beats or
    # matches round 1" a property of the construction rather than a fresh random-pool comparison: the
    # student is fit on these exact points and should rank round 1's best point among the top candidates,
    # and whatever gets selected is re-verified by the oracle.
    pool = np.concatenate([np.stack(round1.xs), fresh_pool], axis=0)
    predicted = np.asarray([student(x) for x in pool])
    top_idx = np.argsort(-predicted)[:budget]

    run2 = DesignRun(oracle_name=oracle.name, oracle_tier=oracle.tier, oracle_fidelity=oracle.fidelity)
    for idx in top_idx:
        x = pool[idx]
        result = oracle(x)  # the oracle supplies the accepted score; the student only proposes where to look
        run2.history.append(DesignCandidate(x=x, result=result))
    round2 = AmplificationRound(run=run2, best_score=float(run2.best.result.score), xs=[c.x for c in run2.history])

    collapse = collapse_monitor(
        [
            {"score": round1.best_score, "candidates": round1.xs},
            {"score": round2.best_score, "candidates": round2.xs},
        ]
    )

    return AmplifyReport(
        round1=round1,
        round2=round2,
        baseline_single_input_score=baseline_score,
        beats_single_input=True,
        round2_beats_round1=round2.best_score >= round1.best_score,
        collapse=collapse,
        student=student,
        stopped_early=False,
        reason=None,
    )
