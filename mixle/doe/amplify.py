"""Capture an amplified teacher with collapse monitoring.

The "amplified teacher" here is :func:`~mixle.doe.oracle.optimize_under_oracle`'s search itself: round 1
spends an ``n_init + n_iter`` oracle-call budget searching, and must be verified stronger than a
*budget-matched* random baseline of the same size before anything else happens. A search's best-of-N
compared against a single extra random draw is a multiple-comparisons trap, not a fair test: for N+1
values drawn i.i.d. from the same distribution, exchangeability alone gives P(max of the first N > the
last one) = N / (N+1), so passage becomes increasingly automatic as the budget N grows even when the
search adds no real capability, and the one extra draw was never counted against the stated budget
either. Matching the baseline's budget to the search's and testing "is round 1's best score bigger than
the baseline's best score by more than a fair reshuffling of the same ``2N`` scores would typically
produce" with a one-sided permutation test (rather than a bare, un-tested "is my number bigger" check on
two single points) is what keeps the gate a calibrated statistical claim instead of a counter that trends
toward "always passes". If the search does not clear the gate at the ``significance`` level, there is
nothing to capture, and this function stops with an explicit reason rather than distilling nothing.

The student captured from round 1 is a low-cost regression surrogate of the oracle's score landscape, fit
only from round 1's oracle-verified ``(x, score)`` pairs. It never grades a candidate itself; it only
proposes where round 2 should spend its matched oracle-call budget -- against a FRESH candidate pool that
excludes round 1's own points, so round 2 cannot pad its score by re-verifying already-known winners (see
``amplify_and_capture``). ``student(x)`` is a plain ``candidate -> predicted_score`` callable, the same
shape any other teacher/task-model in this codebase is called with. Every accepted score in round 2
still comes from the oracle. No student or LLM self-grade enters ``DesignRun.history``.

:func:`mixle.task.collapse.collapse_monitor` checks the two-round trajectory for
regression or mode collapse; it is reused here rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import permutation_test

from mixle.doe.designs import Bounds, random_design
from mixle.doe.oracle import DesignCandidate, DesignRun, VerifiableOracle, optimize_under_oracle

if TYPE_CHECKING:
    # mixle.doe (package init, via amplify) <-> mixle.task (package init, via emulate/propose) is a
    # live, bidirectional, eager import cycle: a top-level import here of mixle.task.collapse closes
    # it. It currently "works" only because each package's __init__.py happens to load its own
    # unaffected submodules before reaching the one that reaches into the other -- reorder either
    # list (e.g. an alphabetize-imports pass) and it breaks exactly like the epistemic<->task cycle
    # already fixed for pilot_ladder.py. CollapseVerdict is only ever used as a type annotation here
    # (deferred to a string by the `from __future__ import annotations` above); collapse_monitor is
    # a local import inside amplify_and_capture(), at call time, once both packages have loaded.
    from mixle.task.collapse import CollapseVerdict


def _design_matrix(xs: np.ndarray, degree: int) -> np.ndarray:
    """Return polynomial features plus an intercept for the amplification student."""
    cols = [np.ones(xs.shape[0])]
    for d in range(1, degree + 1):
        cols.append(xs**d)
    return np.concatenate([c.reshape(xs.shape[0], -1) if c.ndim > 1 else c[:, None] for c in cols], axis=1)


def _best_score_gap(x: np.ndarray, y: np.ndarray) -> float:
    """Permutation-test statistic: how much ``x``'s best score exceeds ``y``'s (see ``amplify_and_capture``)."""
    return float(np.max(x) - np.max(y))


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

    ``round1`` is the initial oracle-driven search. ``baseline`` is a random draw matched to round 1's
    exact oracle-call budget (not a single draw -- see ``amplify_and_capture``), so every oracle call
    either side spent is accounted for in its own receipted :class:`AmplificationRound`.
    ``baseline_p_value`` is the one-sided permutation-test p-value for "round 1's best score exceeds the
    baseline's best score by more than chance", tested against the prespecified ``significance`` level to
    produce ``beats_baseline``. ``round2`` is present only when round 1 clears that gate, and a captured student
    is used to propose the matched-budget follow-up batch from a FRESH candidate pool (round 1's own
    points are never re-offered to round 2, so ``round2_beats_round1`` measures generalization to unseen
    candidates, not re-verification of already-known ones). ``incumbent_best_score`` is bookkeeping only
    -- the best score retained across whichever rounds ran -- and, unlike round 1's era of this report,
    plays no part in computing ``round2_beats_round1`` or ``collapse``. ``collapse`` records the
    trajectory-level collapse verdict when both rounds run.
    """

    round1: AmplificationRound
    round2: AmplificationRound | None
    baseline: AmplificationRound
    baseline_p_value: float
    significance: float
    beats_baseline: bool
    round2_beats_round1: bool
    incumbent_best_score: float
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
    significance: float = 0.05,
    seed: int | None = None,
) -> AmplifyReport:
    """Round 1: search the oracle for a budget of ``n_init + n_iter`` calls (the amplified teacher).

    The search must beat a *budget-matched* random baseline -- also ``n_init + n_iter`` oracle-verified
    draws, not one -- on a one-sided permutation test of "round 1's best score exceeds the baseline's best
    score by more than a fair reshuffling of the pooled ``2 * budget`` scores would typically produce", at
    the prespecified ``significance`` level. A single best-of-N-vs-one-draw comparison is a
    multiple-comparisons trap (see the module docstring): matching the baseline's budget to the search's,
    and testing the two best-of-budget scores against the null distribution obtained by repeatedly
    reshuffling which of the ``2 * budget`` oracle-verified scores "belong" to round 1 versus the baseline
    (rather than a bare, un-tested "is my number bigger" check), is what makes ``beats_baseline`` a
    calibrated claim instead of a counter that trends toward "always passes" as the budget grows. If the
    search does not clear the gate, this returns the explicit ``stopped_early=True`` result with nothing
    distilled.

    Otherwise: fit :class:`StudentTeacher` from round 1's history; round 2 uses the student to rank a
    large, FRESH candidate pool (never round 1's own points -- see the round 2 comment below) and spends
    the same oracle-call budget verifying only the top-ranked fresh candidates -- student-guided, not
    blind, but every accepted score is still oracle-verified. Runs
    :func:`mixle.task.collapse.collapse_monitor` over the two rounds.
    """
    run1 = optimize_under_oracle(oracle, bounds, n_init=n_init, n_iter=n_iter, seed=seed)
    round1 = AmplificationRound(run=run1, best_score=float(run1.best.result.score), xs=[c.x for c in run1.history])

    budget = int(n_init) + int(n_iter)

    # Budget-matched baseline (MXR-080-0161): the same number of oracle-verified draws as round 1 spent,
    # every one of them receipted in its own DesignRun -- not one extra, uncounted draw. Reuses round 1's
    # own `seed` (a different RandomState-consuming call than `optimize_under_oracle`'s BayesianOptimizer,
    # so this does not replay round 1's sequence); `fresh_pool` below deliberately uses `seed + 1` so the
    # round-2 candidate pool's draw never repeats this baseline's draw.
    baseline_xs = random_design(bounds, budget, seed=seed)
    baseline_run = DesignRun(oracle_name=oracle.name, oracle_tier=oracle.tier, oracle_fidelity=oracle.fidelity)
    for x in baseline_xs:
        baseline_run.history.append(DesignCandidate(x=x, result=oracle(x)))
    baseline = AmplificationRound(
        run=baseline_run, best_score=float(baseline_run.best.result.score), xs=[c.x for c in baseline_run.history]
    )

    # Statistical improvement test, not a point comparison: is round 1's best score bigger than the
    # baseline's best score by more than a fair reshuffling of the pooled `2 * budget` scores would
    # typically produce? A permutation test answers exactly this without needing to actually re-run the
    # search: it repeatedly re-splits the `2 * budget` already-collected scores (every oracle call from
    # both sides) into two same-sized groups uniformly at random and asks how often a random split's
    # "max(group A) - max(group B)" is at least as large as what was actually observed. Under a genuinely
    # zero-skill search (its proposals no better than random), round 1's and the baseline's scores are
    # exchangeable and this p-value is ~Uniform(0, 1), so `beats_baseline` fires at the nominal
    # `significance` rate rather than approaching certainty as `budget` grows -- unlike a bare
    # `round1.best_score > baseline.best_score` check, which is exactly the old, unmatched-budget bug's
    # comparison with none of its calibration.
    permutation = permutation_test(
        (run1.scores(), baseline_run.scores()),
        _best_score_gap,
        permutation_type="independent",
        alternative="greater",
        n_resamples=9999,
        rng=seed,
    )
    baseline_p_value = float(permutation.pvalue)
    beats_baseline = bool(baseline_p_value < significance)
    if not beats_baseline:
        return AmplifyReport(
            round1=round1,
            round2=None,
            baseline=baseline,
            baseline_p_value=baseline_p_value,
            significance=significance,
            beats_baseline=False,
            round2_beats_round1=False,
            incumbent_best_score=round1.best_score,
            collapse=None,
            student=None,
            stopped_early=True,
            reason=(
                "the amplified teacher (round 1 search) did not beat a budget-matched random baseline "
                f"(one-sided permutation test p={baseline_p_value:.4g}, significance={significance}); "
                "nothing to capture"
            ),
        )

    student = fit_student(run1, degree=degree)

    # MXR-080-0162 fix: the round 2 pool is FRESH candidates only -- round 1's own points are never
    # unioned in. Including them (the earlier code did) spends round 2's budget re-verifying points
    # already scored in round 1 and, since the student is fit on exactly those points and the oracle is
    # re-queried at the same x, tautologically floors round2.best_score at round1.best_score: that is a
    # retained incumbent smuggled into the evaluation budget, not evidence round 2 proposed anything
    # better. `round1`'s best point is still available -- see `incumbent_best_score` below -- just kept
    # separate from the budget spent testing the student on candidates it has not seen.
    fresh_pool = random_design(bounds, int(candidate_pool_size), seed=None if seed is None else seed + 1)
    predicted = np.asarray([student(x) for x in fresh_pool])
    top_idx = np.argsort(-predicted)[:budget]

    run2 = DesignRun(oracle_name=oracle.name, oracle_tier=oracle.tier, oracle_fidelity=oracle.fidelity)
    for idx in top_idx:
        x = fresh_pool[idx]
        result = oracle(x)  # the oracle supplies the accepted score; the student only proposes where to look
        run2.history.append(DesignCandidate(x=x, result=result))
    round2 = AmplificationRound(run=run2, best_score=float(run2.best.result.score), xs=[c.x for c in run2.history])

    from mixle.task.collapse import collapse_monitor

    # Both scores below come from fresh-candidate rounds now (round 1 from its own search, round 2 from
    # the student's ranking of candidates round 1 never saw), so this is a genuine generalization
    # trajectory, not one propped up by a retained incumbent re-appearing in round 2's pool.
    collapse = collapse_monitor(
        [
            {"score": round1.best_score, "candidates": round1.xs},
            {"score": round2.best_score, "candidates": round2.xs},
        ]
    )

    return AmplifyReport(
        round1=round1,
        round2=round2,
        baseline=baseline,
        baseline_p_value=baseline_p_value,
        significance=significance,
        beats_baseline=True,
        round2_beats_round1=round2.best_score >= round1.best_score,
        incumbent_best_score=max(round1.best_score, round2.best_score),
        collapse=collapse,
        student=student,
        stopped_early=False,
        reason=None,
    )
