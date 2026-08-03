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

from mixle.doe.designs import Bounds, _as_bounds, _require_exact_positive_int, random_design
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


def _genuine_scores(run: DesignRun) -> np.ndarray:
    """Return ``run``'s oracle scores restricted to genuine (non-abstained) observations.

    Mirrors ``DesignRun.scores()`` but filters through ``genuine_history`` first. ``DesignRun.history``
    (and ``.scores()``, deliberately -- see its docstring) keep every attempted candidate, abstentions
    included, for the receipted record. But an abstention's placeholder ``-inf`` (``OracleResult.
    abstained``; e.g. a timeout) is not a real draw from the oracle's score distribution, and a
    permutation test assumes its inputs ARE exchangeable draws from "the oracle's score at a proposed
    candidate" -- feeding it a fabricated ``-inf`` silently changes the pooled sample it resamples from
    (and hence the resampling null distribution and p-value) exactly whenever a call times out, the same
    class of contamination MXR-080-0189 fixed for ``optimize_under_oracle``'s GP fit, just one boundary
    further downstream. ``np.max`` itself ignores a lone ``-inf`` whenever a finite score is also
    present, which is why this is easy to miss from a single observed statistic value alone.
    """
    return np.asarray([c.result.score for c in run.genuine_history], dtype=float)


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
    """Fit :class:`StudentTeacher` from ``run``'s GENUINE (non-abstained) oracle-verified history.

    Fits against ``run.genuine_history``, not ``run.history``: an abstained candidate's placeholder
    ``-inf`` score (``OracleResult.abstained`` -- e.g. a timeout) is not a real observation of the
    oracle's objective, and least-squares regression has no tolerance for a non-finite target at all --
    unlike the permutation test in ``amplify_and_capture``, where a lone ``-inf`` only nudges a p-value,
    here a SINGLE abstained candidate anywhere in ``run.history`` drives every coefficient in ``coef`` to
    ``nan`` (confirmed: a clean fit against 8 well-posed genuine points, then adding one abstained
    candidate to the same history turns the entire coefficient vector to ``nan``), silently turning the
    captured student that guides round 2's candidate ranking into a useless, uniformly-``nan`` predictor
    with no visible error at the call site.
    """
    degree = _require_exact_positive_int(degree, "degree", minimum=0)
    genuine = run.genuine_history
    if not genuine:
        raise ValueError(
            f"cannot fit a student: all {len(run.history)} candidate(s) in the run history abstained "
            "(e.g. every oracle call timed out); there is no genuine, verified observation to fit a "
            "surrogate to."
        )
    xs = np.stack([c.x for c in genuine]).astype(np.float64)
    ys = np.asarray([c.result.score for c in genuine], dtype=np.float64)
    x_design = _design_matrix(xs, degree)
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)) or not np.all(np.isfinite(x_design)):
        raise ValueError("cannot fit a student from non-finite verified design evidence.")
    n_parameters = x_design.shape[1]
    rank = int(np.linalg.matrix_rank(x_design))
    if x_design.shape[0] < n_parameters or rank < n_parameters:
        raise ValueError(
            "cannot fit an identifiable student: "
            f"{x_design.shape[0]} genuine observations provide rank {rank} for {n_parameters} coefficients."
        )
    coef, *_ = np.linalg.lstsq(x_design, ys, rcond=None)
    if not np.all(np.isfinite(coef)):
        raise ValueError("student fit produced non-finite coefficients.")
    return StudentTeacher(coef=coef, degree=degree)


@dataclass
class AmplificationRound:
    """One oracle-verified amplification round and its best observed score."""

    run: DesignRun
    best_score: float | None
    xs: list[np.ndarray]


@dataclass(frozen=True)
class AmplificationSeeds:
    """Concrete independent random streams used by an amplification receipt."""

    search: int
    baseline: int
    permutation: int
    round2_pool: int


@dataclass(frozen=True)
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
    baseline_p_value: float | None
    significance: float
    beats_baseline: bool
    round2_beats_round1: bool
    incumbent_best_score: float | None
    collapse: CollapseVerdict | None
    student: StudentTeacher | None
    stopped_early: bool
    reason: str | None
    seeds: AmplificationSeeds
    round1_effective_n: int
    baseline_effective_n: int
    round2_effective_n: int | None


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
    b = _as_bounds(bounds)
    n_init = _require_exact_positive_int(n_init, "n_init")
    n_iter = _require_exact_positive_int(n_iter, "n_iter", minimum=0)
    candidate_pool_size = _require_exact_positive_int(candidate_pool_size, "candidate_pool_size")
    degree = _require_exact_positive_int(degree, "degree", minimum=0)
    budget = n_init + n_iter
    if candidate_pool_size < budget:
        raise ValueError(
            f"candidate_pool_size must be at least the matched evaluation budget {budget}, got {candidate_pool_size}."
        )
    try:
        significance = float(significance)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"significance must be a finite probability in (0, 1), got {significance!r}.") from exc
    if not np.isfinite(significance) or not (0.0 < significance < 1.0):
        raise ValueError(f"significance must be a finite probability in (0, 1), got {significance!r}.")

    root_sequence = np.random.SeedSequence(seed)
    child_sequences = root_sequence.spawn(4 if seed is None else 3)
    search_seed = int(child_sequences[0].generate_state(1, dtype=np.uint32)[0]) if seed is None else int(seed)
    offset = 1 if seed is None else 0
    seeds = AmplificationSeeds(
        search=search_seed,
        baseline=int(child_sequences[offset].generate_state(1, dtype=np.uint32)[0]),
        permutation=int(child_sequences[offset + 1].generate_state(1, dtype=np.uint32)[0]),
        round2_pool=int(child_sequences[offset + 2].generate_state(1, dtype=np.uint32)[0]),
    )

    run1 = optimize_under_oracle(oracle, b, n_init=n_init, n_iter=n_iter, seed=seeds.search)
    run1_scores = _genuine_scores(run1)
    round1 = AmplificationRound(
        run=run1,
        best_score=float(np.max(run1_scores)) if run1_scores.size else None,
        xs=[c.x for c in run1.history],
    )

    # Budget-matched baseline (MXR-080-0161): the same number of oracle-verified draws as round 1 spent,
    # every one receipted in its own DesignRun. Its recorded child seed is independent of the search,
    # permutation, and round-2 streams; restarting both search and baseline from the same integer seed
    # can replay the identical first point and invalidate the comparison.
    baseline_xs = random_design(b, budget, seed=seeds.baseline)
    baseline_run = DesignRun(oracle_name=oracle.name, oracle_tier=oracle.tier, oracle_fidelity=oracle.fidelity)
    for x in baseline_xs:
        # Every draw is recorded, abstained or not (mirrors optimize_under_oracle's own "always record,
        # selectively use" pattern) -- best_score above already goes through DesignRun.best, which is
        # abstention-safe; the permutation test below is the other consumer of this run's scores, and is
        # made abstention-safe via _genuine_scores() rather than here.
        baseline_run.append(DesignCandidate(x=x, result=oracle(x)))
    if baseline_run.oracle_calls != budget:
        raise RuntimeError(f"baseline executed {baseline_run.oracle_calls} calls; expected exactly {budget}.")
    baseline_scores = _genuine_scores(baseline_run)
    baseline = AmplificationRound(
        run=baseline_run,
        best_score=float(np.max(baseline_scores)) if baseline_scores.size else None,
        xs=[c.x for c in baseline_run.history],
    )

    def stopped_report(
        reason: str,
        *,
        p_value: float | None = None,
        gate_passed: bool = False,
    ) -> AmplifyReport:
        return AmplifyReport(
            round1=round1,
            round2=None,
            baseline=baseline,
            baseline_p_value=p_value,
            significance=significance,
            beats_baseline=gate_passed,
            round2_beats_round1=False,
            incumbent_best_score=round1.best_score,
            collapse=None,
            student=None,
            stopped_early=True,
            reason=reason,
            seeds=seeds,
            round1_effective_n=int(run1_scores.size),
            baseline_effective_n=int(baseline_scores.size),
            round2_effective_n=None,
        )

    if run1_scores.size < 2 or baseline_scores.size < 2:
        return stopped_report(
            "insufficient genuine observations for the baseline permutation test: "
            f"round1={run1_scores.size}, baseline={baseline_scores.size}; at least 2 per group are required"
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
    #
    # Both sides feed _genuine_scores(), not .scores(): an abstained oracle call (e.g. a timeout; see
    # OracleResult.abstained) is a placeholder -inf, not a real draw from either distribution, and
    # exchangeability -- the assumption the whole test leans on -- only holds among genuine observations.
    # run1.history and baseline_run.history both still keep every attempted candidate, abstained or not,
    # for the receipted record; this simply is not the right input for a test that assumes its inputs are
    # exchangeable draws from the same distribution.
    permutation = permutation_test(
        (run1_scores, baseline_scores),
        _best_score_gap,
        permutation_type="independent",
        alternative="greater",
        n_resamples=9999,
        rng=seeds.permutation,
    )
    baseline_p_value = float(permutation.pvalue)
    if not np.isfinite(baseline_p_value) or not (0.0 <= baseline_p_value <= 1.0):
        raise ValueError(f"permutation test produced invalid p-value {baseline_p_value!r}.")
    beats_baseline = bool(baseline_p_value < significance)
    if not beats_baseline:
        return stopped_report(
            (
                "the amplified teacher (round 1 search) did not beat a budget-matched random baseline "
                f"(one-sided permutation test p={baseline_p_value:.4g}, significance={significance}); "
                "nothing to capture"
            ),
            p_value=baseline_p_value,
        )

    try:
        student = fit_student(run1, degree=degree)
    except ValueError as exc:
        return stopped_report(
            f"baseline gate passed but student evidence is insufficient: {exc}",
            p_value=baseline_p_value,
            gate_passed=True,
        )

    # MXR-080-0162 fix: the round 2 pool is FRESH candidates only -- round 1's own points are never
    # unioned in. Including them (the earlier code did) spends round 2's budget re-verifying points
    # already scored in round 1 and, since the student is fit on exactly those points and the oracle is
    # re-queried at the same x, tautologically floors round2.best_score at round1.best_score: that is a
    # retained incumbent smuggled into the evaluation budget, not evidence round 2 proposed anything
    # better. `round1`'s best point is still available -- see `incumbent_best_score` below -- just kept
    # separate from the budget spent testing the student on candidates it has not seen.
    fresh_pool = random_design(b, candidate_pool_size, seed=seeds.round2_pool)
    round1_keys = {np.asarray(point, dtype=np.float64).tobytes() for point in round1.xs}
    fresh_pool = np.asarray(
        [point for point in fresh_pool if np.asarray(point, dtype=np.float64).tobytes() not in round1_keys],
        dtype=np.float64,
    ).reshape(-1, b.shape[0])
    if fresh_pool.shape[0] < budget:
        raise ValueError(
            f"fresh candidate pool contains only {fresh_pool.shape[0]} points after excluding round 1; "
            f"{budget} are required."
        )
    predicted = np.asarray([student(x) for x in fresh_pool], dtype=np.float64)
    if not np.all(np.isfinite(predicted)):
        raise ValueError("student produced non-finite candidate rankings.")
    top_idx = np.argsort(-predicted)[:budget]
    if top_idx.size != budget:
        raise RuntimeError(f"round 2 selected {top_idx.size} candidates; expected exactly {budget}.")

    run2 = DesignRun(oracle_name=oracle.name, oracle_tier=oracle.tier, oracle_fidelity=oracle.fidelity)
    for idx in top_idx:
        x = fresh_pool[idx]
        result = oracle(x)  # the oracle supplies the accepted score; the student only proposes where to look
        run2.append(DesignCandidate(x=x, result=result))
    if run2.oracle_calls != budget:
        raise RuntimeError(f"round 2 executed {run2.oracle_calls} calls; expected exactly {budget}.")
    run2_scores = _genuine_scores(run2)
    round2 = AmplificationRound(
        run=run2,
        best_score=float(np.max(run2_scores)) if run2_scores.size else None,
        xs=[c.x for c in run2.history],
    )
    if round2.best_score is None:
        return AmplifyReport(
            round1=round1,
            round2=round2,
            baseline=baseline,
            baseline_p_value=baseline_p_value,
            significance=significance,
            beats_baseline=True,
            round2_beats_round1=False,
            incumbent_best_score=round1.best_score,
            collapse=None,
            student=student,
            stopped_early=True,
            reason="round 2 produced no genuine oracle observations",
            seeds=seeds,
            round1_effective_n=int(run1_scores.size),
            baseline_effective_n=int(baseline_scores.size),
            round2_effective_n=0,
        )

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
        seeds=seeds,
        round1_effective_n=int(run1_scores.size),
        baseline_effective_n=int(baseline_scores.size),
        round2_effective_n=int(run2_scores.size),
    )
