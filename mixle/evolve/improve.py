"""The one-shot self-improvement driver: propose -> verify -> pick the best verified win (or nothing).

``improve`` is the minimal closed loop. It splits the data once into a train and a held-out verify
split, proposes a challenger from every applicable operator on the train split, gates each challenger
against the champion on the verify split, and returns the verified challenger with the largest delta
(ties broken toward the lower-cost operator). If nothing verifies it returns the *unchanged* champion with
``verified=False`` -- the anti-regression guarantee: ``improve`` can never hand back a worse model.

Every attempt is recorded to the ledger, so the run leaves a serializable trail of what was
tried and why it won or lost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mixle.evolve.ledger import EvolutionLedger
from mixle.evolve.objective import Objective
from mixle.evolve.operators import ImprovementOperator, default_operators
from mixle.evolve.verify import Verdict, challenger_beats_champion


@dataclass(frozen=True)
class ImprovementResult:
    """The outcome of an :func:`improve` run."""

    model: Any  # the promoted challenger, or the unchanged champion if none verified
    verified: bool
    operator: str | None
    delta: float
    verdict: Verdict | None
    evidence: dict = field(default_factory=dict)
    parent_hash: str | None = None


def _split(data: Sequence[Any], holdout: float | tuple, seed: int) -> tuple[list[Any], list[Any]]:
    """Split ``data`` into (train, verify). A ``(train, verify)`` tuple is passed through verbatim."""
    if isinstance(holdout, tuple):
        train, verify = holdout
        return list(train), list(verify)
    rows = list(data)
    n = len(rows)
    if n < 4:
        raise ValueError(f"improve needs at least 4 observations to split; got {n}.")
    if not 0.0 < holdout < 1.0:
        raise ValueError("holdout must be in (0, 1) or an explicit (train, verify) tuple.")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_verify = max(1, int(round(holdout * n)))
    n_verify = min(n_verify, n - 1)
    verify_idx = perm[:n_verify]
    train_idx = perm[n_verify:]
    return [rows[i] for i in train_idx], [rows[i] for i in verify_idx]


def improve(
    model: Any,
    data: Sequence[Any],
    *,
    objective: Objective,
    operators: Sequence[ImprovementOperator] | None = None,
    holdout: float | tuple = 0.25,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    budget: float | None = None,
    seed: int = 0,
    ledger: EvolutionLedger | None = None,
    parent_hash: str | None = None,
    require_calibration: bool = True,
) -> ImprovementResult:
    """Propose challengers, gate them, and return the best verified improvement (or the champion).

    Args:
        model: the champion (a fitted mixle distribution).
        data: the raw held-out dataset (split once into train/verify here).
        objective: the :class:`~mixle.evolve.objective.Objective` to improve on.
        operators: the proposal moves; defaults to the applicable subset of
            ``[Refit, OnlineUpdate, AutoSelect, Recalibrate]``.
        holdout: train/verify split fraction, or an explicit ``(train, verify)`` tuple.
        alpha: significance level for the verify gate.
        min_effect: practical effect-size floor passed to the gate.
        budget: optional cost ceiling -- operators whose ``cost_hint`` exceeds the remaining budget are
            skipped after ascending-cost ordering.
        seed: RNG seed for the split and the sampled objectives.
        ledger: optional :class:`~mixle.evolve.ledger.EvolutionLedger` to record every attempt into.
        parent_hash: optional lineage hash for the champion (carried onto candidates and ledger rows).
        require_calibration: forward to the gate's calibration no-regression check.

    Returns:
        An :class:`ImprovementResult`. ``result.verified is True`` guarantees ``result.model`` beat the
        champion significantly and non-regressively on the verify split.
    """
    ops = list(operators) if operators is not None else default_operators()
    train, verify = _split(data, holdout, seed)
    ctx = {"parent_hash": parent_hash, "seed": seed, "objective": objective}

    # Cost-aware ordering: lower-cost operators first so a budget cuts the expensive tail.
    ops = sorted(ops, key=lambda op: getattr(op, "cost_hint", 1.0))

    best: ImprovementResult | None = None
    spent = 0.0
    for op in ops:
        cost = float(getattr(op, "cost_hint", 1.0))
        if budget is not None and spent + cost > budget:
            continue
        try:
            if not op.applicable(model, train, ctx=ctx):
                continue
            candidate = op.propose(model, train, ctx=ctx)
        except Exception as exc:
            if ledger is not None:
                ledger.record(
                    operator=op.name,
                    delta=0.0,
                    verdict=None,
                    cost=cost,
                    parent_hash=parent_hash,
                    meta={"error": str(exc)},
                )
            continue
        spent += cost

        # a family swap (AutoSelect / a new family) gets the non-nested robustness cross-check.
        nonnested = type(candidate.model).__name__ != type(model).__name__
        verdict = challenger_beats_champion(
            model,
            candidate.model,
            verify,
            objective=objective,
            alpha=alpha,
            min_effect=min_effect,
            require_calibration=require_calibration,
            nonnested=nonnested,
            seed=seed,
        )
        if ledger is not None:
            ledger.record(
                operator=op.name,
                delta=verdict.delta,
                verdict=verdict.as_dict(),
                cost=cost,
                parent_hash=parent_hash,
                meta=candidate.meta,
            )

        if verdict.promote:
            cand_result = ImprovementResult(
                candidate.model,
                True,
                op.name,
                verdict.delta,
                verdict,
                {"candidate_meta": candidate.meta, "cost": cost},
                parent_hash,
            )
            # keep the largest verified delta; ties -> the cheaper operator (already cost-sorted, so
            # the first to reach a given delta is the cheaper one).
            if best is None or verdict.delta > best.delta:
                best = cand_result

    if best is not None:
        return best
    return ImprovementResult(model, False, None, 0.0, None, {"reason": "no verified improvement"}, parent_hash)


__all__ = ["ImprovementResult", "improve"]
