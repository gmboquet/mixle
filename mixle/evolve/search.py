"""``auto_select``: the "propose a better mixle model from data" front door (Phase 1).

Phase 1 ships only :func:`auto_select`. It elevates the existing automatic engine
(:func:`mixle.utils.automatic.get_estimator`) into the evolve contract and -- when the criterion is a
proper-score :class:`~mixle.evolve.objective.Objective` rather than an information criterion -- adds the
held-out champion/challenger gate on top of the in-sample BIC pick, so the returned model is the one
that wins *out of sample*, not merely the lowest-BIC one.

``search()`` / ``Space`` / the BO and evolutionary backends are Phase 2+ and are intentionally not built
here; the seam is left clean (this module exposes only ``auto_select``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mixle.evolve.improve import ImprovementResult, _split
from mixle.evolve.objective import Objective
from mixle.evolve.operators import Refit
from mixle.evolve.verify import challenger_beats_champion


def _fit_auto(rows: list[Any], *, max_its: int) -> Any:
    """BIC/auto family inference + EM fit (the in-sample automatic pick)."""
    from mixle.inference.estimation import optimize
    from mixle.utils.automatic import get_estimator

    estimator = get_estimator(rows)
    return optimize(rows, estimator, max_its=max_its, out=None)


def auto_select(
    data: Sequence[Any],
    *,
    space: Any | None = None,
    criterion: str | Objective = "bic",
    verify: bool = True,
    holdout: float = 0.25,
    seed: int = 0,
    max_its: int = 20,
) -> ImprovementResult:
    """Infer and fit a model from raw ``data``, optionally gated by a held-out proper score.

    Args:
        data: the raw dataset.
        space: reserved for the Phase-2 typed search space; must be ``None`` in Phase 1.
        criterion: ``'bic'`` (delegate to the automatic in-sample pick) or a proper-score
            :class:`~mixle.evolve.objective.Objective` (add the held-out verify gate on top of BIC).
        verify: when ``criterion`` is an :class:`Objective`, whether to run the held-out gate (the BIC
            pick fitted on the train split is the *champion*; the BIC pick refitted on all data is the
            *challenger*, promoted only if it wins out of sample).
        holdout: held-out fraction for the proper-score gate.
        seed: RNG seed for the split and sampled objectives.
        max_its: EM iterations for the fits.

    Returns:
        An :class:`~mixle.evolve.improve.ImprovementResult`. For ``criterion='bic'`` it carries the
        fitted automatic model with ``verified=False`` (no out-of-sample test was requested). For an
        :class:`Objective` criterion with ``verify=True`` it carries the gate verdict and
        ``verified`` reflects whether the full-data model beats the train-only model out of sample.
    """
    if space is not None:
        raise NotImplementedError("auto_select: a typed search 'space' is a Phase-2 feature; pass space=None.")

    rows = list(data)

    if isinstance(criterion, str):
        if criterion != "bic":
            raise ValueError(
                f"string criterion must be 'bic' (got {criterion!r}); pass a proper-score Objective for "
                "out-of-sample selection."
            )
        model = _fit_auto(rows, max_its=max_its)
        return ImprovementResult(
            model,
            False,
            "auto_select[bic]",
            0.0,
            None,
            {"criterion": "bic", "family": type(model).__name__},
            None,
        )

    # proper-score Objective: BIC pick + held-out gate.
    objective: Objective = criterion
    if not verify:
        model = _fit_auto(rows, max_its=max_its)
        return ImprovementResult(
            model,
            False,
            "auto_select[%s]" % objective.name,
            0.0,
            None,
            {"criterion": objective.name, "verify": False, "family": type(model).__name__},
            None,
        )

    train, val = _split(rows, holdout, seed)
    champion = _fit_auto(train, max_its=max_its)
    # the challenger is the same automatic family warm-fitted on the full data (more data, same shape).
    challenger = Refit(max_its=max_its).propose(champion, rows, ctx={"parent_hash": None}).model

    verdict = challenger_beats_champion(
        champion,
        challenger,
        val,
        objective=objective,
        seed=seed,
    )
    if verdict.promote:
        return ImprovementResult(
            challenger,
            True,
            "auto_select[%s]" % objective.name,
            verdict.delta,
            verdict,
            {"criterion": objective.name, "family": type(challenger).__name__},
            None,
        )
    # the full-data fit did not beat the train-only fit out of sample -> keep the more-evidenced full fit
    # but report it as unverified (no out-of-sample improvement over the train-only model).
    full = _fit_auto(rows, max_its=max_its)
    return ImprovementResult(
        full,
        False,
        "auto_select[%s]" % objective.name,
        verdict.delta,
        verdict,
        {"criterion": objective.name, "family": type(full).__name__, "verified_gate": False},
        None,
    )


__all__ = ["auto_select"]
