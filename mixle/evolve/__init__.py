"""``mixle.evolve`` -- the self-improvement core: measure -> propose -> verify -> promote.

A pure, serving-agnostic *orchestration algebra* over capabilities that already exist in the library.
It adds no new modeling capability: the proper scores and calibration diagnostics
(:mod:`mixle.inference.scoring` / :mod:`mixle.inference.calibration`), the comparison tests
(:mod:`mixle.inference.model_comparison`), the fitters (:func:`mixle.inference.estimation.optimize`,
the streaming estimators, :func:`mixle.utils.automatic.get_estimator`), and the decision layer
(:func:`mixle.inference.decision.bayes_action`) are all pre-existing. ``evolve`` wires them into a
single anti-regression loop.

Phase 1 surface:

* **measure** -- :class:`Objective` + ``nll`` / ``log_score`` / ``crps`` / ``interval`` / ``calibration``
  / ``decision_regret`` builders.
* **propose** -- :class:`ImprovementOperator` + :class:`Refit`, :class:`OnlineUpdate`,
  :class:`AutoSelect`, :class:`Recalibrate`, a scoped operator registry.
* **verify** -- :func:`challenger_beats_champion` + :class:`Verdict` (the anti-regression gate).
* **drive** -- :func:`improve` + :class:`ImprovementResult`, :func:`auto_select`, and the
  :class:`EvolutionLedger` telemetry.

``search`` / ``Space`` / ``Population`` / ``OperatorBandit`` and the ``Recompose`` / ``Mutate`` operators
are Phase 2-4 and are deliberately not exported here.
"""

from __future__ import annotations

from mixle.evolve.improve import ImprovementResult, improve
from mixle.evolve.ledger import EvolutionLedger
from mixle.evolve.objective import (
    Objective,
    calibration_objective,
    crps_objective,
    decision_regret_objective,
    interval_objective,
    log_score_objective,
    nll_objective,
)
from mixle.evolve.operators import (
    AutoSelect,
    Candidate,
    ImprovementOperator,
    OnlineUpdate,
    Recalibrate,
    Refit,
    default_operators,
    register_operator,
    registered_operators,
    unregister_operator,
)
from mixle.evolve.search import auto_select
from mixle.evolve.verify import Verdict, challenger_beats_champion

__all__ = [
    # one-shot loop
    "improve",
    "ImprovementResult",
    # data -> better mixle model (held-out gated)
    "auto_select",
    # measure
    "Objective",
    "nll_objective",
    "log_score_objective",
    "crps_objective",
    "interval_objective",
    "calibration_objective",
    "decision_regret_objective",
    # verify gate
    "challenger_beats_champion",
    "Verdict",
    # propose contract + operators
    "ImprovementOperator",
    "Candidate",
    "Refit",
    "OnlineUpdate",
    "AutoSelect",
    "Recalibrate",
    "register_operator",
    "unregister_operator",
    "registered_operators",
    "default_operators",
    # telemetry
    "EvolutionLedger",
]
